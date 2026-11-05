"""PHASE 1 stage 1: vector memory, prior/posterior updaters, reward-only decoder.

Mirrors scripts/train_smt_prelim.py: the DataLoader is built from cfg.data_anchors (one item
per timestep anchor), while cfg.data still holds the STANDARD windowed image dataset and is
carried into the checkpoint, so train_readout_phase1.py, train_decoder.py and eval_memtest1.py
rebuild ordinary EpisodeWindows from it and need no changes.

    pixi run train-smt-phase1
    pixi run train-smt-phase1 trainer.steps=0          # emit an untrained checkpoint
    pixi run train-smt-phase1 model.eval_warmup=60     # the smt_prelim-equivalent control
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
import numpy as np  # noqa: E402
from torch.utils.data import DataLoader, Sampler  # noqa: E402

from train_world_model import resume_path  # noqa: E402
from world_model.models.smt_prelim.embed_cache import read_embed_attrs  # noqa: E402
from world_model.training.loop import to_device, train_loop  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402
from world_model.utils.logger import Logger  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402


class PairBatchSampler(Sampler):
    """Batches that carry positive pairs (align_label: pairs): `per_batch` random pairs, both partners as
    anchors, and the rest of the batch drawn uniformly as usual. Pairs whose frames are not valid anchors of
    the training set are skipped."""

    def __init__(self, train_set, pair_file, batch_size, per_batch, seed, neg_per_batch=0):
        where = {(int(e), int(t)): i for i, (e, t) in enumerate(train_set.index)}
        d = np.load(pair_file)

        def usable(prefix):
            if f"{prefix}x_episode" not in d.files:
                return np.zeros((0, 2), np.int64), 0
            pairs = [(where.get((int(a), int(b))), where.get((int(c), int(e)))) for a, b, c, e in
                     zip(d[f"{prefix}x_episode"], d[f"{prefix}x_t"], d[f"{prefix}y_episode"], d[f"{prefix}y_t"])]
            return np.array([p for p in pairs if None not in p], dtype=np.int64).reshape(-1, 2), len(pairs)
        (self.pairs, n_pos), (self.negs, n_neg) = usable(""), usable("neg_")
        self.n, self.batch_size, self.per_batch = len(train_set), int(batch_size), int(per_batch)
        self.neg_per_batch = int(neg_per_batch) if len(self.negs) else 0
        self.rng = np.random.default_rng(seed)
        print(f"pair sampler: {len(self.pairs)} of {n_pos} positive pairs usable, {self.per_batch} per batch; "
              f"{len(self.negs)} of {n_neg} negative pairs usable, {self.neg_per_batch} per batch")

    def __len__(self):
        return self.n // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            p = self.pairs[self.rng.integers(0, len(self.pairs), self.per_batch)].reshape(-1)
            if self.neg_per_batch:
                p = np.concatenate([p, self.negs[self.rng.integers(0, len(self.negs), self.neg_per_batch)].reshape(-1)])
            rest = self.rng.integers(0, self.n, self.batch_size - len(p))
            yield np.concatenate([p, rest]).tolist()


@hydra.main(version_base=None, config_path="../configs/smt_phase1", config_name="train_memtest4_phase1")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    logdir = Path(cfg.logdir)
    checkpoints = CheckpointManager(logdir / "checkpoints", "wm")
    ckpt = resume_path(cfg.trainer.resume, checkpoints)
    state = CheckpointManager.load(ckpt) if ckpt else None

    # refuse a cache built by a different encoder than the one this model is configured with
    # ORACLE mode: embed_oracle_state is a symbolic observation, not an encoder output, so
    # there is no source checkpoint to match (see scripts/build_oracle_state.py).
    read_embed_attrs(cfg.data_anchors.zarr_path, cfg.model.embed_key,
                     None if bool(cfg.model.get("oracle_obs", False)) else cfg.model.encoder_checkpoint)

    train_set, holdout_set = instantiate(cfg.data_anchors)
    model = instantiate(cfg.model, action_dim=train_set.action_dim, device=cfg.device, _recursive_=False)
    start_step = 0
    if state is not None:
        model.load_train_state_dict(state["model"])
        start_step = int(state["step"])
        print(f"resumed from {Path(ckpt).resolve()} at step {start_step}")

    config = OmegaConf.to_container(cfg, resolve=True)
    logger = Logger(logdir, config, cfg.wandb, job_type="world_model", name=f"{cfg.model.name}/{logdir.name}",
                    group=cfg.wandb.group, run_id=state["wandb_run_id"] if state else None,
                    tensorboard=cfg.tensorboard)

    loader_kwargs = dict(batch_size=cfg.loader.batch_size, num_workers=cfg.loader.num_workers,
                         pin_memory=True, persistent_workers=cfg.loader.num_workers > 0)
    if cfg.loader.get("pair_file"):
        sampler = PairBatchSampler(train_set, cfg.loader.pair_file, cfg.loader.batch_size,
                                   cfg.loader.pairs_per_batch, cfg.seed, cfg.loader.get("neg_pairs_per_batch", 0))
        train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=cfg.loader.num_workers,
                                  pin_memory=True, persistent_workers=cfg.loader.num_workers > 0)
    else:
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    holdout_loader = DataLoader(holdout_set, shuffle=False, drop_last=False, **loader_kwargs)

    best_metric = str(cfg.trainer.best_metric)
    best_value = state.get("best_holdout", float("inf")) if state else float("inf")

    def evaluate(step):
        nonlocal best_value
        totals, count = {}, 0
        for batch in holdout_loader:
            n = batch["t"].shape[0]
            for k, v in model.eval_step(to_device(batch, cfg.device)).items():
                totals[k] = totals.get(k, 0.0) + float(v) * n
            count += n
        if not count:
            return
        values = {k: v / count for k, v in totals.items()}
        logger.scalars(step, {f"holdout/{k}": v for k, v in values.items()})
        if best_metric in values and values[best_metric] < best_value:
            best_value = values[best_metric]
            path = checkpoints.save("best", state_fn(step))
            print(f"[{step}] new best holdout/{best_metric}={best_value:.4g} -> {path}", flush=True)

    def state_fn(step):
        return {"kind": "world_model", "step": step, "config": config, "model": model.train_state_dict(),
                "best_holdout": best_value, "wandb_run_id": logger.run_id}

    if state is None:
        # write a loadable checkpoint before training, so stage 2 can be wired up (and the
        # untrained floor measured) before any SMT machinery has learned anything
        checkpoints.save("latest", state_fn(start_step))

    params = sum(p.numel() for p in model.trainable())
    print(f"train {len(train_set)} anchors | holdout {len(holdout_set)} | ctx {cfg.model.ctx_len} "
          f"fut {cfg.model.fut_len} | eval_warmup {cfg.model.eval_warmup} | "
          f"{params / 1e6:.2f}M trainable | action_dim {train_set.action_dim} | logdir {logdir}")
    train_loop(model.train_step, train_loader, cfg.device, logger, checkpoints, state_fn,
               steps=int(cfg.trainer.steps), log_every=int(cfg.trainer.log_every),
               ckpt_every=int(cfg.trainer.ckpt_every), start_step=start_step,
               callbacks=[(int(cfg.trainer.eval_every), evaluate)])
    logger.close()


if __name__ == "__main__":
    main()
