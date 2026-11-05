"""PRELIMINARY stage 1b: SMT-supervised memory training on cached frame embeddings.

Mirrors scripts/train_world_model.py, with one substitution: the DataLoader is built from
cfg.data_anchors (one item per timestep anchor), while cfg.data still holds the STANDARD
windowed image dataset and is carried into the checkpoint, so train_decoder.py and
eval_memtest1.py rebuild ordinary EpisodeWindows from it and need no changes.

    pixi run train-smt-prelim
    pixi run train-smt-prelim trainer.steps=0          # emit an untrained checkpoint
    pixi run train-smt-prelim model.ctx_len=60         # the full-prefix control run
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from train_world_model import resume_path  # noqa: E402
from world_model.models.smt_prelim.embed_cache import read_embed_attrs  # noqa: E402
from world_model.training.loop import to_device, train_loop  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402
from world_model.utils.logger import Logger  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402


@hydra.main(version_base=None, config_path="../configs/smt_prelim", config_name="train_memtest4_prelim")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    logdir = Path(cfg.logdir)
    checkpoints = CheckpointManager(logdir / "checkpoints", "wm")
    ckpt = resume_path(cfg.trainer.resume, checkpoints)
    state = CheckpointManager.load(ckpt) if ckpt else None

    # refuse a cache built by a different encoder than the one this model is configured with
    read_embed_attrs(cfg.data_anchors.zarr_path, cfg.model.embed_key, cfg.model.encoder_checkpoint)

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
        # write a loadable checkpoint before training: the pixel head is fitted from one of
        # these, and stage 3 fits it before any SMT machinery exists (trainer.steps=0)
        checkpoints.save("latest", state_fn(start_step))

    print(f"train {len(train_set)} anchors | holdout {len(holdout_set)} | ctx {cfg.model.ctx_len} "
          f"fut {cfg.model.fut_len} | action_dim {train_set.action_dim} | logdir {logdir}")
    train_loop(model.train_step, train_loader, cfg.device, logger, checkpoints, state_fn,
               steps=int(cfg.trainer.steps), log_every=int(cfg.trainer.log_every),
               ckpt_every=int(cfg.trainer.ckpt_every), start_step=start_step,
               callbacks=[(int(cfg.trainer.eval_every), evaluate)])
    logger.close()


if __name__ == "__main__":
    main()
