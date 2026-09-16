"""Stage 1: train a world model on the offline cups-and-ball data.

    pixi run train-wm                                          # configs/dreamer/train.yaml
    pixi run train-wm trainer.steps=1000 wandb.mode=offline
    pixi run train-wm logdir=<run dir> trainer.resume=auto     # continue a run
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

from world_model.training.loop import to_device, train_loop  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402
from world_model.utils.logger import Logger  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402


def resume_path(resume, checkpoints):
    """None, "auto" (latest checkpoint in this run, if any) or an explicit path."""
    if resume is None:
        return None
    if resume == "auto":
        return checkpoints.latest()
    return Path(resume)


@hydra.main(version_base=None, config_path="../configs/dreamer", config_name="train")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")  # as in r2dreamer's train.py
    logdir = Path(cfg.logdir)
    checkpoints = CheckpointManager(logdir / "checkpoints", "wm")
    ckpt = resume_path(cfg.trainer.resume, checkpoints)
    state = CheckpointManager.load(ckpt) if ckpt else None

    train_set, holdout_set = instantiate(cfg.data)
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

    loader_kwargs = dict(batch_size=cfg.loader.batch_size, num_workers=cfg.loader.num_workers, pin_memory=True,
                         persistent_workers=cfg.loader.num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)  # fixed shapes for compile
    holdout_loader = DataLoader(holdout_set, shuffle=False, drop_last=False, **loader_kwargs)

    best_metric = str(cfg.trainer.best_metric)
    best_value = state.get("best_holdout", float("inf")) if state else float("inf")

    def evaluate(step):
        nonlocal best_value
        totals, count = {}, 0
        for batch in holdout_loader:
            n = batch["is_first"].shape[0]
            for k, v in model.eval_step(to_device(batch, cfg.device)).items():
                totals[k] = totals.get(k, 0.0) + float(v) * n
            count += n
        values = {k: v / count for k, v in totals.items()}
        logger.scalars(step, {f"holdout/{k}": v for k, v in values.items()})
        if values[best_metric] < best_value:
            best_value = values[best_metric]
            path = checkpoints.save("best", state_fn(step))
            print(f"[{step}] new best holdout/{best_metric}={best_value:.4g} -> {path}", flush=True)

    def state_fn(step):
        return {"kind": "world_model", "step": step, "config": config, "model": model.train_state_dict(),
                "best_holdout": best_value, "wandb_run_id": logger.run_id}

    print(f"train {len(train_set)} windows x {train_set.window} steps | holdout {len(holdout_set)} | "
          f"action_dim {train_set.action_dim} | logdir {logdir}")
    train_loop(model.train_step, train_loader, cfg.device, logger, checkpoints, state_fn,
               steps=int(cfg.trainer.steps), log_every=int(cfg.trainer.log_every),
               ckpt_every=int(cfg.trainer.ckpt_every), start_step=start_step,
               callbacks=[(int(cfg.trainer.eval_every), evaluate)])
    logger.close()


if __name__ == "__main__":
    main()
