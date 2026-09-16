"""Stage 2: fit a decoder on a frozen stage-1 world model and log open-loop rollouts.

    pixi run train-decoder wm_run=logdir/world_model/r2dreamer/<timestamp>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, default_collate  # noqa: E402

from world_model.training.loop import to_device, train_loop  # noqa: E402
from world_model.training.readout import DecoderReadout  # noqa: E402
from world_model.training.rollout import action_onsets, open_loop_rollout  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402
from world_model.utils.logger import Logger  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402

from train_world_model import resume_path  # noqa: E402


@hydra.main(version_base=None, config_path="../configs/dreamer", config_name="decoder")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")
    logdir = Path(cfg.logdir)

    # data and model exactly as in stage 1, rebuilt from the config stored in its checkpoint
    wm_state = CheckpointManager.load(cfg.wm_checkpoint)
    wm_cfg = OmegaConf.create(wm_state["config"])
    train_set, holdout_set = instantiate(wm_cfg.data)
    model = instantiate(wm_cfg.model, action_dim=train_set.action_dim, device=cfg.device, _recursive_=False)
    model.load_train_state_dict(wm_state["model"])
    readout = DecoderReadout(model, cfg.decoder.lr)

    checkpoints = CheckpointManager(logdir / "checkpoints", "decoder")
    ckpt = resume_path(cfg.trainer.resume, checkpoints)
    state = CheckpointManager.load(ckpt) if ckpt else None
    start_step = 0
    if state is not None:
        readout.load_state_dict(state["readout"])
        start_step = int(state["step"])
        print(f"resumed from {Path(ckpt).resolve()} at step {start_step}")

    config = OmegaConf.to_container(cfg, resolve=True)
    config["world_model"] = wm_state["config"]
    logger = Logger(logdir, config, cfg.wandb, job_type="decoder",
                    name=f"{wm_cfg.model.name}/{Path(cfg.wm_run).name}/decoder", group=wm_cfg.wandb.group,
                    run_id=state["wandb_run_id"] if state else None, tensorboard=cfg.tensorboard)

    loader = DataLoader(train_set, batch_size=cfg.loader.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.loader.num_workers, pin_memory=True,
                        persistent_workers=cfg.loader.num_workers > 0)

    def clips(dataset, n):
        whole = dataset.with_window(None)
        count = min(int(n), len(whole))
        return to_device(default_collate([whole[i] for i in range(count)]), cfg.device) if count else None

    # rollout videos are optional: trainer.video_every <= 0 trains the decoder without them,
    # and a split with no episodes (e.g. no holdout) simply contributes no clips
    video_every = int(cfg.trainer.video_every)
    rollout_clips = {} if video_every <= 0 else {
        name: batch for name, batch in (("holdout", clips(holdout_set, cfg.rollout.episodes)),
                                        ("train", clips(train_set, cfg.rollout.episodes))) if batch is not None
    }

    def log_rollouts(step):
        for split, batch in rollout_clips.items():
            panel, metrics = open_loop_rollout(model, readout.decoder, batch, int(cfg.rollout.context))
            logger.video(step, f"rollout/{split}", panel, fps=train_set.arrays.fps)
            logger.scalars(step, {f"rollout/{split}/{k}": v for k, v in metrics.items()})

    def state_fn(step):
        return {"kind": "decoder", "step": step, "config": config,
                "wm_checkpoint": str(Path(cfg.wm_checkpoint).resolve()), "readout": readout.state_dict(),
                "wandb_run_id": logger.run_id}

    print(f"decoder on frozen {Path(cfg.wm_checkpoint).resolve()} | logdir {logdir}")
    train_loop(readout.step, loader, cfg.device, logger, checkpoints, state_fn, steps=int(cfg.trainer.steps),
               log_every=int(cfg.trainer.log_every), ckpt_every=int(cfg.trainer.ckpt_every),
               start_step=start_step, callbacks=[(video_every, log_rollouts)] if rollout_clips else [])
    logger.close()


if __name__ == "__main__":
    main()
