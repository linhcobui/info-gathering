"""PHASE 1 stage 2a: fit StateReadout on FROZEN stage-1 states.

The memory is a plain vector and phase 1 has no embedding predictor, so something must turn an
imagined state back into a frame before the existing pixel head can render it. StateReadout is
that map, and it is fitted here -- after stage 1, on states whose weights no longer move, so it
can never shape the representation.

Regression target: the true (normalised, cached-distribution) embedding at the same step, MSE.
Source states are whatever model.observe() produces, which with observe_mode `recurrent` is the
eval distribution itself: one teacher window at t=0, then the posterior chained over every real
step. `readout.states: window` fits it on teacher encodings instead -- gate 5's ceiling.

Output is a world-model checkpoint carrying the STAGE-1 config verbatim, so train_decoder.py
and eval_memtest1.py rebuild the identical model from it and need no changes. Point their
wm_run at this run directory, not at stage 1.

    pixi run train-readout-phase1 wm_run=logdir/world_model/smt_phase1_memtest4/<ts>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.amp import autocast  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from train_world_model import resume_path  # noqa: E402
from world_model.training.loop import to_device, train_loop  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402
from world_model.utils.logger import Logger  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402


class ReadoutTrainer:
    """MSE from frozen memory states onto the true embedding. Only `readout` ever moves."""

    def __init__(self, model, cfg):
        self.model = model.requires_grad_(False).eval()
        self.model.readout.requires_grad_(True)
        self.device = model.device
        self.amp_dtype = getattr(torch, str(cfg.amp_dtype))
        self.clip = float(cfg.clip_grad_norm)
        self.optimizer = torch.optim.AdamW(self.model.readout.parameters(), lr=float(cfg.lr),
                                           weight_decay=float(cfg.weight_decay))

    def _pairs(self, batch):
        """(states, true embeddings), both detached: observe() is decorated @torch.no_grad."""
        latents = self.model.observe(batch)
        return latents["z"].float(), latents["embed"].float()

    def step(self, batch):
        self.model.readout.train()
        z, target = self._pairs(batch)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, dtype=self.amp_dtype):
            pred = self.model.readout(z)
            loss = (pred.float() - target).pow(2).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.readout.parameters(), self.clip)
        self.optimizer.step()
        with torch.no_grad():
            r2 = 1.0 - loss / target.var().clamp(min=1e-8)
        return {"loss/readout": loss.detach(), "readout/r2": r2,
                "readout/cos": torch.cosine_similarity(pred.float().flatten(0, 1),
                                                       target.flatten(0, 1), dim=-1).mean(),
                "opt/grad_norm": grad_norm}

    @torch.no_grad()
    def evaluate(self, batch):
        self.model.readout.eval()
        z, target = self._pairs(batch)
        with autocast(self.device.type, dtype=self.amp_dtype):
            pred = self.model.readout(z).float()
        mse = (pred - target).pow(2).mean()
        return {"loss/readout": mse, "readout/r2": 1.0 - mse / target.var().clamp(min=1e-8)}

    def state_dict(self):
        return {"readout": self.model.readout.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.model.readout.load_state_dict(state["readout"])
        self.optimizer.load_state_dict(state["optimizer"])


@hydra.main(version_base=None, config_path="../configs/smt_phase1", config_name="readout_phase1")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")
    logdir = Path(cfg.logdir)

    wm_state = CheckpointManager.load(cfg.wm_checkpoint)
    wm_cfg = OmegaConf.create(wm_state["config"])
    train_set, holdout_set = instantiate(wm_cfg.data)
    model = instantiate(wm_cfg.model, action_dim=train_set.action_dim, device=cfg.device, _recursive_=False)
    model.load_train_state_dict(wm_state["model"])
    # which frozen states to regress from; the SAVED config keeps stage 1's observe_mode, so
    # this choice never leaks into how the eval reads the model
    model.observe_mode = str(cfg.readout.states)

    trainer = ReadoutTrainer(model, cfg.readout)
    checkpoints = CheckpointManager(logdir / "checkpoints", "wm")
    resume = resume_path(cfg.trainer.resume, checkpoints)
    state = CheckpointManager.load(resume) if resume else None
    start_step = 0
    if state is not None:
        trainer.load_state_dict(state["readout_trainer"])
        start_step = int(state["readout_step"])
        print(f"resumed from {Path(resume).resolve()} at step {start_step}")

    config = OmegaConf.to_container(cfg, resolve=True)
    config["world_model"] = wm_state["config"]
    logger = Logger(logdir, config, cfg.wandb, job_type="readout",
                    name=f"{wm_cfg.model.name}/{Path(cfg.wm_run).name}/readout", group=cfg.wandb.group,
                    run_id=state["wandb_run_id"] if state else None, tensorboard=cfg.tensorboard)

    loader_kwargs = dict(batch_size=cfg.loader.batch_size, num_workers=cfg.loader.num_workers,
                         pin_memory=True, persistent_workers=cfg.loader.num_workers > 0)
    loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    holdout_loader = (DataLoader(holdout_set, shuffle=False, drop_last=False, **loader_kwargs)
                      if len(holdout_set) else None)

    def evaluate(step):
        if holdout_loader is None:
            return
        totals, count = {}, 0
        for batch in holdout_loader:
            n = batch["image"].shape[0]
            for k, v in trainer.evaluate(to_device(batch, cfg.device)).items():
                totals[k] = totals.get(k, 0.0) + float(v) * n
            count += n
        if count:
            logger.scalars(step, {f"holdout/{k}": v / count for k, v in totals.items()})

    def state_fn(step):
        # a WORLD-MODEL checkpoint, carrying stage 1's config verbatim: load_frozen_models and
        # train_decoder.py rebuild the identical model from it, now with a fitted readout
        return {"kind": "world_model", "step": int(wm_state["step"]), "config": wm_state["config"],
                "model": model.train_state_dict(), "readout_step": step,
                "readout_trainer": trainer.state_dict(), "readout_config": config,
                "stage1_checkpoint": str(Path(cfg.wm_checkpoint).resolve()),
                "wandb_run_id": logger.run_id}

    print(f"readout on frozen {Path(cfg.wm_checkpoint).resolve()} (wm step {wm_state['step']}) | "
          f"states {cfg.readout.states} | {len(train_set)} clips | logdir {logdir}")
    train_loop(trainer.step, loader, cfg.device, logger, checkpoints, state_fn,
               steps=int(cfg.trainer.steps), log_every=int(cfg.trainer.log_every),
               ckpt_every=int(cfg.trainer.ckpt_every), start_step=start_step,
               callbacks=[(int(cfg.trainer.eval_every), evaluate)])
    logger.close()


if __name__ == "__main__":
    main()
