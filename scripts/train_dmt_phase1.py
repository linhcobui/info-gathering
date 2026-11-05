"""PHASE 1 stage 1b: DMT -- fine-tune the updaters on their OWN rollout states.

Starts from a stage-1 checkpoint, freezes the teacher encoder, and rolls the posterior over
contiguous slices of real episodes, correcting every visited state against the teacher's
encoding of the same step (world_model/models/smt_phase1/dmt.py). Writes a world-model
checkpoint carrying the stage-1 config verbatim, so train_readout_phase1.py, train_decoder.py
and eval_memtest1.py rebuild the model from it unchanged.

    pixi run train-dmt-phase1 wm_run=<smt_phase1 run dir>
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
from world_model.models.smt_phase1.dmt import dmt_losses, dmt_total, oracle_bank  # noqa: E402
from world_model.training.loop import to_device, train_loop  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402
from world_model.utils.logger import Logger  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402


class DmtTrainer:
    """Teacher frozen; posterior, prior and the reward head train on rolled-out states."""

    def __init__(self, model, cfg, bank=None):
        self.model, self.cfg, self.bank = model, cfg, bank
        model.requires_grad_(False)
        self.params = []
        for m in (model.posterior, model.prior, model.reward_decoder):
            m.requires_grad_(True)
            self.params += list(m.parameters())
        self.amp_dtype = getattr(torch, str(cfg.amp_dtype))
        self.clip = float(cfg.clip_grad_norm)
        self.optimizer = torch.optim.AdamW(self.params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    def step(self, batch):
        self.model.train()
        self.model.encoder.eval()
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.model.device.type, dtype=self.amp_dtype):
            losses, metrics = dmt_losses(self.model, batch, self.cfg, self.bank)
        total = dmt_total(losses, self.cfg)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.optimizer.step()
        out = {f"loss/{k}": v for k, v in losses.items()}
        out.update(metrics)
        out["opt/loss"] = total
        out["opt/grad_norm"] = grad_norm
        return out

    @torch.no_grad()
    def evaluate(self, batch):
        self.model.eval()
        with autocast(self.model.device.type, dtype=self.amp_dtype):
            losses, metrics = dmt_losses(self.model, batch, self.cfg, self.bank)
        out = {f"loss/{k}": v for k, v in losses.items()}
        out.update(metrics)
        out["opt/loss"] = dmt_total(losses, self.cfg)
        return out


@hydra.main(version_base=None, config_path="../configs/smt_phase1", config_name="dmt_phase1")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    logdir = Path(cfg.logdir)
    wm_state = CheckpointManager.load(cfg.wm_checkpoint)
    wm_cfg = OmegaConf.create(wm_state["config"])
    win_train, _ = instantiate(wm_cfg.data)
    model = instantiate(wm_cfg.model, action_dim=win_train.action_dim, device=cfg.device, _recursive_=False)
    model.load_train_state_dict(wm_state["model"])

    roll_train, roll_holdout = instantiate(cfg.data_rollouts, ctx_len=int(wm_cfg.model.ctx_len))
    bank = None
    if bool(cfg.dmt.cache_teacher):
        # the teacher is frozen, so its states never change: encode every (episode, step) once
        bank = oracle_bank(model, roll_train.data, cfg.device)
        print(f"cached teacher states: {tuple(bank.shape)} ({bank.numel() * 2 / 1e6:.0f} MB fp16)")
    trainer = DmtTrainer(model, cfg.dmt, bank)

    checkpoints = CheckpointManager(logdir / "checkpoints", "wm")
    resume = resume_path(cfg.trainer.resume, checkpoints)
    state = CheckpointManager.load(resume) if resume else None
    start_step = 0
    if state is not None:
        model.load_train_state_dict(state["model"])
        trainer.optimizer.load_state_dict(state["dmt_optimizer"])
        start_step = int(state["dmt_step"])
        print(f"resumed from {Path(resume).resolve()} at step {start_step}")

    config = OmegaConf.to_container(cfg, resolve=True)
    config["world_model"] = wm_state["config"]
    logger = Logger(logdir, config, cfg.wandb, job_type="dmt",
                    name=f"{wm_cfg.model.name}/{Path(cfg.wm_run).name}/dmt", group=cfg.wandb.group,
                    run_id=state["wandb_run_id"] if state else None, tensorboard=cfg.tensorboard)
    kw = dict(batch_size=cfg.loader.batch_size, num_workers=cfg.loader.num_workers,
              pin_memory=True, persistent_workers=cfg.loader.num_workers > 0)
    loader = DataLoader(roll_train, shuffle=True, drop_last=True, **kw)
    holdout = DataLoader(roll_holdout, shuffle=False, drop_last=False, **kw) if len(roll_holdout) else None

    def evaluate(step):
        if holdout is None:
            return
        totals, count = {}, 0
        for b in holdout:
            n = b["t"].shape[0]
            for k, v in trainer.evaluate(to_device(b, cfg.device)).items():
                totals[k] = totals.get(k, 0.0) + float(v) * n
            count += n
        if count:
            logger.scalars(step, {f"holdout/{k}": v / count for k, v in totals.items()})

    def state_fn(step):
        return {"kind": "world_model", "step": int(wm_state["step"]), "config": wm_state["config"],
                "model": model.train_state_dict(), "dmt_step": step,
                "dmt_optimizer": trainer.optimizer.state_dict(), "dmt_config": config,
                "stage1_checkpoint": str(Path(cfg.wm_checkpoint).resolve()), "wandb_run_id": logger.run_id}

    print(f"DMT on {Path(cfg.wm_checkpoint).resolve()} (stage-1 step {wm_state['step']}) | "
          f"roll_len {cfg.data_rollouts.roll_len} | {len(roll_train)} slices | logdir {logdir}")
    train_loop(trainer.step, loader, cfg.device, logger, checkpoints, state_fn,
               steps=int(cfg.trainer.steps), log_every=int(cfg.trainer.log_every),
               ckpt_every=int(cfg.trainer.ckpt_every), start_step=start_step,
               callbacks=[(int(cfg.trainer.eval_every), evaluate)])
    logger.close()


if __name__ == "__main__":
    main()
