"""Train DINO-WM (+ proprioception) on cached DINOv2 tokens of an OGBar store.

    pixi run python scripts/train_dino_wm.py                                # configs/dino_wm/train_ogbar_v3_400.yaml
    pixi run python scripts/train_dino_wm.py model.use_proprio=false        # vision-only ablation
    pixi run python scripts/train_dino_wm.py trainer.resume=<logdir>/checkpoints/wm_latest.pt

Writes <logdir>/checkpoints/wm_latest.pt (and wm_step<N>.pt every keep_every steps),
metrics to wandb + metrics.jsonl, and reconstruction grids to <logdir>/recon/.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from info_gathering.ogbar_dino import OgbarDinoWindows  # noqa: E402
from world_model.models.dino_wm import DinoWM  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402
from world_model.utils.logger import Logger  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402


def to_u8(img):
    """(…, 3, H, W) in [-1, 1] -> (…, H, W, 3) uint8."""
    return ((img.float().clamp(-1, 1) + 1) * 127.5).round().byte().movedim(-3, -1).cpu().numpy()


def recon_grid(batch, extras, n=4):
    """Rows per sample: real frames | decoder(real tokens) | decoder(predicted tokens), frames 0..H."""
    rows = []
    for i in range(min(n, batch["rgb"].shape[0])):
        real = batch["rgb"][i].cpu().numpy()
        rec = to_u8(extras["recon"][i])
        pred = np.zeros_like(real)
        if "pred_img" in extras:
            pred[1:] = to_u8(extras["pred_img"][i])
        for r in (real, rec, pred):
            rows.append(np.concatenate(list(r), axis=1))
    return np.concatenate(rows, axis=0)


def build_model(cfg, stats, action_dim, proprio_dim):
    m = cfg.model
    return DinoWM(action_dim=action_dim, proprio_dim=proprio_dim, stats=stats, num_hist=m.num_hist,
                  num_pred=m.num_pred, action_emb_dim=m.action_emb_dim, proprio_emb_dim=m.proprio_emb_dim,
                  use_proprio=m.use_proprio, predictor=OmegaConf.to_container(m.predictor),
                  decoder=OmegaConf.to_container(m.decoder), has_decoder=m.has_decoder)


@hydra.main(version_base=None, config_path="../configs/dino_wm", config_name="train_ogbar_v3_400")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logdir = Path(cfg.logdir)
    checkpoints = CheckpointManager(logdir / "checkpoints", "wm")
    resume = cfg.trainer.resume
    state = CheckpointManager.load(resume) if resume else None

    data = OgbarDinoWindows(ROOT / cfg.data.cache_dir, ROOT / cfg.data.zarr_path,
                            num_frames=cfg.model.num_hist + cfg.model.num_pred, frameskip=cfg.data.frameskip,
                            load_rgb=cfg.model.has_decoder)
    stats = state["stats"] if state else {k: v.tolist() for k, v in data.stats().items()}
    device = torch.device(cfg.device)
    model = build_model(cfg, stats, data.action_dim, data.proprio_dim).to(device)
    opts = model.param_groups(**cfg.optim)
    step = 0
    if state:
        model.load_state_dict(state["model"])
        for o, s in zip(opts, state["optimizers"]):
            o.load_state_dict(s)
        step = int(state["step"])
        print(f"resumed from {resume} at step {step}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{len(data)} windows from {len(data.episode_ends)} episodes; {n_params / 1e6:.1f}M parameters", flush=True)

    config = OmegaConf.to_container(cfg, resolve=True)
    logger = Logger(logdir, config, cfg.wandb, job_type="world_model", name=f"dino_wm/{logdir.name}",
                    group=cfg.wandb.group, run_id=state["wandb_run_id"] if state else None,
                    tensorboard=cfg.tensorboard)
    loader = DataLoader(data, batch_size=cfg.loader.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.loader.num_workers, pin_memory=True,
                        persistent_workers=cfg.loader.num_workers > 0, prefetch_factor=4)

    def save(name):
        return checkpoints.save(name, dict(model=model.state_dict(), optimizers=[o.state_dict() for o in opts],
                                           step=step, stats=stats, config=config, wandb_run_id=logger.run_id))

    (logdir / "recon").mkdir(parents=True, exist_ok=True)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=bool(cfg.trainer.bf16))
    model.train()
    sums, counts, count, t0 = {}, {}, 0, time.time()
    while step < cfg.trainer.steps:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            show = step % cfg.trainer.recon_every == 0
            with torch.autocast(**amp):
                loss, logs, extras = model(batch, decode_pred=show)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o in opts:
                o.step()
            step += 1
            for k, v in logs.items():                  # some keys only exist on recon steps
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
            count += 1
            if step % cfg.trainer.log_every == 0:
                vals = {f"train/{k}": v / counts[k] for k, v in sums.items()}
                vals["perf/steps_per_s"] = count / (time.time() - t0)
                vals["perf/epoch"] = step * cfg.loader.batch_size / len(data)
                logger.scalars(step, vals)
                sums, counts, count, t0 = {}, {}, 0, time.time()
            if show and cfg.model.has_decoder:
                grid = recon_grid(batch, extras)
                imageio.imwrite(logdir / "recon" / f"step{step - 1:07d}.png", grid)
                logger.run.log({"recon": __import__("wandb").Image(grid)}, step=step)
            if step % cfg.trainer.ckpt_every == 0:
                save("latest")
            if step % cfg.trainer.keep_every == 0:
                save(f"step{step}")
            if step >= cfg.trainer.steps:
                break
    save("latest")
    print(f"done at step {step}", flush=True)


if __name__ == "__main__":
    main()
