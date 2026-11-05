"""Scalars to wandb, TensorBoard and <logdir>/metrics.jsonl; videos to wandb and <logdir>/videos/."""
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, logdir, config, wandb_cfg, job_type, name, group=None, run_id=None, tensorboard=True):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.logdir / "metrics.jsonl").open("a")
        self._tb = SummaryWriter(str(self.logdir / "tensorboard")) if tensorboard else None
        self.run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.get("entity"),
            mode=wandb_cfg.mode,
            group=group,
            job_type=job_type,
            name=name,
            tags=list(wandb_cfg.get("tags") or []),
            dir=str(self.logdir),
            config=config,
            id=run_id,
            resume="allow" if run_id else None,
        )

    @property
    def run_id(self):
        return self.run.id

    def scalars(self, step, metrics):
        values = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()}
        self.run.log(values, step=step)
        self._jsonl.write(json.dumps({"step": step, **values}) + "\n")
        self._jsonl.flush()
        if self._tb is not None:
            for k, v in values.items():
                self._tb.add_scalar(k, v, step)
        shown = " ".join(f"{k}={v:.4g}" for k, v in values.items() if "loss" in k or "mse" in k or "perf" in k)
        print(f"[{step}] {shown}", flush=True)
        return values

    def video(self, step, name, frames, fps):
        """frames: (B, T, H, W, 3) float in [0, 1] or uint8; the B clips are tiled side by side.

        Goes to wandb and <logdir>/videos/. Not TensorBoard: torch's add_video calls
        np.reshape(newshape=...), which numpy >= 2.4 rejects.
        """
        if torch.is_tensor(frames):
            frames = frames.detach().cpu().numpy()
        if np.issubdtype(frames.dtype, np.floating):
            frames = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
        B, T, H, W, C = frames.shape
        # (B, T, H, W, C) -> (T, H, B*W, C)
        tiled = frames.transpose(1, 2, 0, 3, 4).reshape(T, H, B * W, C)
        fps = max(1, round(fps))
        self.run.log({name: wandb.Video(tiled.transpose(0, 3, 1, 2), fps=fps, format="mp4")}, step=step)
        path = self.logdir / "videos" / f"{name.replace('/', '_')}_{step:09d}.mp4"
        path.parent.mkdir(exist_ok=True)
        imageio.mimwrite(path, tiled, fps=fps, macro_block_size=1)

    def close(self):
        self._jsonl.close()
        if self._tb is not None:
            self._tb.close()
        self.run.finish()
