"""Step 1 of the error-spike segmentation (configs/segmentation.yaml): the frozen DINO-WM's teacher-forced one-step
error at every valid step of every episode in `stores`.

Writes to paths.out_dir: wm_errors.parquet (+ .meta.json), wm_errors_summary.json, frames.parquet with
frame_mean.npy / frame_action.npy / frame_proprio.npy (g's inputs), and wm_features.npy.

    pixi run python scripts/compute_wm_errors.py [key=value ...]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from seg.config import load_config, out_dir  # noqa: E402
from seg.episodes import load_episodes  # noqa: E402
from seg.metrics import context_index, to_device  # noqa: E402
from seg.pipeline import errors_step  # noqa: E402
from seg.wm import DinoWMAdapter  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    out = out_dir(cfg)
    wm = DinoWMAdapter(ROOT / cfg.paths.wm_ckpt, cfg.device, cfg.errors.bf16, cfg.errors.wm_feature_layer)
    if wm.history_len != cfg.history_len:
        sys.exit(f"history_len {cfg.history_len} != the WM's num_hist {wm.history_len}")
    episodes = load_episodes(cfg)
    print(f"{len(episodes)} episodes ({sum(e.wm_train for e in episodes)} WM-train), WM step {wm.step}", flush=True)
    ep = episodes[0]
    vis, prop, act = to_device(ep, wm.device)
    idx = context_index(torch.arange(wm.history_len - 1, wm.history_len + 7), wm.history_len, wm.device)
    print(f"adapter vs DinoWM.predict: max |diff| {wm.check_against_model(vis[idx], prop[idx], act[idx]):.1e}", flush=True)
    t0 = time.time()
    errors_step(episodes, wm, cfg, out)
    print(f"wrote {out / 'wm_errors.parquet'} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
