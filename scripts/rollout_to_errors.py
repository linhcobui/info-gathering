"""Turn the open-loop errors of compute_rollout_errors.py into an errors table for one horizon h, in a new output
directory, so make_labels.py / train_g.py / seg_video.py run on the h-step error (set error_horizon=h, g.n_actions=h).

Row t of the new wm_errors.parquet = the WM's prediction of frame t+h from the real context ending at t, open loop
with the logged actions a_t .. a_{t+h-1}. The frame tables and g's frame inputs are linked from the source directory.

    pixi run python scripts/rollout_to_errors.py --h 4 --out outputs/v5_h4_topk
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from seg.config import load_config, out_dir, write_meta  # noqa: E402
from seg.pipeline import error_summary  # noqa: E402

LINKED = ("frames.parquet", "frame_mean.npy", "frame_action.npy", "frame_proprio.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", nargs="+", default=["wm_rollout_errors_v3_400.parquet", "wm_rollout_errors_heldout200.parquet"])
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(overrides=args.overrides)
    src = out_dir(cfg)
    df = pd.concat([pd.read_parquet(src / f) for f in args.inputs], ignore_index=True)
    df = df[df.h == args.h].drop(columns="h").sort_values(["store", "episode", "t"]).reset_index(drop=True)
    dst = ROOT / args.out
    dst.mkdir(parents=True, exist_ok=True)
    for f in LINKED:
        link = dst / f
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(src / f, dst), link)
    df.to_parquet(dst / "wm_errors.parquet")
    write_meta(dst / "wm_errors.parquet", cfg, horizon=args.h, inputs=args.inputs)
    (dst / "wm_errors_summary.json").write_text(json.dumps(error_summary(df), indent=1))
    print(f"wrote {dst / 'wm_errors.parquet'}: {len(df)} rows, {df.episode_id.nunique()} episodes, h = {args.h}")


if __name__ == "__main__":
    main()
