"""Generate OGBar data (OGBench cube-single with a hidden-mass bar). Documentation: docs/OGBAR.md.

Usage (current config: configs/data/ogbar_slot_v3_flat_noslip.yaml, play episodes):
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 pixi run python scripts/generate_ogbar.py \
      --config configs/data/ogbar_slot_v3_flat_noslip.yaml --n 400 --workers 8 \
      --zarr data/ogbar_v3_400.zarr --video-dir data/videos_ogbar_v3_400
  ... --append                                      # add episodes to an existing store
  ... --seed 1 --n 200 --zarr data/ogbar_v3_heldout200.zarr   # held-out episodes (ogbar_v3_400 used seed 0)
The first debug.n_episodes episodes of a new store get debug videos. --verify / --no-noise are for the old
pick-and-place configs (ogbar_single.yaml).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
import zarr  # noqa: E402

from ogbar.generate import check_debug_does_not_leak, generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/ogbar_single.yaml")
    ap.add_argument("--n", type=int, default=100, help="episodes to generate")
    ap.add_argument("--append", action="store_true", help="append to the existing zarr store")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: config `workers`)")
    ap.add_argument("--no-noise", action="store_true",
                    help="no action noise or random actions (clean policy); outputs get a _nonoise suffix")
    ap.add_argument("--zarr", default=None, help="output zarr (default: config zarr_path)")
    ap.add_argument("--video-dir", default=None, help="debug video folder (default: config video_dir)")
    ap.add_argument("--seed", type=int, default=None,
                    help="master seed (default: config seed); a different one gives disjoint episodes, e.g. held-out data")
    ap.add_argument("--verify", action="store_true",
                    help="5 fixed-split verification episodes with debug videos into the verify paths")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    workers = args.workers or cfg["workers"]
    cfg["zarr_path"] = args.zarr or cfg["zarr_path"]
    cfg["video_dir"] = args.video_dir or cfg["video_dir"]
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.no_noise:
        cfg["noise"]["noise_max"] = 0.0
        cfg["noise"]["p_random_action"] = 0.0
        for block in (cfg, cfg["verify"]):
            block["zarr_path"] = block["zarr_path"].replace(".zarr", "_nonoise.zarr")
            block["video_dir"] = block["video_dir"] + "_nonoise"

    if args.verify:
        v = cfg["verify"]
        ws = v["w_right_values"]
        generate(cfg, len(ws), v["zarr_path"], v["video_dir"], w_right_values=ws, n_debug=len(ws),
                 workers=min(workers, len(ws)), video_prefix="verify")
        seed0 = int(zarr.open_group(v["zarr_path"], mode="r")["episode_seeds"][0])   # seed actually recorded
        same = check_debug_does_not_leak(cfg, seed0, ws[0])
        print(f"debug rendering leaves the recorded data unchanged: {same}")
        if not all(same.values()):
            sys.exit("FAIL: debug rendering changed the recorded data")
        return

    generate(cfg, args.n, cfg["zarr_path"], cfg["video_dir"], append=args.append,
             n_debug=0 if args.append else cfg["debug"]["n_episodes"], workers=workers)


if __name__ == "__main__":
    main()
