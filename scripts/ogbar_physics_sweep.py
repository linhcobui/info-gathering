"""Phase 0 physics sweep for OGBar: in-hand tilt vs hidden-mass torque per pad contact regime.

Usage: pixi run ogbar-sweep [--config configs/data/ogbar_single.yaml] [--drop | --platform]
Writes data/ogbar_sweep/sweep.json, figures/ogbar_physics_sweep.png and one debug video per regime.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

import json  # noqa: E402

from ogbar.physics_sweep import run_drop_curves, run_platform_curves, run_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/ogbar_single.yaml")
    ap.add_argument("--drop", action="store_true", help="natural drop curves instead of the tilt sweep")
    ap.add_argument("--platform", action="store_true",
                    help="floor -> platform task success vs grasp-to-centre-of-mass distance")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.platform:
        d = cfg["drop"]
        res = run_platform_curves(cfg, d["n_platform_trials"], d["lever_bins_cm"], args.workers)
        os.makedirs(os.path.dirname(d["platform_out_json"]), exist_ok=True)
        with open(d["platform_out_json"], "w") as f:
            json.dump(res, f)
        return
    if args.drop:
        d = cfg["drop"]
        out = run_drop_curves(cfg, d, args.workers)
        os.makedirs(os.path.dirname(d["out_json"]), exist_ok=True)
        with open(d["out_json"], "w") as f:
            json.dump(out, f)
        return
    sw = cfg["sweep"]
    run_sweep(cfg, sw, sw["out_json"], sw["out_fig"], sw["video_dir"])


if __name__ == "__main__":
    main()
