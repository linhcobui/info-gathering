"""Step 2 of the error-spike segmentation (configs/segmentation.yaml): episode-level split of the held-out episodes,
lambda (a quantile of error_metric over held-out g-train episodes) and labels y_t = 1[err_t > lambda].

Writes labels.parquet (+ .meta.json), split.json, threshold.json to paths.out_dir.

    pixi run python scripts/make_labels.py [key=value ...]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seg.config import load_config, out_dir  # noqa: E402
from seg.pipeline import labels_step  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    labels_step(cfg, out_dir(cfg))


if __name__ == "__main__":
    main()
