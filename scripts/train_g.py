"""Step 3 of the error-spike segmentation (configs/segmentation.yaml): train the error predictor
g(e_{t-H+1:t}, a_t) ~ P(err_t > lambda) on held-out episodes (split by episode), report AUROC / AUPRC / calibration
on the held-out test split, run the no-action ablation, and score every step of every episode.

Writes g_scores.parquet (+ .meta.json), g/g.pt, g/metrics.json, diagnostics/g_reliability.png to paths.out_dir.

    pixi run python scripts/train_g.py [key=value ...]          # e.g. g.arch=transformer g.pooling=attn
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from seg.config import load_config, out_dir  # noqa: E402
from seg.episodes import load_episodes  # noqa: E402
from seg.pipeline import g_step  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    episodes = load_episodes(cfg) if cfg.g.pooling == "attn" else None   # only attn pooling reads the patch caches
    g_step(cfg, out_dir(cfg), torch.device(cfg.device), episodes)


if __name__ == "__main__":
    main()
