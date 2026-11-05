"""Steps 4-6 of the error-spike segmentation (configs/segmentation.yaml): segment g into candidate windows, score
each window's resolvability with blind vs informed open-loop rollouts of the frozen DINO-WM (against random control
windows), export candidate_tests.json with frame images, and write diagnostics and REPORT.md.

    pixi run python scripts/find_candidate_tests.py [key=value ...]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seg.config import load_config, out_dir  # noqa: E402
from seg.episodes import load_episodes  # noqa: E402
from seg.pipeline import windows_step  # noqa: E402
from seg.wm import DinoWMAdapter  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    wm = DinoWMAdapter(ROOT / cfg.paths.wm_ckpt, cfg.device, cfg.errors.bf16, cfg.errors.wm_feature_layer)
    if wm.history_len != cfg.history_len:
        sys.exit(f"history_len {cfg.history_len} != the WM's num_hist {wm.history_len}")
    t0 = time.time()
    windows_step(cfg, out_dir(cfg), load_episodes(cfg), wm)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
