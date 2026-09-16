"""Build a resized copy of a cups-and-ball zarr for world-model training.

    pixi run preprocess-frames --size 64
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from info_gathering.dataset import build_cache, cache_path  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", default="data/memtest0.zarr")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--force", action="store_true", help="rebuild if the cache already exists")
    args = p.parse_args()

    out = cache_path(args.zarr, args.cache_dir, args.size)
    if out.exists():
        if not args.force:
            print(f"{out} already exists (--force to rebuild)")
            return
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print("wrote", build_cache(args.zarr, args.cache_dir, args.size, workers=args.workers))


if __name__ == "__main__":
    main()
