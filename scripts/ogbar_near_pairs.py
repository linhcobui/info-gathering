"""Show cross-episode near-duplicate frames at a given 4x4-pooled DINO distance, with one-step pairs for scale.

Model-step frames (raw stride 5, offset 0); 4x4-pooled DINOv2 tokens (16 x 384), Euclidean distance. Pooled tokens are
cached to figures/ogbar_smt/pooled4x4_stride5.npy (fp16) on first use. Captions add the simulator's bar and gripper
positions (privileged, only to show how alike the scenes really are).

    pixi run python scripts/ogbar_near_pairs.py --target 29 --n 8
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

CACHE = ROOT / "figures/ogbar_smt/pooled4x4_stride5.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=29.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    z = zarr.open_group(str(ROOT / "data/ogbar_v3_400.zarr"), "r")
    ends = z["episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    ms = np.concatenate([np.arange(a, b, 5) for a, b in zip(starts, ends)])
    ep = np.repeat(np.arange(len(ends)), [len(np.arange(a, b, 5)) for a, b in zip(starts, ends)])
    if CACHE.exists():
        X = np.load(CACHE).astype(np.float32)
    else:
        P = np.load(ROOT / "data/ogbar_v3_400_dinov2/patch.npy", mmap_mode="r")
        X = np.stack([P[i].astype(np.float32).reshape(4, 4, 4, 4, 384).mean((1, 3)).ravel() for i in ms])
        np.save(CACHE, X.astype(np.float16))
    d = np.load(ROOT / "figures/ogbar_smt/frame_duplicates.npz")
    cross = d["cross"]
    rng = np.random.default_rng(0)
    cand = np.where(np.abs(cross - args.target) < 0.3)[0]
    pick = rng.choice(cand, size=min(args.n, len(cand)), replace=False)
    dev = torch.device(args.device)
    Xt = torch.from_numpy(X).to(dev)
    bar, eff, yaw = z["bar_pos"], z["effector_pos"], z["bar_yaw_mod_pi"]
    rows = []

    def tile(i, j, dist, label):
        a, b = int(ms[i]), int(ms[j])
        ims = [Image.fromarray(z["rgb"][k]) for k in (a, b)]
        for im, k, e in zip(ims, (a, b), (ep[i], ep[j])):
            dr = ImageDraw.Draw(im)
            dr.rectangle((0, 0, 223, 13), fill=(0, 0, 0))
            dr.text((3, 1), f"ep {e} step {(k - starts[e]) // 5}", fill=(255, 255, 0))
        both = Image.new("RGB", (448, 224 + 30), (255, 255, 255))
        both.paste(ims[0], (0, 0)); both.paste(ims[1], (224, 0))
        db = np.linalg.norm(bar[a][:2] - bar[b][:2]) * 100
        dy = abs(((np.degrees(yaw[a] - yaw[b]) + 90) % 180) - 90)
        de = np.linalg.norm(eff[a] - eff[b]) * 100
        dr = ImageDraw.Draw(both)
        dr.text((4, 226), f"{label}: DINO 4x4 distance {dist:.1f}", fill=(0, 0, 0))
        dr.text((4, 239), f"bar centre {db:.1f} cm apart, yaw {dy:.0f} deg; gripper {de:.1f} cm apart", fill=(90, 90, 90))
        return np.asarray(both)

    for i in pick:
        d2 = ((Xt - Xt[i]) ** 2).sum(1)
        d2[torch.from_numpy(ep == ep[i]).to(dev)] = float("inf")
        j = int(d2.argmin())
        rows.append(tile(i, j, float(d2[j].sqrt()), "different episodes"))
    cons = np.linalg.norm(X[1:] - X[:-1], axis=1)
    same = ep[1:] == ep[:-1]
    for i in rng.choice(np.where(same & (np.abs(cons - 33.6) < 0.3))[0], 2, replace=False):
        rows.append(tile(i, i + 1, float(cons[i]), "one model step, same episode"))
    grid = []
    for r in range(0, len(rows), 2):
        pair = rows[r:r + 2]
        if len(pair) == 1:
            pair.append(np.full_like(pair[0], 255))
        grid.append(np.concatenate([pair[0], np.full((pair[0].shape[0], 16, 3), 255, np.uint8), pair[1]], 1))
    out = ROOT / f"figures/ogbar_smt/near_pairs_d{args.target:.0f}.png"
    Image.fromarray(np.concatenate(grid, 0)).save(out)
    print(out)


if __name__ == "__main__":
    main()
