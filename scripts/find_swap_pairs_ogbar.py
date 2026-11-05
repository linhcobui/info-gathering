"""Find near-identical pre-grasp moments in two episodes whose COMs differ, so swapping the grasp should change the tilt.

For every pickup that grasps (both pads) in a training episode: the cut = last model step before the pads close. At the
grasp frame, in the camera's view frame (sign of the bar's x axis along world x): s_view = grasp offset, c_view = COM.
Pairs (A, B) from different episodes with 4x4-pooled DINO distance at the cut < --max-dist, both bars side-on, and:
  A's own grasp is near A's COM (|s_vA - c_vA| < --level cm: A lifts ~level)
  A's grasp is far from B's COM (|s_vA - c_vB| > --swing cm: B gripped there should swing)
Privileged quantities are used only to pick illustrative pairs.

    pixi run python scripts/find_swap_pairs_ogbar.py
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dist", type=float, default=32.0)
    ap.add_argument("--level", type=float, default=3.0)
    ap.add_argument("--swing", type=float, default=9.0)
    ap.add_argument("--side-on", type=float, default=0.85, help="min |cos yaw| relative to the camera's image plane")
    args = ap.parse_args()
    z = zarr.open_group(str(ROOT / "data/ogbar_v3_400.zarr"), "r")
    ends = z["episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    pk_ep, pk_tg = z["pickup/episode"][:], z["pickup/t_grasp"][:]
    c_all = z["episode/com_offset_c"][:] * 100
    quat, gs = z["bar_quat"], z["grasp_offset_s"]
    P = np.load(ROOT / "data/ogbar_v3_400_dinov2/patch.npy", mmap_mode="r")
    rows = []
    for e, tg in zip(pk_ep, pk_tg):
        if tg < 0:
            continue
        r = int(starts[e] + tg)
        cut = (int(tg) - 1) // 5
        if cut < 3:
            continue
        w, x, y, zz = quat[r]
        ax = 1 - 2 * (y * y + zz * zz)
        ay = 2 * (x * y + w * zz)
        if abs(ax) / max(np.hypot(ax, ay), 1e-9) < args.side_on:
            continue
        sgn = np.sign(ax)
        s_v = float(gs[r]) * 100 * -sgn                 # + = towards the image's right (camera D: image right = world -x)
        c_v = c_all[e] * -sgn
        if not np.isfinite(s_v):
            continue
        n_steps = (int(ends[e] - starts[e])) // 5
        if cut + 40 >= n_steps:
            continue
        tilt = np.degrees(np.abs(z["bar_tilt"][int(starts[e] + cut * 5):int(starts[e] + (cut + 30) * 5)])).max()
        rows.append((int(e), cut, s_v, c_v, int(starts[e] + cut * 5), float(tilt)))
    rows = np.array(rows, dtype=object)
    X = torch.from_numpy(np.stack([P[int(r[4])].astype(np.float32).reshape(4, 4, 4, 4, 384).mean((1, 3)).ravel()
                                   for r in rows])).cuda(1)
    D = torch.cdist(X, X).cpu().numpy()
    ep = np.array([r[0] for r in rows]); sv = np.array([r[2] for r in rows]); cv = np.array([r[3] for r in rows])
    tl = np.array([r[5] for r in rows])
    found = []
    for i in range(len(rows)):
        for j in range(len(rows)):
            if ep[i] == ep[j] or D[i, j] > args.max_dist:
                continue
            if abs(sv[i] - cv[i]) < args.level and abs(sv[i] - cv[j]) > args.swing and tl[i] < 8 and tl[j] > 25:
                found.append((D[i, j], i, j))
    found.sort()
    print(f"{len(rows)} side-on grasps; {len(found)} pairs (A level with its own grasp, B should swing with A's grasp)")
    for d, i, j in found[:12]:
        print(f"  dist {d:5.1f} | A ep {ep[i]:3d} step {rows[i][1]:3d}: c_view {cv[i]:+5.1f}, own grasp s_view {sv[i]:+5.1f} "
              f"(lever {sv[i] - cv[i]:+4.1f}) | B ep {ep[j]:3d} step {rows[j][1]:3d}: c_view {cv[j]:+5.1f}, own grasp "
              f"{sv[j]:+5.1f} (lever {sv[j] - cv[j]:+4.1f}); A's grasp on B: lever {sv[i] - cv[j]:+5.1f} | real max tilt A {tl[i]:.0f}, B {tl[j]:.0f}")


if __name__ == "__main__":
    main()
