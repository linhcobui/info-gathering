"""Gate: can a linear read-out recover the held bar's tilt from frozen DINOv2 tokens, and from a frame token?

Held frames (both pads on the bar, bar off the floor) from the training store. Inputs compared:
  patches   all 256 x 384 patch tokens (98,304 numbers)
  pool      a trained OGBar-SMT frame pool (one 256-number token per frame), from --smt
  cls       the 384-number CLS token
  avg4x4    the 16 x 16 patch grid averaged down to 4 x 4 (16 x 384 = 6,144 numbers)
Labels (privileged, scoring only): the bar's angle in the side camera's image plane (camera D looks along -y, so
the plane is world x-z; ends are identical so the angle is folded to (-90, 90]) on side-on frames (|cos yaw| > 0.7),
and |tilt| = |arcsin(axis_z)| on all held frames. Ridge regression (dual form), 5 folds split by episode.

    pixi run python scripts/tilt_probe_tokens.py --smt logdir/world_model/smt_ogbar/v3_400_unif0.1/checkpoints/wm_latest.pt
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

from world_model.models.smt_ogbar import SmtOgbar  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402


def bar_axis(q):
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], 1)


def cv_r2(X, y, groups, lams=(1e-2, 1e-1, 1, 10, 100, 1000)):
    """Best-lambda R^2 of dual ridge on standardised features, folds split by episode."""
    folds = np.unique(groups) % 5
    gi = {g: f for g, f in zip(np.unique(groups), folds)}
    fold = np.array([gi[g] for g in groups])
    best = -np.inf
    for lam in lams:
        pred = np.zeros_like(y)
        for f in range(5):
            tr, te = fold != f, fold == f
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            A = torch.from_numpy((X[tr] - mu) / sd).cuda(1)
            B = torch.from_numpy((X[te] - mu) / sd).cuda(1)
            yt = torch.from_numpy(y[tr] - y[tr].mean()).cuda(1).double()
            K = (A @ A.T).double()
            alpha = torch.linalg.solve(K + lam * A.shape[1] * torch.eye(len(K), device=K.device, dtype=K.dtype), yt)
            pred[te] = ((B @ A.T).double() @ alpha).cpu().numpy() + y[tr].mean()
        best = max(best, 1 - ((pred - y) ** 2).mean() / y.var())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smt", required=True)
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    z = zarr.open_group(str(ROOT / "data/ogbar_v3_400.zarr"), "r")
    P = np.load(ROOT / "data/ogbar_v3_400_dinov2/patch.npy", mmap_mode="r")
    C = np.load(ROOT / "data/ogbar_v3_400_dinov2/cls.npy", mmap_mode="r")
    ends = z["episode_ends"][:]
    held = z["grasped"][:] & ~z["bar_floor_contact"][:]
    idx = np.sort(np.random.default_rng(0).choice(np.where(held)[0], args.n, replace=False))
    ep = np.searchsorted(ends, idx, side="right")
    ax = bar_axis(z["bar_quat"][:][idx])
    abs_tilt = np.degrees(np.abs(np.arcsin(np.clip(ax[:, 2], -1, 1))))
    img = np.degrees(np.arctan2(ax[:, 2], ax[:, 0]))
    img = (img + 90) % 180 - 90                                # identical ends: fold to (-90, 90]
    side_on = np.abs(ax[:, 0]) / np.linalg.norm(ax[:, :2], axis=1) > 0.7

    patches = np.stack([P[i] for i in idx]).astype(np.float32)          # (n, 256, 384)
    feats = {"patches (256 x 384)": patches.reshape(len(idx), -1),
             "cls (384)": np.stack([C[i] for i in idx]).astype(np.float32),
             "avg4x4 (16 x 384)": patches.reshape(-1, 4, 4, 4, 4, 384).mean((2, 4)).reshape(len(idx), -1)}
    st = CheckpointManager.load(args.smt)
    m = SmtOgbar(action_dim=25, **st["config"]["model"]).cuda(1).eval()
    m.load_state_dict(st["model"])
    with torch.no_grad():
        pooled = torch.cat([m.pool(torch.from_numpy(patches[i:i + 256]).cuda(1)).float().cpu() for i in range(0, len(idx), 256)])
    feats[f"learned pool (256), step {st['step']}"] = pooled.numpy()
    print(f"{len(idx)} held frames from {len(np.unique(ep))} episodes; side-on {side_on.sum()}; "
          f"|tilt| mean {abs_tilt.mean():.1f} deg (sd {abs_tilt.std():.1f}); image angle sd (side-on) {img[side_on].std():.1f} deg")
    print(f"{'input':34s} R2 image-plane angle (side-on)   R2 |tilt| (all held)")
    for name, X in feats.items():
        print(f"{name:34s} {cv_r2(X[side_on], img[side_on], ep[side_on]):30.3f}   {cv_r2(X, abs_tilt, ep):.3f}", flush=True)


if __name__ == "__main__":
    main()
