"""How often do OGBar observations repeat? (for TENET's o_j = o_{i-k} matching)

A. Exact: byte-identical 224x224 rgb frames over all raw frames (blake2b hash per frame). Pixels, not DINO tokens: the
   cached tokens of identical images can differ by one fp16 step depending on the encoding batch.
B. Near: 4x4-pooled DINOv2 tokens (16 x 384) of every model-step frame (raw stride 5, offset 0). For each frame: distance
   to its nearest frame in ANOTHER episode, and to its nearest frame in the SAME episode at least 3 model steps away,
   against the distance between consecutive model steps of one episode. Euclidean distance on the pooled fp32 tokens.

    pixi run python scripts/ogbar_frame_duplicates.py
"""
import collections
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402


class Chunks(torch.utils.data.Dataset):
    def __init__(self, n, rows):
        self.b = [(s, min(s + rows, n)) for s in range(0, n, rows)]

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        s, e = self.b[i]
        rgb = zarr.open_group(str(ROOT / "data/ogbar_v3_400.zarr"), "r")["rgb"][s:e]
        return s, [hashlib.blake2b(f.tobytes(), digest_size=16).hexdigest() for f in rgb]


def main():
    z = zarr.open_group(str(ROOT / "data/ogbar_v3_400.zarr"), "r")
    ends = z["episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    n = int(ends[-1])
    ep_of = np.repeat(np.arange(len(ends)), ends - starts)

    # A. exact pixel duplicates
    hashes = [None] * n
    dl = torch.utils.data.DataLoader(Chunks(n, z["rgb"].chunks[0] * 4), batch_size=None, num_workers=12)
    for s, hs in dl:
        hashes[s:s + len(hs)] = hs
    groups = collections.defaultdict(list)
    for i, h in enumerate(hashes):
        groups[h].append(i)
    dup = np.zeros(n, bool); dup_same = np.zeros(n, bool); dup_cross = np.zeros(n, bool)
    for idx in groups.values():
        if len(idx) > 1:
            idx = np.array(idx)
            dup[idx] = True
            eps = ep_of[idx]
            for i, e in zip(idx, eps):
                dup_same[i] |= (eps == e).sum() > 1
                dup_cross[i] |= (eps != e).any()
    print(f"A. raw frames {n} in {len(ends)} episodes; distinct images {len(groups)}")
    print(f"   frames with at least one byte-identical twin: {dup.sum()} ({dup.mean():.2%}); twin in the same episode "
          f"{dup_same.sum()} ({dup_same.mean():.2%}); twin in another episode {dup_cross.sum()} ({dup_cross.mean():.4%})")
    # at model-step frames (stride 5, offset 0)
    ms = np.concatenate([np.arange(a, b, 5) for a, b in zip(starts, ends)])
    hs = collections.Counter(hashes[i] for i in ms)
    print(f"   model-step frames (stride 5): {len(ms)}; with an identical twin among them: "
          f"{sum(1 for i in ms if hs[hashes[i]] > 1)}; cross-episode: "
          f"{sum(1 for i in ms if len({ep_of[j] for j in groups[hashes[i]] if j in set()}) > 1)}")

    # B. near duplicates on 4x4-pooled DINO tokens
    P = np.load(ROOT / "data/ogbar_v3_400_dinov2/patch.npy", mmap_mode="r")
    X = np.stack([P[i].astype(np.float32).reshape(4, 4, 4, 4, 384).mean((1, 3)).ravel() for i in ms])
    ep = ep_of[ms]
    step = np.concatenate([np.arange(b - a) for a, b in zip(*[np.r_[0, np.cumsum([len(np.arange(s, e, 5)) for s, e in zip(starts, ends)])][:-1],
                                                               np.cumsum([len(np.arange(s, e, 5)) for s, e in zip(starts, ends)])])])
    dev = torch.device("cuda:1")
    Xt = torch.from_numpy(X).to(dev)
    sq = (Xt * Xt).sum(1)
    ept, stt = torch.from_numpy(ep).to(dev), torch.from_numpy(step).to(dev)
    cross, within = np.empty(len(X)), np.empty(len(X))
    for i in range(0, len(X), 2048):
        q = Xt[i:i + 2048]
        d2 = (sq[i:i + 2048, None] + sq[None] - 2 * q @ Xt.T).clamp(min=0)
        same = ept[i:i + 2048, None] == ept[None]
        far = (stt[i:i + 2048, None] - stt[None]).abs() >= 3
        cross[i:i + 2048] = d2.masked_fill(same, float("inf")).min(1).values.sqrt().cpu().numpy()
        within[i:i + 2048] = d2.masked_fill(~(same & far), float("inf")).min(1).values.sqrt().cpu().numpy()
    cons = np.linalg.norm(X[1:] - X[:-1], axis=1)[ep[1:] == ep[:-1]]
    q = lambda a: f"median {np.median(a):.2f} [10% {np.percentile(a, 10):.2f}, 90% {np.percentile(a, 90):.2f}]"
    print(f"B. model-step frames {len(X)}; 4x4-pooled DINO tokens (6,144 numbers), Euclidean distance")
    print(f"   one model step within an episode: {q(cons)}")
    print(f"   nearest frame in ANOTHER episode: {q(cross)}")
    print(f"   nearest frame in the SAME episode, >= 3 steps away: {q(within)}")
    for thr_name, thr in (("10th pct of one-step change", np.percentile(cons, 10)), ("median one-step change", np.median(cons))):
        print(f"   frames whose nearest other-episode frame is closer than the {thr_name} ({thr:.2f}): {(cross < thr).mean():.1%}; "
              f"same-episode (>=3 steps away): {(within < thr).mean():.1%}")
    np.savez(ROOT / "figures/ogbar_smt/frame_duplicates.npz", cross=cross, within=within, cons=cons, dup=dup, dup_same=dup_same,
             dup_cross=dup_cross)


if __name__ == "__main__":
    main()
