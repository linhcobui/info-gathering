"""Write a SYMBOL one-hot observation: exactly the pixel information, clean geometry.

The rendered world is a small finite alphabet -- every frame in the dataset is one of N distinct
images. A one-hot over that alphabet is therefore a BIJECTION with what the renderer shows: it
adds nothing and removes nothing, unlike the hand-built oracle vector, which turned out to label
the swap pair on the two frames per block where the cups are stationary and the pixels show
nothing (measured: swap-pair decodable at 1.000 on moving frames, 0.311 on boundary frames).

Comparing this against the R2-Dreamer embedding isolates GEOMETRY with information held fixed.
Comparing it against the oracle vector isolates the SEGMENTATION burden.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, zarr
ZS = ("data/memtest4_big.zarr", "data/memtest4_eval.zarr", "data/memtest4_eval5.zarr")
KEY = "embed_r2d_memtest4_wm15000"
chunks = [np.round(zarr.open_group(z, mode="r")[KEY][:], 3) for z in ZS]
raw, inv0 = np.unique(np.concatenate(chunks), axis=0, return_inverse=True)
# Merge float-noise splits. The distance histogram is bimodal by three orders of magnitude:
# 9 pairs below 0.009, then nothing until 2.84. Without this, six spurious symbols appear that
# are specific to one split, handing eval a one-hot dimension never active in training.
D = np.linalg.norm(raw[:, None] - raw[None], axis=-1); np.fill_diagonal(D, np.inf)
lab = np.arange(len(raw))
for i in range(len(raw)):
    for j in range(i + 1, len(raw)):
        if D[i, j] < 1.0: lab[lab == lab[j]] = lab[i]
uniq, lab = np.unique(lab, return_inverse=True)
inv = lab[inv0]; N = len(uniq)
print(f"raw distinct rows {len(raw)} -> merged alphabet {N} "
      f"(min pairwise distance before merge {D.min():.4f}, median {np.median(D[np.isfinite(D)]):.2f})")
off = 0
for z, c in zip(ZS, chunks):
    codes = inv[off:off + len(c)]; off += len(c)
    oh = np.zeros((len(c), N), dtype=np.float32)
    oh[np.arange(len(c)), codes] = 1.0
    g = zarr.open_group(z, mode="a")
    k = "embed_symbol"
    if k in g: del g[k]
    g.create_dataset(k, data=oh, chunks=(4096, N), dtype="float32")
    g.attrs[k] = {"embed_dim": N, "source_checkpoint": "SYMBOL ONE-HOT (no encoder)",
                  "mean": oh.mean(0).tolist(), "std": (oh.std(0) + 1e-3).tolist(),
                  "note": f"one-hot over the {N}-frame visual alphabet; a bijection with the render"}
    used = len(np.unique(codes))
    print(f"  {z}: {oh.shape}, {used}/{N} symbols used")
