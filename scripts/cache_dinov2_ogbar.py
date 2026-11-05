"""Cache frozen DINOv2 ViT-S/14 tokens for EVERY frame of an OGBar store (fp16, unpooled).

Writes a directory next to the source store (the source is never modified):
    patch.npy         (N, 256, 384) float16   x_norm_patchtokens of the 224x224 `rgb` frame
    cls.npy           (N, 384)      float16   x_norm_clstoken
    episode_ends.npy  copy of the source's, so cache and source frame indices line up 1:1
    meta.json         source, model, preprocessing, format, frames_done

Every raw frame is cached (not only every 5th) because training windows start at any raw
offset (LeWM / DINO-WM style). The .npy files are plain memmaps so a training loader can read
any 4 frames without touching the rest. Resumable: frames_done is written after each flush.

--stride k caches only frames episode_start + k*j (offset 0 at frameskip k), about k times smaller,
for eval stores. Row i of patch.npy is then source frame frame_index.npy[i] (meta "stride": k), so
cache and source indices no longer line up 1:1; OgbarDinoWindows refuses such a cache.

    pixi run python scripts/cache_dinov2_ogbar.py --src data/ogbar_v3_400.zarr
    pixi run python scripts/cache_dinov2_ogbar.py --src data/ogbar_v3_heldout200.zarr --stride 5
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[1]
HUB_DIR = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dinov2(device):
    model = torch.hub.load(str(HUB_DIR), "dinov2_vits14", source="local")
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode(model, frames_u8, device):
    """uint8 (B, 224, 224, 3) -> patch (B, 256, 384) fp16, cls (B, 384) fp16 (fp32 compute)."""
    x = torch.from_numpy(frames_u8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = model.forward_features((x - mean) / std)
    return out["x_norm_patchtokens"].half(), out["x_norm_clstoken"].half()


class Chunks(torch.utils.data.Dataset):
    """One item = one stored zarr chunk of rgb frames (decompression runs in loader workers)."""

    def __init__(self, src, start, rows):
        self.src, self.rows = src, rows
        n = zarr.open_group(src, mode="r")["rgb"].shape[0]
        self.bounds = [(s, min(s + rows, n)) for s in range(start, n, rows)]
        self._rgb = None

    def __len__(self):
        return len(self.bounds)

    def __getitem__(self, i):
        if self._rgb is None:
            self._rgb = zarr.open_group(self.src, mode="r")["rgb"]
        s, e = self.bounds[i]
        return s, self._rgb[s:e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/ogbar_v3_400.zarr")
    ap.add_argument("--out", default=None, help="default: <src stem>_dinov2/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--verify-samples", type=int, default=256)
    ap.add_argument("--stride", type=int, default=1, help="cache only frames episode_start + stride*j (offset 0)")
    args = ap.parse_args()

    src = str(ROOT / args.src) if not os.path.isabs(args.src) else args.src
    out = Path(args.out or src.removesuffix(".zarr") + "_dinov2")
    out.mkdir(parents=True, exist_ok=True)
    z = zarr.open_group(src, mode="r")
    n, h, w, _ = z["rgb"].shape
    assert (h, w) == (224, 224), f"expected 224x224 frames, got {h}x{w}"
    rows = z["rgb"].chunks[0]
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    done = int(meta.get("frames_done", 0))
    mode = "r+" if done else "w+"
    if done and meta.get("n_frames") != n:
        raise SystemExit(f"{out} was started on {meta.get('n_frames')} frames, source now has {n}")
    if done and meta.get("stride", 1) != args.stride:
        raise SystemExit(f"{out} was started with stride {meta.get('stride', 1)}, not {args.stride}")
    ends = z["episode_ends"][:].astype(np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    frame_index = np.concatenate([np.arange(s0, e0, args.stride) for s0, e0 in zip(starts, ends)])
    m = len(frame_index)                                      # = n when stride is 1
    patch = np.lib.format.open_memmap(out / "patch.npy", mode=mode, dtype=np.float16, shape=(m, 256, 384))
    cls = np.lib.format.open_memmap(out / "cls.npy", mode=mode, dtype=np.float16, shape=(m, 384))
    np.save(out / "episode_ends.npy", ends)
    if args.stride != 1:
        np.save(out / "frame_index.npy", frame_index)
    meta = dict(source=os.path.relpath(src, ROOT), n_frames=n, model="dinov2_vits14 (torch.hub facebookresearch/dinov2)",
                feature_keys=["x_norm_patchtokens", "x_norm_clstoken"], image_size=224, patch_grid=[16, 16],
                normalisation=dict(mean=IMAGENET_MEAN, std=IMAGENET_STD, input="rgb uint8 / 255"),
                compute="fp32", storage="float16 npy memmap", frames_done=done)
    if args.stride != 1:
        meta.update(stride=args.stride, n_cached=m, frame_index="frame_index.npy (source frame of each cached row)")
    start = (done // rows) * rows
    print(f"{n} frames, resuming at {start}", flush=True)

    device = torch.device(args.device)
    model = load_dinov2(device)
    loader = torch.utils.data.DataLoader(Chunks(src, start, rows * 5), batch_size=None, num_workers=args.workers,
                                         prefetch_factor=4, persistent_workers=False)
    t0, last_flush = time.time(), time.time()
    for s, frames in loader:
        frames = frames.numpy() if torch.is_tensor(frames) else frames
        end = s + len(frames)
        r0, r1 = np.searchsorted(frame_index, [s, end])        # cache rows whose source frame is in this chunk
        frames = frames[frame_index[r0:r1] - s]
        for b in range(0, len(frames), args.batch):
            p, c = encode(model, frames[b:b + args.batch], device)
            patch[r0 + b:r0 + b + len(p)] = p.cpu().numpy()
            cls[r0 + b:r0 + b + len(p)] = c.cpu().numpy()
        if time.time() - last_flush > 30 or end == n:
            patch.flush(); cls.flush()
            meta["frames_done"] = end
            meta_path.write_text(json.dumps(meta, indent=1))
            last_flush = time.time()
            rate = (end - start) / (time.time() - t0)
            print(f"{end}/{n} frames  {rate:.0f} fr/s  eta {(n - end) / max(rate, 1) / 60:.1f} min", flush=True)

    # gate: re-encode a random sample one frame at a time and compare. Batch size changes fp32
    # rounding, so allow ~1 fp16 step on the largest tokens (|x| up to ~22) and check direction.
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(m, size=min(args.verify_samples, m), replace=False))
    worst, min_cos = 0.0, 1.0
    for i in idx:
        f = int(frame_index[i])
        p, c = encode(model, z["rgb"][f:f + 1], device)
        a, b = p.cpu().numpy()[0].astype(np.float32), patch[i].astype(np.float32)
        worst = max(worst, float(np.abs(a - b).max()),
                    float(np.abs(c.cpu().numpy()[0].astype(np.float32) - cls[i].astype(np.float32)).max()))
        min_cos = min(min_cos, float(((a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)).min()))
        assert np.isfinite(patch[i]).all() and np.isfinite(cls[i]).all(), f"non-finite tokens at frame {i}"
    meta["verify"] = dict(samples=len(idx), max_abs_diff_reencode=worst, min_token_cosine=min_cos)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"verify: {len(idx)} frames re-encoded alone, max |diff| {worst:.2e}, min token cosine {min_cos:.7f}", flush=True)
    assert worst < 0.05 and min_cos > 0.9999, "re-encoding disagrees with the cache"


if __name__ == "__main__":
    main()
