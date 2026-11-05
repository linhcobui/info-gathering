"""OGBar anchors for the SMT world model (docs/OGBAR_SMT.md), on cached DINOv2 tokens.

One item = one anchor: a raw frame r anywhere in an episode (every offset, LeWM-style). The model
step sequence is the episode's frames at stride `frameskip` from r's offset, s0 + o, s0 + o + 5, ...,
and the anchor is the model step t that frame r is. Each step's action a_s is the 5 raw actions
from frame s to s+1, concatenated (25 numbers); a step is kept only if all 5 exist.

The teacher reads 200 steps ending at t (and at t+1), the decoder up to 200 steps after t. Episodes
are at most 200 steps at stride 5, so together they always span the whole episode: the item IS the
episode at that offset (<= 200 frames of 256 x 384 fp16, ~39 MB) plus the index t. Anchors come
from random episodes; nothing is shared between items.

Item:
    patch   (S, 256, 384) float16   DINOv2 patch tokens of every step
    action  (S, 25)       float32   a_s, the actions taken AT step s
    length  int                     S
    t       int                     the anchor step, 0 <= t <= S - 2 (x_{t+1} exists)
    episode, offset  int
    cview   float  (privileged=True only) c_view at the anchor step t, cm: the COM offset signed by which side
                   of the side-view image the heavy end is on (+ = image right; camera D's image-right is world -x).
                   PRIVILEGED: read only by the alignment loss, never by the model's inputs
`collate` pads S to the batch maximum (zeros; `length` says what is real).
"""
import json
from pathlib import Path

import numpy as np
import torch
import zarr


class OgbarSmtAnchors(torch.utils.data.Dataset):
    def __init__(self, cache_dir, zarr_path, frameskip=5, episodes=None, privileged=False):
        self.cache_dir, self.k = str(cache_dir), int(frameskip)
        meta = json.loads((Path(cache_dir) / "meta.json").read_text())
        z = zarr.open_group(str(zarr_path), mode="r")
        n = z["rgb"].shape[0]
        if meta["n_frames"] != n or meta["frames_done"] != n:
            raise ValueError(f"cache {cache_dir} covers {meta['frames_done']}/{meta['n_frames']} frames, source has {n}")
        ends = z["episode_ends"][:].astype(np.int64)
        if not np.array_equal(ends, np.load(Path(cache_dir) / "episode_ends.npy")):
            raise ValueError("cache and source episode_ends differ")
        self.starts, self.ends = np.concatenate([[0], ends[:-1]]), ends
        self.action_raw = z["action"][:].astype(np.float32)               # (N, 5), small
        self.cview_raw = None
        if privileged:                                                    # per raw frame, cm (scripts/belief_midframe_ogbar.py)
            q = z["bar_quat"][:].astype(np.float64)
            ax = 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)                    # bar body x axis, world x component
            c = np.repeat(z["episode/com_offset_c"][:].astype(np.float64) * 100, np.diff(np.concatenate([[0], ends])))
            self.cview_raw = (-c * np.sign(ax)).astype(np.float32)
        eps = range(len(ends)) if episodes is None else list(episodes)
        rows = []
        for e in eps:
            L = int(ends[e] - self.starts[e])
            for o in range(min(self.k, L)):
                S = (L - o) // self.k                                     # steps with all 5 actions
                if S >= 2:
                    t = np.arange(S - 1)                                  # x_{t+1} must exist
                    rows.append(np.stack([np.full_like(t, e), np.full_like(t, o), t], 1))
        self.anchors = np.concatenate(rows)                               # (n_anchors, 3): episode, offset, t
        self.action_dim = 5 * self.k
        self.max_steps = int(max((self.ends - self.starts).max() // self.k, 1))
        self._patch = None

    def __len__(self):
        return len(self.anchors)

    def episode_steps(self, e, o):
        """Raw frame indices of episode e's steps at offset o."""
        s0, L = int(self.starts[e]), int(self.ends[e] - self.starts[e])
        return s0 + o + self.k * np.arange((L - o) // self.k)

    def load(self, e, o):
        if self._patch is None:                                           # opened lazily, once per worker
            self._patch = np.load(Path(self.cache_dir) / "patch.npy", mmap_mode="r")
        idx = self.episode_steps(e, o)
        act = np.stack([self.action_raw[i:i + self.k].reshape(-1) for i in idx])
        return torch.from_numpy(np.ascontiguousarray(self._patch[idx])), torch.from_numpy(act)

    def __getitem__(self, i):
        e, o, t = (int(v) for v in self.anchors[i])
        patch, act = self.load(e, o)
        out = dict(patch=patch, action=act, length=len(patch), t=t, episode=e, offset=o)
        if self.cview_raw is not None:
            out["cview"] = float(self.cview_raw[self.episode_steps(e, o)[t]])
        return out


class AnchorBatches(torch.utils.data.Dataset):
    """Whole batches read straight into one shared-memory tensor (no per-item copy, no second copy when the
    worker hands it over). Use with DataLoader(batch_size=None, sampler=BatchSampler(...))."""

    def __init__(self, anchors):
        self.a = anchors

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        a = self.a
        if a._patch is None:
            a._patch = np.load(Path(a.cache_dir) / "patch.npy", mmap_mode="r")
        rows = [tuple(int(v) for v in a.anchors[i]) for i in idx]
        steps = [a.episode_steps(e, o) for e, o, _ in rows]
        S = max(len(s) for s in steps)
        patch = torch.zeros(len(rows), S, *a._patch.shape[1:], dtype=torch.float16).share_memory_()
        action = torch.zeros(len(rows), S, a.action_dim)
        pn = patch.numpy()
        for b, st in enumerate(steps):
            pn[b, :len(st)] = a._patch[st]
            action[b, :len(st)] = torch.from_numpy(np.stack([a.action_raw[i:i + a.k].reshape(-1) for i in st]))
        out = dict(patch=patch, action=action, length=torch.tensor([len(s) for s in steps]),
                   t=torch.tensor([r[2] for r in rows]), episode=torch.tensor([r[0] for r in rows]),
                   offset=torch.tensor([r[1] for r in rows]))
        if a.cview_raw is not None:
            out["cview"] = torch.tensor([float(a.cview_raw[st[r[2]]]) for st, r in zip(steps, rows)])
        return out


def collate(items):
    S = max(it["length"] for it in items)
    B = len(items)
    p0 = items[0]["patch"]
    patch = torch.zeros(B, S, *p0.shape[1:], dtype=p0.dtype)
    action = torch.zeros(B, S, items[0]["action"].shape[-1])
    for b, it in enumerate(items):
        n = it["length"]
        patch[b, :n] = it["patch"]
        action[b, :n] = it["action"]
    out = dict(patch=patch, action=action)
    for key in ("length", "t", "episode", "offset"):
        out[key] = torch.tensor([it[key] for it in items])
    if "cview" in items[0]:
        out["cview"] = torch.tensor([it["cview"] for it in items])
    return out
