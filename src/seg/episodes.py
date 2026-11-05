"""Episodes as model steps (frameskip k, raw offset 0), read from an OGBar zarr store and its DINOv2 cache.

Model frame t of an episode is raw frame episode_start + k t; its action is the k raw 5-D actions from frame t to
frame t+1, concatenated (as in OgbarDinoWindows.episode_sequence). An episode has T = length // k frames: the last
raw frames without a full action chunk after them are dropped. The cache may hold every raw frame or only every
k-th (scripts/cache_dinov2_ogbar.py --stride k, rows mapped by frame_index.npy). Patch tokens are read per episode
from the memmap, never the whole cache.
"""
import json

import numpy as np
import zarr

from .config import ROOT

PROPRIO_DIM = 19   # OGBench state layout: proprio first, then the bar's pose (never an input)


class Episode:
    """One episode. visual (T, P, D) is either held (toy data) or read from a cache memmap on load_visual().

    events: privileged {name: model steps} used by diagnostics only (e.g. grasp onsets), never by the pipeline.
    """

    def __init__(self, episode_id, action, proprio, visual=None, frame_index=None, wm_train=False, store="", index=0,
                 patch=None, rows=None, rgb=None, events=None, render=None):
        self.episode_id, self.store, self.index, self.wm_train = episode_id, store, int(index), bool(wm_train)
        self.action = np.asarray(action, dtype=np.float32)
        self.proprio = np.asarray(proprio, dtype=np.float32)
        self.visual = visual
        self.frame_index = np.arange(len(self.action)) if frame_index is None else np.asarray(frame_index)
        self._patch, self._rows, self._rgb, self._render = patch, rows, rgb, render
        self.events = events or {}

    @property
    def T(self):
        return len(self.action)

    def load_visual(self):
        """(T, P, D) patch tokens (float16 for cached episodes)."""
        if self.visual is not None:
            return self.visual
        return np.asarray(self._patch[self._rows])

    def frame_rgb(self, t):
        """(h, w, 3) uint8 image of model frame t, or None when the episode has no images."""
        if self._render is not None:
            return self._render(self, t)
        if self._rgb is None:
            return None
        return np.asarray(self._rgb[int(self.frame_index[t])])


def load_store(store, frameskip):
    """All episodes of one `stores:` entry of the config."""
    z = zarr.open_group(str(ROOT / store.zarr), mode="r")
    cache = ROOT / store.cache
    meta = json.loads((cache / "meta.json").read_text())
    n = z["rgb"].shape[0]
    if meta["n_frames"] != n or meta["frames_done"] != n:
        raise ValueError(f"cache {cache} covers {meta['frames_done']}/{meta['n_frames']} frames, source has {n}")
    ends = z["episode_ends"][:].astype(np.int64)
    if not np.array_equal(ends, np.load(cache / "episode_ends.npy")):
        raise ValueError(f"cache {cache} and {store.zarr} disagree on episode_ends")
    stride = int(meta.get("stride", 1))
    if stride not in (1, frameskip):
        raise ValueError(f"cache stride {stride} does not match frameskip {frameskip}")
    cached = np.load(cache / "frame_index.npy") if stride != 1 else None
    patch = np.load(cache / "patch.npy", mmap_mode="r")
    action_raw = z["action"][:].astype(np.float32)
    proprio = z["state"][:, :PROPRIO_DIM].astype(np.float32)
    grasped = z["grasped"][:].astype(bool) if "grasped" in z else None
    starts = np.concatenate([[0], ends[:-1]])
    episodes = []
    for e, (s0, e0) in enumerate(zip(starts, ends)):
        T = int(e0 - s0) // frameskip
        if T == 0:
            continue
        fi = s0 + frameskip * np.arange(T)
        rows = fi if cached is None else np.searchsorted(cached, fi)
        if cached is not None and not np.array_equal(cached[rows], fi):
            raise ValueError(f"cache {cache} lacks offset-0 frames of episode {e}")
        events = {}
        if grasped is not None:   # first model frame at or after each raw grasp onset
            g = grasped[s0:e0]
            on = np.flatnonzero(g[1:] & ~g[:-1]) + 1
            events["grasp"] = np.unique(-(-on // frameskip))
        episodes.append(Episode(f"{store.name}/ep{e:03d}", action_raw[s0:s0 + T * frameskip].reshape(T, -1),
                                proprio[fi], frame_index=fi, wm_train=store.wm_train, store=store.name, index=e,
                                patch=patch, rows=rows, rgb=z["rgb"], events=events))
    return episodes


def load_episodes(cfg):
    return [ep for store in cfg.stores for ep in load_store(store, cfg.frameskip)]
