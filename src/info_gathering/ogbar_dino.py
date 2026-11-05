"""OGBar windows on cached DINOv2 tokens, sliced the DINO-WM / LeWM way.

One item = `num_frames` frames at stride `frameskip`, starting at ANY raw frame of an episode
(every offset, not only multiples of the frameskip). Each frame's action is the `frameskip`
raw 5-D actions from that frame to the next one, concatenated (5 x 5 = 25 numbers), as in
DINO-WM's TrajSlicerDataset. Episodes may have different lengths.

Reads the cache written by scripts/cache_dinov2_ogbar.py (patch.npy memmap) and, for the
pixel decoder's target, the matching 224x224 `rgb` frames from the source zarr. Nothing is
loaded whole: tokens and frames are read per item.

Item:
    visual   (F, 256, 384) float16   DINOv2 patch tokens
    proprio  (F, 19)       float32   OGBench proprio = state[:19] (joint pos/vel, effector pos,
                                     cos/sin yaw, gripper opening, gripper contact)
    action   (F, 25)       float32   raw executed actions, concatenated per model step
    rgb      (F, 224, 224, 3) uint8  (only with load_rgb)
    episode, start  int
"""
import json
from pathlib import Path

import numpy as np
import torch
import zarr

PROPRIO_DIM = 19   # OGBench manipspace state layout: proprio first, then the bar's pose


class OgbarDinoWindows(torch.utils.data.Dataset):
    def __init__(self, cache_dir, zarr_path, num_frames=4, frameskip=5, episodes=None, load_rgb=True):
        self.cache_dir, self.zarr_path = str(cache_dir), str(zarr_path)
        self.num_frames, self.frameskip, self.load_rgb = int(num_frames), int(frameskip), bool(load_rgb)
        meta = json.loads((Path(cache_dir) / "meta.json").read_text())
        if meta.get("stride", 1) != 1:
            raise ValueError(f"cache {cache_dir} holds every {meta['stride']}th frame only (frame_index.npy); "
                             "this loader needs every raw frame")
        z = zarr.open_group(self.zarr_path, mode="r")
        n = z["rgb"].shape[0]
        if meta["n_frames"] != n or meta["frames_done"] != n:
            raise ValueError(f"cache {cache_dir} covers {meta['frames_done']}/{meta['n_frames']} frames, source has {n}")
        ends = z["episode_ends"][:].astype(np.int64)
        if not np.array_equal(ends, np.load(Path(cache_dir) / "episode_ends.npy")):
            raise ValueError("cache and source episode_ends differ")
        self.episode_starts = np.concatenate([[0], ends[:-1]])
        self.episode_ends = ends
        self.action_raw = z["action"][:].astype(np.float32)          # (N, 5), ~8 MB
        self.proprio = z["state"][:, :PROPRIO_DIM].astype(np.float32)  # (N, 19), ~30 MB
        span = self.num_frames * self.frameskip                      # raw steps an item needs actions for
        eps = range(len(ends)) if episodes is None else episodes
        starts, epi = [], []
        for e in eps:
            s0, e0 = self.episode_starts[e], ends[e]
            if e0 - s0 >= span:
                st = np.arange(s0, e0 - span + 1)
                starts.append(st)
                epi.append(np.full(len(st), e))
        self.starts = np.concatenate(starts)
        self.start_episode = np.concatenate(epi)
        self.action_dim = 5 * self.frameskip
        self.proprio_dim = PROPRIO_DIM
        self._patch = self._rgb = None

    def stats(self):
        """Mean/std of the raw 5-D action (tiled to 25) and of the proprio, over all frames used."""
        a_mu, a_sd = self.action_raw.mean(0), self.action_raw.std(0)
        p_mu, p_sd = self.proprio.mean(0), self.proprio.std(0)
        return dict(action_mean=np.tile(a_mu, self.frameskip), action_std=np.tile(np.maximum(a_sd, 1e-6), self.frameskip),
                    proprio_mean=p_mu, proprio_std=np.maximum(p_sd, 1e-6))

    def __len__(self):
        return len(self.starts)

    def frame_index(self, i):
        return self.starts[i] + self.frameskip * np.arange(self.num_frames)

    def __getitem__(self, i):
        if self._patch is None:   # opened lazily, once per worker
            self._patch = np.load(Path(self.cache_dir) / "patch.npy", mmap_mode="r")
            self._rgb = zarr.open_group(self.zarr_path, mode="r")["rgb"]
        g = int(self.starts[i])
        idx = self.frame_index(i)
        item = dict(
            visual=torch.from_numpy(np.ascontiguousarray(self._patch[idx])),
            proprio=torch.from_numpy(self.proprio[idx]),
            action=torch.from_numpy(self.action_raw[g:g + self.num_frames * self.frameskip].reshape(self.num_frames, -1)),
            episode=int(self.start_episode[i]), start=g,
        )
        if self.load_rgb:
            item["rgb"] = torch.from_numpy(self._rgb[g:idx[-1] + 1][::self.frameskip])
        return item

    def episode_sequence(self, e, offset=0, load_rgb=True):
        """A whole episode at stride frameskip from raw offset `offset`, for rollouts and videos."""
        if self._patch is None:
            self[0]
        s0, e0 = int(self.episode_starts[e]) + offset, int(self.episode_ends[e])
        n_steps = (e0 - s0) // self.frameskip            # frames whose 5 following actions all exist
        idx = s0 + self.frameskip * np.arange(n_steps)
        out = dict(visual=torch.from_numpy(np.ascontiguousarray(self._patch[idx])),
                   proprio=torch.from_numpy(self.proprio[idx]),
                   action=torch.from_numpy(self.action_raw[s0:s0 + n_steps * self.frameskip].reshape(n_steps, -1)),
                   frame_index=idx)
        if load_rgb:
            out["rgb"] = torch.from_numpy(self._rgb[s0:idx[-1] + 1][::self.frameskip])
        return out
