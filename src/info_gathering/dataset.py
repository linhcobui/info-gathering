"""Read the zarr written by ZarrRecorder as frameskipped episode windows.

World models see every `frameskip`-th frame (always 5 here). The raw per-frame
actions inside each chunk are concatenated, so one model step's action has
3 * frameskip dims (15).
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

META_KEYS = ("ball_slot", "phase", "cup_order")


def cache_path(zarr_path, cache_dir, size):
    """Location of the copy of `zarr_path` resized to `size` px."""
    return Path(cache_dir) / f"{Path(zarr_path).stem}_{size}px.zarr"


def _resize_range(args):
    zarr_path, start, stop, size = args
    rgb = zarr.open_group(str(zarr_path), mode="r")["rgb"][start:stop]
    # BOX = area averaging, so the small ball doesn't alias away
    return start, np.stack([np.asarray(Image.fromarray(f).resize((size, size), Image.Resampling.BOX)) for f in rgb])


def build_cache(zarr_path, cache_dir, size, workers=16, frames_per_task=2025):
    """Write a copy of the dataset with rgb resized to `size` px; other arrays are copied as-is."""
    src = zarr.open_group(str(zarr_path), mode="r")
    out = cache_path(zarr_path, cache_dir, size)
    tmp = out.with_name(out.name + ".tmp")
    dst = zarr.open_group(str(tmp), mode="w")
    n = src["rgb"].shape[0]
    episode_len = int(src["episode_ends"][0])
    rgb = dst.create_dataset("rgb", shape=(n, size, size, 3), chunks=(episode_len, size, size, 3), dtype="uint8")
    tasks = [(str(zarr_path), s, min(s + frames_per_task, n), size) for s in range(0, n, frames_per_task)]
    with ProcessPoolExecutor(workers) as pool:
        for start, frames in tqdm(pool.map(_resize_range, tasks), total=len(tasks), desc=f"resize to {size}px"):
            rgb[start : start + len(frames)] = frames
    for key in src.array_keys():
        if key != "rgb":
            dst.create_dataset(key, data=src[key][:], chunks=src[key].chunks)
    dst.attrs.update(src.attrs.asdict())
    dst.attrs.update({"resolution": size, "resized_from": str(zarr_path), "resize_filter": "PIL BOX"})
    tmp.rename(out)
    return out


@dataclass
class EpisodeArrays:
    """All episodes in memory, at model-step rate."""

    image: np.ndarray  # (E, S, H, W, 3) uint8, first frame of each chunk
    action: np.ndarray  # (E, S, frameskip * 3) float32, raw actions of the chunk concatenated
    meta: dict  # key -> (E, S, ...) labels at each step's frame
    frameskip: int
    fps: float  # model steps per second of real time

    @property
    def num_episodes(self):
        return self.image.shape[0]

    @property
    def steps(self):
        return self.image.shape[1]


def load_episodes(zarr_path, image_size, frameskip=5, cache_dir="data/cache", meta_keys=META_KEYS,
                  episodes_per_read=50):
    src = zarr.open_group(str(zarr_path), mode="r")
    path = Path(zarr_path)
    if image_size != int(src.attrs["resolution"]):
        path = cache_path(zarr_path, cache_dir, image_size)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; build it with: pixi run preprocess-frames --size {image_size}")
    z = zarr.open_group(str(path), mode="r")

    lengths = np.diff(np.concatenate([[0], z["episode_ends"][:]]))
    if not (lengths == lengths[0]).all():
        raise ValueError(f"{path}: episodes must all have the same length")
    E, L = len(lengths), int(lengths[0])
    S = L // frameskip  # a trailing partial chunk is dropped
    frames = np.arange(S) * frameskip

    image = np.empty((E, S, image_size, image_size, 3), dtype=np.uint8)
    for e0 in tqdm(range(0, E, episodes_per_read), desc=f"load {path.name}"):
        e1 = min(e0 + episodes_per_read, E)
        block = z["rgb"][e0 * L : e1 * L].reshape(e1 - e0, L, image_size, image_size, 3)
        image[e0:e1] = block[:, frames]
    # (E*L, 3) -> (E, S*frameskip, 3) -> (E, S, frameskip*3)
    action = z["action"][:].reshape(E, L, -1)[:, : S * frameskip].reshape(E, S, -1).astype(np.float32)
    meta = {k: z[k][:].reshape(E, L, *z[k].shape[1:])[:, frames] for k in meta_keys}
    return EpisodeArrays(image, action, meta, frameskip, float(z.attrs["fps"]) / frameskip)


class EpisodeWindows(Dataset):
    """Windows of `window` consecutive model steps (whole episodes when window is None).

    Items (DataLoader adds a leading batch dim):
        image        (T, H, W, 3) uint8
        action       (T, A) float32  taken at step t, produces image[t + 1]
        prev_action  (T, A) float32  the action leading into step t; zeros at an episode start
        is_first     (T,) bool       True only at step 0 of an episode
        episode, t0  () int64
        meta/<key>   (T, ...)        labels, never a model input
    """

    def __init__(self, arrays, episodes, window=None, stride=1):
        self.arrays = arrays
        self.episodes = np.asarray(episodes, dtype=np.int64)
        self.window = arrays.steps if window is None else int(window)
        if not 1 <= self.window <= arrays.steps:
            raise ValueError(f"window {self.window} outside [1, {arrays.steps}]")
        starts = np.arange(0, arrays.steps - self.window + 1, stride)
        self.index = np.stack(np.meshgrid(self.episodes, starts, indexing="ij"), -1).reshape(-1, 2)

    def with_window(self, window, stride=1):
        return EpisodeWindows(self.arrays, self.episodes, window, stride)

    @property
    def action_dim(self):
        return self.arrays.action.shape[-1]

    @property
    def image_size(self):
        return self.arrays.image.shape[2]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t0 = (int(v) for v in self.index[i])
        t1 = t0 + self.window
        action = self.arrays.action[e]
        prev_action = np.zeros((self.window, action.shape[-1]), dtype=np.float32)
        prev_action[1:] = action[t0 : t1 - 1]
        if t0 > 0:
            prev_action[0] = action[t0 - 1]
        is_first = np.zeros(self.window, dtype=bool)
        is_first[0] = t0 == 0
        item = {
            "image": torch.from_numpy(self.arrays.image[e, t0:t1]),
            "action": torch.from_numpy(action[t0:t1]),
            "prev_action": torch.from_numpy(prev_action),
            "is_first": torch.from_numpy(is_first),
            "episode": torch.tensor(e),
            "t0": torch.tensor(t0),
        }
        item.update({f"meta/{k}": torch.from_numpy(v[e, t0:t1]) for k, v in self.arrays.meta.items()})
        return item


def cups_datasets(zarr_path, image_size, frameskip=5, window=None, holdout_episodes=50, cache_dir="data/cache",
                  meta_keys=META_KEYS):
    """(train, holdout) windows over one in-memory copy; the last `holdout_episodes` episodes are held out."""
    arrays = load_episodes(zarr_path, int(image_size), int(frameskip), cache_dir, tuple(meta_keys))
    split = arrays.num_episodes - int(holdout_episodes)
    return (EpisodeWindows(arrays, range(split), window),
            EpisodeWindows(arrays, range(split, arrays.num_episodes), window))
