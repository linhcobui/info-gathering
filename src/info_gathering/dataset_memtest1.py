"""memtest1 datasets for world-model training.

memtest1 episodes vary in length (35-45 steps) because the hold is random. Every episode
ends with the cup fully raised and motionless, so episodes are padded to the longest length
by repeating the final frame -- physically identical to recording a few more "show" frames.
That keeps one fixed-size window per episode, with no loss mask and no model changes.
A `meta/pad` flag marks the repeated steps so evaluation can ignore them.
"""
import numpy as np
import zarr

from .dataset import EpisodeArrays, EpisodeWindows

META_KEYS = ("ball_present", "phase")


def load_memtest1(zarr_path, image_size=64, meta_keys=META_KEYS):
    z = zarr.open_group(str(zarr_path), mode="r")
    res = int(z.attrs["resolution"])
    if int(image_size) != res:
        raise ValueError(f"{zarr_path} is {res}px; train memtest1 at its native resolution")
    ends = z["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    E, T = len(ends), int((ends - starts).max())

    image = np.empty((E, T, res, res, 3), dtype=np.uint8)
    action = np.zeros((E, T, z["action"].shape[1]), dtype=np.float32)
    meta = {k: np.zeros((E, T, *z[k].shape[1:]), dtype=z[k].dtype) for k in meta_keys}
    meta["pad"] = np.zeros((E, T), dtype=np.uint8)
    for e, (s, t) in enumerate(zip(starts, ends)):
        n = int(t - s)
        image[e, :n], image[e, n:] = z["rgb"][s:t], z["rgb"][t - 1]
        action[e, :n], action[e, n:] = z["action"][s:t], z["action"][t - 1]
        for k in meta_keys:
            values = z[k][s:t]
            meta[k][e, :n], meta[k][e, n:] = values, values[-1]
        meta["pad"][e, n:] = 1
    return EpisodeArrays(image, action, meta, frameskip=1, fps=float(z.attrs["fps"]))


def memtest1_datasets(zarr_path, image_size=64, frameskip=1, window=None, holdout_episodes=20,
                      meta_keys=META_KEYS):
    """(train, holdout) windows; the last `holdout_episodes` episodes are held out."""
    if int(frameskip) != 1:
        raise ValueError("memtest1 is recorded at model-step rate: frameskip must be 1")
    arrays = load_memtest1(zarr_path, int(image_size), tuple(meta_keys))
    split = arrays.num_episodes - int(holdout_episodes)
    return (EpisodeWindows(arrays, range(split), window),
            EpisodeWindows(arrays, range(split, arrays.num_episodes), window))
