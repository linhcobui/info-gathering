"""Ablation datasets: variants of info_gathering.dataset.cups_datasets."""
from dataclasses import replace

import numpy as np

from .dataset import META_KEYS, EpisodeWindows, load_episodes


def first_action_segment_end(action):
    """Step right after each episode's first non-zero action segment: (E, S, A) -> (E,)."""
    active = np.abs(action).sum(-1) > 0  # (E, S)
    if not active.any(1).all():
        raise ValueError("every episode needs at least one non-zero action")
    onset = active.argmax(1)
    after = ~active & (np.arange(active.shape[1])[None] > onset[:, None])
    return np.where(after.any(1), after.argmax(1), active.shape[1])


def cups_datasets_first_lift(zarr_path, image_size, frameskip=5, window=None, holdout_episodes=50,
                             cache_dir="data/cache", meta_keys=META_KEYS):
    """cups_datasets with every episode cut right after its first action segment (the first lift).

    For memtest0 that keeps steps 0-42 (frames 0-214): reveal, cover, 5 swaps and lift 1.
    """
    arrays = load_episodes(zarr_path, int(image_size), int(frameskip), cache_dir, tuple(meta_keys))
    ends = first_action_segment_end(arrays.action)
    if not (ends == ends[0]).all():
        raise ValueError(f"episodes end their first action segment at different steps: {np.unique(ends)}")
    S = int(ends[0])
    arrays = replace(arrays, image=arrays.image[:, :S], action=arrays.action[:, :S],
                     meta={k: v[:, :S] for k, v in arrays.meta.items()})
    split = arrays.num_episodes - int(holdout_episodes)
    return (EpisodeWindows(arrays, range(split), window),
            EpisodeWindows(arrays, range(split, arrays.num_episodes), window))
