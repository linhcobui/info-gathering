"""Flattened zarr store for OGBar episodes (same layout idea as info_gathering.memtest4.Memtest4Recorder).

Root (per frame, shape (total_frames, ...)): `rgb`, `action`, plus every per-frame privileged key, delimited by
`episode_ends` / `episode_seeds` (so info_gathering.dataset.load_episodes can read it unchanged).
Groups (flat tables): `episode/<key>` one row per episode; `subtask/<key>` one row per pick-and-place sub-task (play
episodes: `segment/<key>`, one row per play segment); `pickup/<key>` one row per pickup attempt. Every table row
carries its global episode index.

Arrays are created from the first episode's dtypes/shapes and appended afterwards; `append=True` reopens a store.
"""
import json

import numpy as np
import zarr
from numcodecs import Blosc

# Images: zstd level 5 without shuffle stores 224 px frames in ~10% of raw (lz4, zarr's default: ~23%), same decode speed.
IMAGE_COMPRESSOR = Blosc(cname='zstd', clevel=5, shuffle=Blosc.NOSHUFFLE)

OBSERVATION_KEYS = ('rgb', 'rgb_front', 'state')
ACTION_KEYS = ('action',)


def _chunks(shape, n_rows):
    return (n_rows, *shape)


class OgbarRecorder:
    def __init__(self, path, res, append=False):
        self.path = str(path)
        self.res = int(res)
        if append:
            self.root = zarr.open_group(self.path, mode='a')
            if 'resolution' in self.root.attrs and int(self.root.attrs['resolution']) != self.res:
                raise ValueError(f"{path} is {self.root.attrs['resolution']}px, cannot append {res}px episodes")
            ends = self.root['episode_ends'][:]
            self.frames = int(ends[-1]) if len(ends) else 0
            self.n_episodes = len(ends)
        else:
            self.root = zarr.open_group(self.path, mode='w')
            self.frames = 0
            self.n_episodes = 0

    def _append(self, group, key, values, rows_per_chunk, compressor=None):
        values = np.asarray(values)
        node = self.root if group is None else self.root.require_group(group)
        if key not in node:
            kw = {} if compressor is None else {'compressor': compressor}
            node.create_dataset(key, shape=(0, *values.shape[1:]), dtype=values.dtype,
                                chunks=_chunks(values.shape[1:], rows_per_chunk), **kw)
        node[key].append(values)

    def add_episode(self, frames, episode, tables, seed):
        """frames: {key: (T, ...)}; episode: {key: scalar or 1-D}; tables: {group: {key: list per row}}."""
        T = len(frames['rgb'])
        for key, values in frames.items():
            if key.startswith('rgb'):
                # ~8 MB raw per chunk (53 frames at 224 px, 651 at 64 px): a training window decodes little extra
                rows = max(16, int(8e6 // np.asarray(values[0]).nbytes))
                self._append(None, key, values, rows, compressor=IMAGE_COMPRESSOR)
            else:
                self._append(None, key, values, 4096)
        ep_idx = self.n_episodes
        for key, value in episode.items():
            self._append('episode', key, np.asarray(value)[None], 1024)
        for name, table in tables.items():
            n = len(next(iter(table.values()))) if table else 0
            if n == 0:
                continue
            self._append(name, 'episode', np.full(n, ep_idx, dtype=np.int64), 4096)
            for key, values in table.items():
                self._append(name, key, np.asarray(values), 4096)
        self.frames += T
        self.n_episodes += 1
        self._append(None, 'episode_ends', np.array([self.frames], dtype=np.int64), 4096)
        self._append(None, 'episode_seeds', np.array([int(seed)], dtype=np.int64), 4096)

    def finalize(self, cfg, master_seed, extra_attrs=None):
        z = self.root
        privileged = sorted(k for k in z.array_keys()
                            if k not in OBSERVATION_KEYS + ACTION_KEYS + ('episode_ends', 'episode_seeds'))
        configs = list(z.attrs.get('configs', []))
        z.attrs.update({
            'config': z.attrs.get('config', json.dumps(cfg)),
            'configs': configs + [json.dumps(cfg)],
            'master_seed': int(master_seed),
            'n_episodes': self.n_episodes,
            'fps': cfg['fps'],
            'resolution': cfg['resolution'],
            'total_mass': cfg['mass']['total_mass'],
            'mass_offset_d': cfg['mass']['offset_d'],
            'privileged_keys': privileged,
            **(extra_attrs or {}),
        })
