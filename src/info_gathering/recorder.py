"""Flattened zarr recorder for shell-game episodes."""
import json
import numpy as np
import zarr


class ZarrRecorder:
    """Append episodes to a flattened zarr store.
    Arrays are (total_frames, ...) with episode_ends marking boundaries."""

    def __init__(self, path, res):
        self.path = path
        self.res = res
        self.root = zarr.open_group(path, mode="w")
        self._init_arrays()
        self.episode_ends = []
        self.episode_seeds = []
        self._cursor = 0

    def _init_arrays(self):
        r = self.res
        z = self.root
        # resizable along axis 0; chunk ~one block of frames
        z.create_dataset("rgb", shape=(0, r, r, 3), chunks=(64, r, r, 3),
                         dtype="uint8")
        z.create_dataset("action", shape=(0, 3), chunks=(4096, 3), dtype="uint8")
        z.create_dataset("ball_slot", shape=(0,), chunks=(4096,), dtype="uint8")
        z.create_dataset("cup_order", shape=(0, 3), chunks=(4096, 3), dtype="uint8")
        z.create_dataset("phase", shape=(0,), chunks=(4096,), dtype="uint8")

    def add_episode(self, rgb, ep, seed):
        """rgb: (T,res,res,3) uint8. ep: dict from choreograph(). seed: int."""
        T = rgb.shape[0]
        z = self.root
        z["rgb"].append(rgb)
        z["action"].append(ep["action"])
        z["ball_slot"].append(ep["ball_slot"])
        z["cup_order"].append(ep["cup_order"])
        z["phase"].append(ep["phase"])
        self._cursor += T
        self.episode_ends.append(self._cursor)
        self.episode_seeds.append(int(seed))

    def finalize(self, cfg, master_seed):
        z = self.root
        z.create_dataset("episode_ends", data=np.array(self.episode_ends, dtype="int64"))
        z.create_dataset("episode_seeds", data=np.array(self.episode_seeds, dtype="int64"))
        z.attrs["config"] = json.dumps(cfg)
        z.attrs["master_seed"] = int(master_seed)
        z.attrs["n_episodes"] = len(self.episode_ends)
        z.attrs["fps"] = cfg["fps"]
        z.attrs["resolution"] = cfg["resolution"]
