"""memtest1: single-cup memory episodes. The ball is either under the cup or not.

Phases, one recorded frame per world-model step (no frameskip):
    closed  closed_frames  cup down                                    action 0
    reveal  reveal_frames  cup raises over move_frames: ball or empty   action 1
    hold    H              cup lowers over move_frames, stays down      action 0
    show    show_frames    cup raises again                            action 1
with H = clip(round(N(hold_mean, hold_std)), hold_min, hold_max).

Uses the 3-cup scene from scene_builder: cup 0 stands at x=0, cups 1-2 (and the ball when
absent) are parked at cfg["park_xyz"], behind the camera.
"""
import json

import numpy as np
import zarr

from .choreography import hold, vertical_move

PHASES = {"closed": 0, "reveal": 1, "hold": 2, "show": 3}


def sample_hold(cfg, rng):
    h = int(np.rint(rng.normal(cfg["hold_mean"], cfg["hold_std"])))
    return int(np.clip(h, cfg["hold_min"], cfg["hold_max"]))


def sample_episode(cfg, rng, has_ball, hold_frames=None):
    """Per-frame poses and labels for one episode; hold_frames overrides the sampled hold."""
    rest_z, up_z, move = cfg["cup_rest_z"], cfg["cup_up_z"], cfg["move_frames"]
    hold_frames = sample_hold(cfg, rng) if hold_frames is None else int(hold_frames)
    if min(cfg["reveal_frames"], cfg["show_frames"], hold_frames) < move:
        raise ValueError("reveal_frames, show_frames and the hold must all be >= move_frames")
    park = np.asarray(cfg["park_xyz"], dtype=np.float64)
    cup_pos = np.stack([[0.0, 0.0, rest_z], park, park])
    ball_pos = np.array([0.0, 0.0, cfg["ball_z"]]) if has_ball else park.copy()

    C, B, A, PH = [], [], [], []

    def phase(n, z_to, action, name):
        """n frames: move cup 0 to z_to over the first move frames (if not there yet), then hold."""
        nonlocal cup_pos
        if np.isclose(cup_pos[0, 2], z_to):
            cups, balls = hold(n, cup_pos, ball_pos)
        else:
            c1, b1 = vertical_move(move, cup_pos, ball_pos, [0], cup_pos[0, 2], z_to)
            c2, b2 = hold(n - move, c1[-1], ball_pos)
            cups, balls = np.concatenate([c1, c2]), np.concatenate([b1, b2])
        C.append(cups)
        B.append(balls)
        A.append(np.full((n, 1), action, dtype=np.uint8))
        PH.append(np.full(n, PHASES[name], dtype=np.uint8))
        cup_pos = cups[-1].copy()

    phase(cfg["closed_frames"], rest_z, 0, "closed")
    phase(cfg["reveal_frames"], up_z, 1, "reveal")
    phase(hold_frames, rest_z, 0, "hold")
    phase(cfg["show_frames"], up_z, 1, "show")

    cup_xyz = np.concatenate(C).astype(np.float32)
    return {
        "cup_xyz": cup_xyz,
        "ball_xyz": np.concatenate(B).astype(np.float32),
        "action": np.concatenate(A),
        "phase": np.concatenate(PH),
        "ball_present": np.full(len(cup_xyz), int(has_ball), dtype=np.uint8),
        "cup_z": cup_xyz[:, 0, 2].copy(),
        "hold_frames": hold_frames,
    }


class Memtest1Recorder:
    """Flattened zarr store: per-frame arrays (total_frames, ...) plus per-episode arrays."""

    PER_EPISODE = ("episode_ends", "episode_seeds", "episode_ball", "episode_hold")

    def __init__(self, path, res, append=False):
        if append:
            # add episodes to an existing store; the per-episode arrays are rebuilt by finalize()
            z = self.root = zarr.open_group(str(path), mode="a")
            if int(z.attrs["resolution"]) != int(res):
                raise ValueError(f"{path} is {z.attrs['resolution']}px, cannot append {res}px episodes")
            self.episode_ends = z["episode_ends"][:].tolist()
            self.episode_seeds = z["episode_seeds"][:].tolist()
            self.episode_ball = z["episode_ball"][:].tolist()
            self.episode_hold = z["episode_hold"][:].tolist()
            self.frames = int(self.episode_ends[-1]) if self.episode_ends else 0
            for key in self.PER_EPISODE:
                del z[key]
            return
        z = self.root = zarr.open_group(str(path), mode="w")
        z.create_dataset("rgb", shape=(0, res, res, 3), chunks=(512, res, res, 3), dtype="uint8")
        z.create_dataset("action", shape=(0, 1), chunks=(4096, 1), dtype="uint8")
        z.create_dataset("phase", shape=(0,), chunks=(4096,), dtype="uint8")
        z.create_dataset("ball_present", shape=(0,), chunks=(4096,), dtype="uint8")
        z.create_dataset("cup_z", shape=(0,), chunks=(4096,), dtype="float32")
        self.frames = 0
        self.episode_ends, self.episode_seeds, self.episode_ball, self.episode_hold = [], [], [], []

    def add_episode(self, rgb, ep, seed):
        """rgb: (T, res, res, 3) uint8. ep: dict from sample_episode(). seed: int."""
        z = self.root
        z["rgb"].append(rgb)
        for key in ("action", "phase", "ball_present", "cup_z"):
            z[key].append(ep[key])
        self.frames += len(rgb)
        self.episode_ends.append(self.frames)
        self.episode_seeds.append(int(seed))
        self.episode_ball.append(int(ep["ball_present"][0]))
        self.episode_hold.append(int(ep["hold_frames"]))

    def finalize(self, cfg, master_seed):
        z = self.root
        z.create_dataset("episode_ends", data=np.array(self.episode_ends, dtype="int64"))
        z.create_dataset("episode_seeds", data=np.array(self.episode_seeds, dtype="int64"))
        z.create_dataset("episode_ball", data=np.array(self.episode_ball, dtype="uint8"))
        z.create_dataset("episode_hold", data=np.array(self.episode_hold, dtype="int64"))
        configs = list(z.attrs.get("configs", []))
        z.attrs.update({
            "config": z.attrs.get("config", json.dumps(cfg)),  # the first config that wrote this store
            "configs": configs + [json.dumps(cfg)],
            "master_seed": int(master_seed),
            "n_episodes": len(self.episode_ends),
            "fps": cfg["fps"],
            "resolution": cfg["resolution"],
            "phases": PHASES,
        })
