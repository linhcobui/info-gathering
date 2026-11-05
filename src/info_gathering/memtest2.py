"""memtest2: two cups, the ball always under one of them, and a 50/50 swap during the hold.

Phases, one recorded frame per world-model step (no frameskip):
    closed  closed_frames  both cups down                                  action [0, 0]
    reveal  reveal_frames  both cups rise over move_frames, ball visible    action [1, 0]
    hold    H              both cups lower, then a swap or nothing          action [0, 0]
    show    show_frames    the cup standing at target_slot rises            action [0, 1]

H = clip(round(N(hold_mean, hold_std)), hold_min, hold_max) and must fit move_frames +
swap_frames, so swap and no-swap episodes share one hold distribution and the timing never
reveals which happened.

The ball rides its cup through a swap, so "is the ball under the lifted cup" needs both the
start side and whether the swap happened. Per-frame ball_slot is read off the ball's actual x
position, so labels never lead the pixels.

Scene: the 3-cup scene from scene_builder with cup 2 parked off camera; cups 0 and 1 stand at
-cup_spacing/2 and +cup_spacing/2.
"""
import json

import numpy as np
import zarr

from .choreography import hold, swap_arc, vertical_move

PHASES = {"closed": 0, "reveal": 1, "hold": 2, "show": 3}


def hold_base(cfg, n_swaps):
    """Frames the hold must contain: lowering the cups, then each swap (plus any gaps)."""
    n_swaps, gap = int(n_swaps), int(cfg.get("swap_gap", 0))
    return int(cfg["move_frames"]) + int(cfg["swap_frames"]) * n_swaps + max(0, n_swaps - 1) * gap


def sample_hold(cfg, rng, n_swaps=0):
    """Hold length in frames.

    With slack_* configured the hold is cumulative -- hold_base(n_swaps) + slack, where
    slack = clip(round(N(slack_mean, slack_std)), slack_min, slack_max) -- so every swap count
    gets the room it needs and the slack keeps the three length ranges overlapping.
    Otherwise the hold is drawn from one fixed distribution (memtest2's behaviour).
    """
    if "slack_mean" in cfg:
        slack = int(np.rint(rng.normal(cfg["slack_mean"], cfg["slack_std"])))
        return hold_base(cfg, n_swaps) + int(np.clip(slack, cfg["slack_min"], cfg["slack_max"]))
    h = int(np.rint(rng.normal(cfg["hold_mean"], cfg["hold_std"])))
    return int(np.clip(h, cfg["hold_min"], cfg["hold_max"]))


def sample_episode(cfg, rng, ball_start, n_swaps=0, hold_frames=None):
    """Per-frame poses and labels for one episode. n_swaps cups swaps happen during the hold."""
    rest, up = cfg["cup_rest_z"], cfg["cup_up_z"]
    move, swap_n = int(cfg["move_frames"]), int(cfg["swap_frames"])
    n_swaps, gap = int(n_swaps), int(cfg.get("swap_gap", 0))
    hold_frames = sample_hold(cfg, rng, n_swaps) if hold_frames is None else int(hold_frames)
    needed = move + n_swaps * swap_n + max(0, n_swaps - 1) * gap
    if hold_frames < needed:
        raise ValueError(f"hold {hold_frames} too short: needs {move} to lower + "
                         f"{n_swaps} x {swap_n} to swap (+{gap} between swaps)")
    if min(cfg["reveal_frames"], cfg["show_frames"]) < move:
        raise ValueError("reveal_frames and show_frames must be >= move_frames")

    half = cfg["cup_spacing"] / 2
    xs = np.array([-half, half])                     # slot 0 = left, slot 1 = right
    park = np.asarray(cfg["park_xyz"], dtype=np.float64)
    target = int(cfg["target_slot"])
    slot_of_identity = np.array([0, 1])              # cup identity -> slot it stands in
    ball_identity = int(ball_start)                  # the ball rides this cup
    cup_pos = np.stack([[xs[0], 0.0, rest], [xs[1], 0.0, rest], park])
    ball_pos = np.array([xs[ball_start], 0.0, cfg["ball_z"]])

    C, B, A, PH = [], [], [], []

    def emit(cups, balls, action, phase):
        C.append(cups)
        B.append(balls)
        A.append(np.repeat(np.array(action, dtype=np.uint8)[None], len(cups), axis=0))
        PH.append(np.full(len(cups), PHASES[phase], dtype=np.uint8))

    # closed: nothing visible
    cups, balls = hold(cfg["closed_frames"], cup_pos, ball_pos)
    emit(cups, balls, [0, 0], "closed")

    # reveal: both cups rise, the ball is visible under one of them
    c1, b1 = vertical_move(move, cup_pos, ball_pos, [0, 1], rest, up)
    c2, b2 = hold(cfg["reveal_frames"] - move, c1[-1], ball_pos)
    emit(np.concatenate([c1, c2]), np.concatenate([b1, b2]), [1, 0], "reveal")
    cup_pos = c2[-1].copy() if len(c2) else c1[-1].copy()

    # hold: both cups come down, then a swap or nothing, then wait
    c3, b3 = vertical_move(move, cup_pos, ball_pos, [0, 1], up, rest)
    cups, balls = [c3], [b3]
    cup_pos = c3[-1].copy()
    for k in range(n_swaps):
        if k and gap:
            cg, bg = hold(gap, cup_pos, ball_pos)
            cups.append(cg)
            balls.append(bg)
        c4, b4 = swap_arc(swap_n, cup_pos, ball_pos, 0, 1, cfg["cup_spacing"],
                          cfg["arc_height_front"], cfg["arc_height_back"], ball_rides_id=ball_identity)
        cups.append(c4)
        balls.append(b4)
        cup_pos, ball_pos = c4[-1].copy(), b4[-1].copy()
        slot_of_identity = slot_of_identity[::-1].copy()
    c5, b5 = hold(hold_frames - needed, cup_pos, ball_pos)
    cups.append(c5)
    balls.append(b5)
    emit(np.concatenate(cups), np.concatenate(balls), [0, 0], "hold")
    cup_pos = np.concatenate(cups)[-1].copy()

    # show: whichever cup now stands at the target slot rises
    lifted = int(np.flatnonzero(slot_of_identity == target)[0])
    c6, b6 = vertical_move(move, cup_pos, ball_pos, [lifted], rest, up)
    c7, b7 = hold(cfg["show_frames"] - move, c6[-1], ball_pos)
    emit(np.concatenate([c6, c7]), np.concatenate([b6, b7]), [0, 1], "show")

    cup_xyz = np.concatenate(C).astype(np.float32)
    ball_xyz = np.concatenate(B).astype(np.float32)
    # read the ball's slot off its actual position, so the label never leads the pixels
    ball_slot = np.argmin(np.abs(ball_xyz[:, :1] - xs[None, :]), axis=1).astype(np.uint8)
    return {
        "cup_xyz": cup_xyz,
        "ball_xyz": ball_xyz,
        "action": np.concatenate(A),
        "phase": np.concatenate(PH),
        "ball_slot": ball_slot,
        "ball_under_target": (ball_slot == target).astype(np.uint8),
        "cup_z_max": cup_xyz[:, :2, 2].max(axis=1),
        "hold_frames": hold_frames,
        "ball_start": int(ball_start),
        "swapped": n_swaps,          # number of swaps during the hold (0, 1, 2, ...)
        "target_slot": target,
    }


class Memtest2Recorder:
    """Flattened zarr store: per-frame arrays (total_frames, ...) plus per-episode arrays."""

    PER_FRAME = ("action", "phase", "ball_slot", "ball_under_target", "cup_z_max")
    PER_EPISODE = ("episode_ends", "episode_seeds", "episode_ball_start", "episode_swapped",
                   "episode_target", "episode_hold")

    def __init__(self, path, res, append=False):
        if append:
            z = self.root = zarr.open_group(str(path), mode="a")
            if int(z.attrs["resolution"]) != int(res):
                raise ValueError(f"{path} is {z.attrs['resolution']}px, cannot append {res}px episodes")
            self.episodes = {k: z[k][:].tolist() for k in self.PER_EPISODE}
            self.frames = int(self.episodes["episode_ends"][-1]) if self.episodes["episode_ends"] else 0
            for key in self.PER_EPISODE:
                del z[key]
            return
        z = self.root = zarr.open_group(str(path), mode="w")
        z.create_dataset("rgb", shape=(0, res, res, 3), chunks=(512, res, res, 3), dtype="uint8")
        z.create_dataset("action", shape=(0, 2), chunks=(4096, 2), dtype="uint8")
        for key in ("phase", "ball_slot", "ball_under_target"):
            z.create_dataset(key, shape=(0,), chunks=(4096,), dtype="uint8")
        z.create_dataset("cup_z_max", shape=(0,), chunks=(4096,), dtype="float32")
        self.frames = 0
        self.episodes = {k: [] for k in self.PER_EPISODE}

    def add_episode(self, rgb, ep, seed):
        """rgb: (T, res, res, 3) uint8. ep: dict from sample_episode(). seed: int."""
        z = self.root
        z["rgb"].append(rgb)
        for key in self.PER_FRAME:
            z[key].append(ep[key])
        self.frames += len(rgb)
        for key, value in (("episode_ends", self.frames), ("episode_seeds", int(seed)),
                           ("episode_ball_start", ep["ball_start"]), ("episode_swapped", ep["swapped"]),
                           ("episode_target", ep["target_slot"]), ("episode_hold", ep["hold_frames"])):
            self.episodes[key].append(value)

    def finalize(self, cfg, master_seed):
        z = self.root
        dtypes = {"episode_ends": "int64", "episode_seeds": "int64", "episode_hold": "int64",
                  "episode_ball_start": "uint8", "episode_swapped": "uint8", "episode_target": "uint8"}
        for key in self.PER_EPISODE:
            z.create_dataset(key, data=np.array(self.episodes[key], dtype=dtypes[key]))
        configs = list(z.attrs.get("configs", []))
        z.attrs.update({
            "config": z.attrs.get("config", json.dumps(cfg)),
            "configs": configs + [json.dumps(cfg)],
            "master_seed": int(master_seed),
            "n_episodes": len(self.episodes["episode_ends"]),
            "fps": cfg["fps"],
            "resolution": cfg["resolution"],
            "phases": PHASES,
        })
