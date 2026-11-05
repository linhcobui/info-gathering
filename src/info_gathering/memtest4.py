"""memtest4/5: two or three cups (n_cups), the ball under one, and 0-4 swaps of random pairs.

Phases, one recorded frame per world-model step (no frameskip). The episode starts with the
cups already up, so there is no initial closed phase:
    reveal  reveal_frames   cups up, the ball is visible under one           action [0, 0]
    hold    4 + 6n + slack  cups lower, then n swaps of a random pair        action [1, 0]
    show    show_frames     the cup standing at target_slot rises            action [0, 1]

Each swap picks a uniformly random pair of the three cups, so the ball's cup is left alone
about a third of the time: the ball's position is a walk over three slots rather than a parity
bit, which is what memtest2/memtest3 reduced to. Unlike memtest0 every swap is fully visible,
so a failure means the model cannot track rather than that the information was never shown.

Both actions land on their phase's first frame while the cups are still at their old height,
so the onset step never gives the answer away. Per-frame ball_slot is tracked through the swaps
rather than read off the ball's x: during a swap it holds the start slot and switches to the
final slot on the swap's last frame, when the ball has arrived. Reading it off x instead would
briefly report slot 1 during a 0<->2 swap, as the arc passes over the middle slot.
"""
import json

import numpy as np
import zarr

from .choreography import hold, slot_positions, swap_arc, vertical_move

PHASES = {"reveal": 0, "hold": 1, "show": 2}


def pairs_for(n_cups):
    """Every unordered pair of cups; with 2 cups there is only one."""
    return [(a, b) for a in range(n_cups) for b in range(a + 1, n_cups)]


def hold_base(cfg, n_swaps):
    """Frames the hold must contain: lowering the cups, then each swap (plus any gaps)."""
    n_swaps, gap = int(n_swaps), int(cfg.get("swap_gap", 0))
    return int(cfg["move_frames"]) + int(cfg["swap_frames"]) * n_swaps + max(0, n_swaps - 1) * gap


def sample_hold(cfg, rng, n_swaps=0):
    """hold_base(n_swaps) + clip(round(N(slack_mean, slack_std)), slack_min, slack_max)."""
    slack = int(np.rint(rng.normal(cfg["slack_mean"], cfg["slack_std"])))
    return hold_base(cfg, n_swaps) + int(np.clip(slack, cfg["slack_min"], cfg["slack_max"]))


def sample_episode(cfg, rng, ball_start, n_swaps=0, hold_frames=None, closed_start=False, peek_after=None):
    """Per-frame poses and labels for one episode; n_swaps random-pair swaps during the hold.

    closed_start  the cups already stand over the ball from frame 0, so its start slot is never shown.
                  The reveal frames show the covered scene and the 4 lowering frames are static, so
                  every action and every timing is unchanged.
    peek_after    None, or k in 0..n_swaps: once k swaps are done, the cup standing at target_slot
                  rises (move_frames + peek_up_frames, action [0, 1], phase show), then comes back
                  down (move_frames, action [1, 0], phase hold) before the remaining swaps. The ball
                  is seen under it, or the cup is seen empty. Adds 2 * move_frames + peek_up_frames
                  frames on top of hold_frames, which keeps its meaning (lowering + swaps + slack).
    Neither draws from rng, so the same seed gives the same swaps and slack with or without them.
    """
    rest, up = cfg["cup_rest_z"], cfg["cup_up_z"]
    move, swap_n = int(cfg["move_frames"]), int(cfg["swap_frames"])
    n_swaps, gap = int(n_swaps), int(cfg.get("swap_gap", 0))
    hold_frames = sample_hold(cfg, rng, n_swaps) if hold_frames is None else int(hold_frames)
    needed = hold_base(cfg, n_swaps)
    if hold_frames < needed:
        raise ValueError(f"hold {hold_frames} too short: needs {move} to lower + "
                         f"{n_swaps} x {swap_n} to swap (+{gap} between swaps)")
    if min(cfg["reveal_frames"], cfg["show_frames"]) < move:
        raise ValueError("reveal_frames and show_frames must be >= move_frames")

    n_cups = int(cfg.get("n_cups", 3))
    xs = slot_positions(cfg["cup_spacing"])[:n_cups]  # slots on the same grid: -spacing, 0, +spacing
    pairs = pairs_for(n_cups)
    target = int(cfg["target_slot"])
    slot_of_identity = np.arange(n_cups)              # cup identity -> slot it stands in
    ball_identity = int(ball_start)                   # the ball rides this cup
    park = np.asarray(cfg.get("park_xyz", [0.0, -5.0, 0.05]), dtype=np.float64)
    # the scene always has three cups; unused ones sit behind the camera
    cup_pos = np.stack([[xs[k], 0.0, rest if closed_start else up] if k < n_cups else park for k in range(3)])
    ball_pos = np.array([xs[ball_start], 0.0, cfg["ball_z"]])
    peek_up = int(cfg.get("peek_up_frames", 2))
    if peek_after is not None and not 0 <= int(peek_after) <= n_swaps:
        raise ValueError(f"peek_after {peek_after} outside 0..{n_swaps}")

    C, B, A, PH, S = [], [], [], [], []
    pairs_used = []

    def emit(cups, balls, action, phase):
        C.append(cups)
        B.append(balls)
        A.append(np.repeat(np.array(action, dtype=np.uint8)[None], len(cups), axis=0))
        PH.append(np.full(len(cups), PHASES[phase], dtype=np.uint8))

    def ball_slot_now():
        return int(slot_of_identity[ball_identity])

    # reveal: the cups are already up and the ball is in plain view (closed start: already down over it)
    cups, balls = hold(cfg["reveal_frames"], cup_pos, ball_pos)
    emit(cups, balls, [0, 0], "reveal")
    S.append(np.full(len(cups), ball_slot_now()))

    # hold: all three cups come down (closed start: they stay down), then the swaps, then waiting
    if closed_start:
        c1, b1 = hold(move, cup_pos, ball_pos)
    else:
        c1, b1 = vertical_move(move, cup_pos, ball_pos, list(range(n_cups)), up, rest)
    cups, balls = [c1], [b1]
    S.append(np.full(len(c1), ball_slot_now()))
    cup_pos = c1[-1].copy()

    def peek():
        """Flush the hold so far, lift the cup at the target slot, show it, lower it again."""
        nonlocal cups, balls, cup_pos
        emit(np.concatenate(cups), np.concatenate(balls), [1, 0], "hold")
        lifted_now = int(np.flatnonzero(slot_of_identity == target)[0])
        cu, bu = vertical_move(move, cup_pos, ball_pos, [lifted_now], rest, up)
        ch, bh = hold(peek_up, cu[-1], ball_pos)
        emit(np.concatenate([cu, ch]), np.concatenate([bu, bh]), [0, 1], "show")
        S.append(np.full(move + peek_up, ball_slot_now()))
        cd, bd = vertical_move(move, ch[-1], ball_pos, [lifted_now], up, rest)
        cups, balls = [cd], [bd]
        S.append(np.full(move, ball_slot_now()))
        cup_pos = cd[-1].copy()

    for k in range(n_swaps):
        if peek_after is not None and int(peek_after) == k:
            peek()
        if k and gap:
            cg, bg = hold(gap, cup_pos, ball_pos)
            cups.append(cg)
            balls.append(bg)
            S.append(np.full(gap, ball_slot_now()))
        id_a, id_b = pairs[int(rng.integers(len(pairs)))]
        pairs_used.append((id_a, id_b))
        c2, b2 = swap_arc(swap_n, cup_pos, ball_pos, id_a, id_b, cfg["cup_spacing"],
                          cfg["arc_height_front"], cfg["arc_height_back"], ball_rides_id=ball_identity)
        cups.append(c2)
        balls.append(b2)
        cup_pos, ball_pos = c2[-1].copy(), b2[-1].copy()
        before = ball_slot_now()
        slot_of_identity[[id_a, id_b]] = slot_of_identity[[id_b, id_a]]
        # binary label per swap: the start slot while the cups travel, the final slot once the
        # swap has landed (its last frame) -- never an intermediate slot the arc passes over
        S.append(np.array([before] * (swap_n - 1) + [ball_slot_now()]))
    if peek_after is not None and int(peek_after) == n_swaps:
        peek()
    c3, b3 = hold(hold_frames - needed, cup_pos, ball_pos)
    cups.append(c3)
    balls.append(b3)
    S.append(np.full(len(c3), ball_slot_now()))
    emit(np.concatenate(cups), np.concatenate(balls), [1, 0], "hold")
    cup_pos = np.concatenate(cups)[-1].copy()

    # show: whichever cup now stands at the target slot rises
    lifted = int(np.flatnonzero(slot_of_identity == target)[0])
    c4, b4 = vertical_move(move, cup_pos, ball_pos, [lifted], rest, up)
    c5, b5 = hold(cfg["show_frames"] - move, c4[-1], ball_pos)
    emit(np.concatenate([c4, c5]), np.concatenate([b4, b5]), [0, 1], "show")
    S.append(np.full(len(c4) + len(c5), ball_slot_now()))

    cup_xyz = np.concatenate(C).astype(np.float32)
    ball_xyz = np.concatenate(B).astype(np.float32)
    ball_slot = np.concatenate(S).astype(np.uint8)
    if len(ball_slot) != len(ball_xyz):
        raise AssertionError(f"slot labels ({len(ball_slot)}) out of step with frames ({len(ball_xyz)})")
    # wherever the ball is at rest (every frame except the middle of a swap), the tracked slot
    # must match the slot nearest its actual position
    nearest = np.argmin(np.abs(ball_xyz[:, :1] - xs[None, :]), axis=1)
    at_rest = np.isclose(ball_xyz[:, 0][:, None], xs[None, :], atol=1e-6).any(axis=1)
    if not (nearest[at_rest] == ball_slot[at_rest]).all():
        raise AssertionError("tracked ball_slot disagrees with the ball's resting position")
    return {
        "cup_xyz": cup_xyz,
        "ball_xyz": ball_xyz,
        "action": np.concatenate(A),
        "phase": np.concatenate(PH),
        "ball_slot": ball_slot,
        "ball_under_target": (ball_slot == target).astype(np.uint8),
        "cup_z_max": cup_xyz[:, :n_cups, 2].max(axis=1),
        "hold_frames": hold_frames,
        "ball_start": int(ball_start),
        "swapped": n_swaps,          # number of swaps during the hold
        "target_slot": target,
        "pairs": pairs_used,
        "closed_start": bool(closed_start),
        "peek_after": -1 if peek_after is None else int(peek_after),
    }


class Memtest4Recorder:
    """Flattened zarr store: per-frame arrays (total_frames, ...) plus per-episode arrays."""

    PER_FRAME = ("action", "phase", "ball_slot", "ball_under_target", "cup_z_max")
    PER_EPISODE = ("episode_ends", "episode_seeds", "episode_ball_start", "episode_swapped",
                   "episode_target", "episode_hold", "episode_closed", "episode_peek")
    DEFAULTS = {"episode_closed": 0, "episode_peek": -1}     # stores written before these existed

    def __init__(self, path, res, append=False):
        if append:
            z = self.root = zarr.open_group(str(path), mode="a")
            if int(z.attrs["resolution"]) != int(res):
                raise ValueError(f"{path} is {z.attrs['resolution']}px, cannot append {res}px episodes")
            n_old = len(z["episode_ends"])
            self.episodes = {k: (z[k][:].tolist() if k in z else [self.DEFAULTS[k]] * n_old) for k in self.PER_EPISODE}
            self.frames = int(self.episodes["episode_ends"][-1]) if self.episodes["episode_ends"] else 0
            for key in self.PER_EPISODE:
                if key in z:
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
                           ("episode_target", ep["target_slot"]), ("episode_hold", ep["hold_frames"]),
                           ("episode_closed", int(ep.get("closed_start", False))),
                           ("episode_peek", int(ep.get("peek_after", -1)))):
            self.episodes[key].append(value)

    def finalize(self, cfg, master_seed):
        z = self.root
        dtypes = {"episode_ends": "int64", "episode_seeds": "int64", "episode_hold": "int64",
                  "episode_ball_start": "uint8", "episode_swapped": "uint8", "episode_target": "uint8",
                  "episode_closed": "uint8", "episode_peek": "int16"}
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
