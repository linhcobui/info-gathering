"""The exact swap sequence of every memtest4 episode: (ball start slot, slot pair of each swap).

Two independent readings, which must agree:

    replay   re-run memtest4.sample_episode with the stored episode seed and convert the cup
             IDENTITY pairs it drew into the SLOT pairs they occupied. Exact, and checked against
             the stored ball_slot and hold length.
    labels   the ball's slot after each 6-frame swap block (the dense halfway relabel); a swap
             that leaves the ball alone must be the one pair not containing its slot. This is the
             reading every earlier seen/unseen split in this repo used.

"Seen" anywhere in the TWIN_FINETUNE experiment means: the same (start slot, slot-pair sequence)
occurs in the named training store. Slack (hold length) is ignored, as it always was.
"""
import json

import numpy as np
import zarr

from .dataset_smt_prelim import dense_ball_slot
from .memtest4 import pairs_for, sample_episode

SLOT_PAIRS = ((0, 1), (0, 2), (1, 2))
HOLD_PHASE = 1


def _episode_bounds(z):
    ends = z["episode_ends"][:]
    return np.concatenate([[0], ends[:-1]]), ends


def replay_sequences(zarr_path):
    """Sequences by re-running the generator from each episode's seed. Raises on any mismatch."""
    z = zarr.open_group(str(zarr_path), mode="r")
    starts, ends = _episode_bounds(z)
    seeds, ball_start = z["episode_seeds"][:], z["episode_ball_start"][:]
    swapped, hold = z["episode_swapped"][:], z["episode_hold"][:]
    ball_slot = z["ball_slot"][:]
    # closed start / mid-hold peek (memtest4.sample_episode); absent in stores written before them
    closed = z["episode_closed"][:] if "episode_closed" in z else np.zeros(len(ends), np.uint8)
    peek = z["episode_peek"][:] if "episode_peek" in z else np.full(len(ends), -1)
    # one store can hold several generator runs (--append); each episode's config is the one its
    # run used. Every store in this experiment is a single run or a subset whose episodes share
    # the generation protocol, which is checked per episode below.
    cfgs = [json.loads(c) for c in z.attrs.get("configs", [z.attrs["config"]])]
    out = []
    for e in range(len(ends)):
        stored = ball_slot[starts[e]:ends[e]]
        ep = None
        for cfg in cfgs:
            # training stores draw the slack first (hold_frames None); sweep stores pass it in
            for hold_arg in ((None, int(hold[e])) if not (cfg.get("slack_range") or cfg.get("slack_values"))
                             else (int(hold[e]), None)):
                cand = sample_episode(cfg, np.random.default_rng(int(seeds[e])), int(ball_start[e]),
                                      int(swapped[e]), hold_frames=hold_arg, closed_start=bool(closed[e]),
                                      peek_after=None if int(peek[e]) < 0 else int(peek[e]))
                # compare where the ball is AT REST: old stores (data/memtest4.zarr) labelled the
                # slot off the ball's x mid-swap, so only resting frames are comparable across versions
                xs = np.array([-1.0, 0.0, 1.0]) * float(cfg["cup_spacing"])
                rest = np.isclose(cand["ball_xyz"][:, :1], xs[None], atol=1e-6).any(1) \
                    if len(cand["ball_xyz"]) == len(stored) else None
                if (cand["hold_frames"] == int(hold[e]) and rest is not None
                        and (cand["ball_slot"][rest] == stored[rest]).all()):
                    ep, used = cand, cfg
                    break
            if ep is not None:
                break
        if ep is None:
            raise AssertionError(f"{zarr_path} episode {e}: no generator replay reproduces its ball_slot")
        slot_of_identity = np.arange(int(used.get("n_cups", 3)))
        pairs = []
        for a, b in ep["pairs"]:
            pairs.append(tuple(sorted((int(slot_of_identity[a]), int(slot_of_identity[b])))))
            slot_of_identity[[a, b]] = slot_of_identity[[b, a]]
        out.append((int(ball_start[e]), tuple(pairs)))
    return out


def label_sequences(zarr_path):
    """Sequences read off the ball-slot labels alone (no replay)."""
    z = zarr.open_group(str(zarr_path), mode="r")
    starts, ends = _episode_bounds(z)
    lengths = ends - starts
    bs, ph, sw = z["ball_slot"][:], z["phase"][:], z["episode_swapped"][:]
    peek = z["episode_peek"][:] if "episode_peek" in z else np.full(len(ends), -1)
    cfg = json.loads(z.attrs["config"])
    peek_len = 2 * int(cfg["move_frames"]) + int(cfg.get("peek_up_frames", 2))
    padded = np.zeros((len(ends), int(lengths.max())), dtype=np.int64)
    for e in range(len(ends)):
        padded[e, :lengths[e]] = bs[starts[e]:ends[e]]
    dense = dense_ball_slot(padded, lengths)
    out = []
    for e in range(len(ends)):
        h0 = int(np.flatnonzero(ph[starts[e]:ends[e]] == HOLD_PHASE)[0])
        # a peek after k swaps delays every later swap by peek_len frames
        shift = lambda j: peek_len if 0 <= int(peek[e]) <= j else 0
        traj = [int(dense[e, h0])] + [int(dense[e, h0 + 4 + 6 * (j + 1) - 1 + shift(j)]) for j in range(int(sw[e]))]
        out.append((traj[0], tuple(tuple(sorted((a, b))) if a != b else tuple(sorted({0, 1, 2} - {a}))
                                   for a, b in zip(traj, traj[1:]))))
    return out


def sequences(zarr_path):
    """Replay, cross-checked against the labels. The single entry point the experiment uses."""
    replay = replay_sequences(zarr_path)
    try:
        labels = label_sequences(zarr_path)
    except ValueError as err:      # old x-based labels: the dense relabel refuses them
        print(f"[swap_sequences] {zarr_path}: label cross-check unavailable ({err}); replay only")
        return replay
    bad = [e for e, (r, l) in enumerate(zip(replay, labels)) if r != l]
    if bad:
        raise AssertionError(f"{zarr_path}: replay and label readings disagree on episodes {bad[:10]}")
    return replay


def all_sequences(depth, n_slots=3):
    """Every possible (start, pairs) at this depth: 3 x 3^depth."""
    from itertools import product
    return [(s, p) for s in range(n_slots) for p in product(SLOT_PAIRS, repeat=depth)]


def final_slot(seq):
    """The ball's slot after the sequence -- whether the lifted cup (slot 0) has it."""
    slot = seq[0]
    for a, b in seq[1]:
        slot = b if slot == a else a if slot == b else slot
    return slot


assert pairs_for(3) == [(0, 1), (0, 2), (1, 2)] == list(SLOT_PAIRS)
