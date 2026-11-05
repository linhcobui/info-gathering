"""ALL3-U: memtest4 stores where the start slot is sometimes NEVER SHOWN, so the honest belief is uncertain.

    pixi run python scripts/build_uncertain_data.py train   # data/memtest4_all3u.zarr
    pixi run python scripts/build_uncertain_data.py eval    # data/memtest4_evalu.zarr

train: the 384 ALL3 training episodes copied unchanged (start shown), then 384 CLOSED-START episodes (cups down
       over the ball from frame 0): 96 per depth 0-3 with every swap order present, half of them with one PEEK (the
       cup at the lifted slot rises and comes back down after k swaps, k uniform on 0..n), then 24 held out at the
       end (8 shown, 8 closed, 8 closed + peek, random 0-3 orders). Training reads the first 768.
eval:  closed start only, 0-8 swaps, 45 per depth: 15 without a peek (honest answer at the lift: 1/3) and 30 with
       one (ball seen -> certain; cup seen empty -> the other two, then the swaps). Slack 0-20 as in the eval sets.

Closed start and peek change pixels and add frames; they never draw from the episode rng, so every episode replays
from its seed (swap_sequences.sequences checks this on the finished store). Embeddings are not cached here.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
import zarr  # noqa: E402
from tqdm import tqdm  # noqa: E402

from build_all_orders_data import slot_pairs  # noqa: E402
from info_gathering.memtest4 import Memtest4Recorder, hold_base, sample_episode  # noqa: E402
from info_gathering.rendering import build_model, render_episode  # noqa: E402
from info_gathering.swap_sequences import all_sequences, sequences  # noqa: E402

ALL3 = "data/memtest4_all3.zarr"
N_ALL3_TRAIN = 384


def search_seed(cfg, rng, start, pairs, **kw):
    """A seed whose episode draws exactly this slot-pair order."""
    while True:
        seed = int(rng.integers(0, 2**31 - 1))
        if slot_pairs(sample_episode(cfg, np.random.default_rng(seed), start, len(pairs), **kw)) == tuple(pairs):
            return seed


def render_all(cfg, plan, rec):
    model, data, handles = build_model(cfg, str(ROOT / "submodules" / "object_sim"))
    renderer = mujoco.Renderer(model, height=cfg["resolution"], width=cfg["resolution"])
    try:
        for seed, start, n, kw in tqdm(plan, desc="rendering"):
            ep = sample_episode(cfg, np.random.default_rng(seed), start, n, **kw)
            rec.add_episode(render_episode(model, data, handles, ep, cfg, renderer=renderer), ep, seed)
    finally:
        renderer.close()


def train(args):
    src = zarr.open_group(str(ROOT / ALL3), mode="r")
    cfg_all3 = json.loads(src.attrs["config"])
    cfg = dict(cfg_all3, seed=args.seed, zarr_path=args.out, video_dir=f"data/videos_{Path(args.out).stem}",
               peek_up_frames=2)
    rng = np.random.default_rng(args.seed)

    # closed start: every order of each depth, 96 per depth, half with a peek after k ~ U{0..n} swaps
    closed = []
    for d in range(4):
        seqs = all_sequences(d)
        reps = [seqs[i % len(seqs)] for i in range(96)] if len(seqs) <= 96 else seqs
        if len(seqs) < 96 and 96 % len(seqs):
            extra = rng.choice(len(seqs), 96 - len(seqs) * (96 // len(seqs)), replace=False)
            reps = [s for s in seqs for _ in range(96 // len(seqs))] + [seqs[i] for i in extra]
        peeks = rng.permutation(np.r_[np.ones(48, bool), np.zeros(48, bool)])
        closed += [(s, int(rng.integers(0, d + 1)) if p else None) for s, p in zip(reps, peeks)]
    closed = [closed[i] for i in rng.permutation(len(closed))]
    held = []
    for kind in ("shown", "closed", "peek"):
        for _ in range(8):
            d = int(rng.integers(0, 4))
            s = all_sequences(d)[rng.integers(len(all_sequences(d)))]
            held.append((s, kind, int(rng.integers(0, d + 1)) if kind == "peek" else None))
    held = [held[i] for i in rng.permutation(len(held))]

    plan = []
    for (start, pairs), k in tqdm(closed, desc="seeds, closed"):
        kw = dict(closed_start=True, peek_after=k)
        plan.append((search_seed(cfg, rng, start, pairs, **kw), start, len(pairs), kw))
    for (start, pairs), kind, k in held:
        kw = dict(closed_start=kind != "shown", peek_after=k)
        plan.append((search_seed(cfg, rng, start, pairs, **kw), start, len(pairs), kw))

    rec = Memtest4Recorder(str(ROOT / args.out), res=cfg["resolution"], append=False)
    st = np.r_[0, src["episode_ends"][:]]
    for e in tqdm(range(N_ALL3_TRAIN), desc="copying ALL3"):          # unchanged frames, start shown
        a, b = int(st[e]), int(st[e + 1])
        ep = {k: src[k][a:b] for k in Memtest4Recorder.PER_FRAME}
        ep.update(ball_start=int(src["episode_ball_start"][e]), swapped=int(src["episode_swapped"][e]),
                  target_slot=int(src["episode_target"][e]), hold_frames=int(src["episode_hold"][e]))
        rec.add_episode(src["rgb"][a:b], ep, int(src["episode_seeds"][e]))
    render_all(cfg, plan, rec)
    rec.finalize(cfg, args.seed)
    z = zarr.open_group(str(ROOT / args.out), mode="a")
    z.attrs["configs"] = [json.dumps(cfg_all3), json.dumps(cfg)]
    z.attrs["n_train"] = N_ALL3_TRAIN + len(closed)
    report(args.out)


def evalset(args):
    cfg = yaml.safe_load(open(ROOT / "configs/data/memtest4_eval.yaml"))
    cfg.update(seed=args.seed, zarr_path=args.out, video_dir=f"data/videos_{Path(args.out).stem}",
               peek_up_frames=2, swap_counts=list(range(9)), slack_range=[0, 20])
    rng = np.random.default_rng(args.seed)
    plan = []
    for d in range(9):
        for i in range(45):
            start, k = i % 3, (None if i < 15 else int(rng.integers(0, d + 1)))
            hold = hold_base(cfg, d) + int(rng.integers(0, 21))
            kw = dict(closed_start=True, peek_after=k, hold_frames=hold)
            plan.append((int(rng.integers(0, 2**31 - 1)), start, d, kw))
    cfg["n_episodes"] = len(plan)
    rec = Memtest4Recorder(str(ROOT / args.out), res=cfg["resolution"], append=False)
    render_all(cfg, plan, rec)
    rec.finalize(cfg, args.seed)
    report(args.out)


def report(out):
    z = zarr.open_group(str(ROOT / out), mode="r")
    seqs = sequences(ROOT / out)                       # replays every episode from its seed; raises on mismatch
    closed, peek, sw = z["episode_closed"][:], z["episode_peek"][:], z["episode_swapped"][:]
    ends = z["episode_ends"][:]
    lengths = np.diff(np.r_[0, ends])
    print(f"-> {out}: {len(seqs)} episodes replay exactly; length {lengths.min()}-{lengths.max()}")
    for d in sorted(set(sw.tolist())):
        m = sw == d
        print(f"  depth {d}: shown {int((m & (closed == 0)).sum())}, closed {int((m & (closed == 1) & (peek < 0)).sum())}, "
              f"closed + peek {int((m & (peek >= 0)).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["train", "eval"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.which == "train":
        args.out, args.seed = args.out or "data/memtest4_all3u.zarr", 31 if args.seed is None else args.seed
        train(args)
    else:
        args.out, args.seed = args.out or "data/memtest4_evalu.zarr", 32 if args.seed is None else args.seed
        evalset(args)


if __name__ == "__main__":
    main()
