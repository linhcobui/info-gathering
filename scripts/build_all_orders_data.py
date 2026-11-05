"""A training store that contains EVERY swap order of 0-3 swaps (3 + 9 + 27 + 81 = 120 sequences).

    pixi run python scripts/build_all_orders_data.py [--per_depth 96] [--holdout 24] [--out data/memtest4_all3.zarr]

Each depth gets the same number of training episodes (as in data/memtest4.zarr). Within a depth every order
appears at least once, the rest are spread evenly (depth 3: 81 orders once + 15 drawn at random). The last
`--holdout` episodes are extra random 0-3 episodes for the held-out loss, so the training part keeps every
order. The order is forced by searching episode seeds: sample_episode draws its pairs from the seeded rng, so
the stored seed replays exactly (swap_sequences.sequences checks this); the slack is drawn as in training.
Embeddings are not cached here: run cache-embeddings-prelim on the new store afterwards.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from info_gathering.memtest4 import Memtest4Recorder, sample_episode  # noqa: E402
from info_gathering.rendering import build_model, render_episode  # noqa: E402
from info_gathering.swap_sequences import all_sequences, sequences  # noqa: E402


def slot_pairs(ep, n_cups=3):
    slot_of_identity = np.arange(n_cups)
    out = []
    for a, b in ep["pairs"]:
        out.append(tuple(sorted((int(slot_of_identity[a]), int(slot_of_identity[b])))))
        slot_of_identity[[a, b]] = slot_of_identity[[b, a]]
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/data/memtest4.yaml"))
    ap.add_argument("--out", default="data/memtest4_all3.zarr")
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--per_depth", type=int, default=96)
    ap.add_argument("--holdout", type=int, default=24)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg.update(seed=args.seed, swap_counts=list(range(args.max_depth + 1)), zarr_path=args.out,
               video_dir=f"data/videos_{Path(args.out).stem}")
    rng = np.random.default_rng(args.seed)

    want = []
    for d in range(args.max_depth + 1):
        seqs = all_sequences(d)
        reps = [seqs[i % len(seqs)] for i in range(args.per_depth)] if len(seqs) <= args.per_depth else seqs
        if len(seqs) < args.per_depth and args.per_depth % len(seqs):
            extra = rng.choice(len(seqs), args.per_depth - len(seqs) * (args.per_depth // len(seqs)), replace=False)
            reps = [s for s in seqs for _ in range(args.per_depth // len(seqs))] + [seqs[i] for i in extra]
        want += reps
    want = [want[i] for i in rng.permutation(len(want))]
    hold = [all_sequences(d)[rng.integers(len(all_sequences(d)))]
            for d in range(args.max_depth + 1) for _ in range(args.holdout // (args.max_depth + 1))]
    want += [hold[i] for i in rng.permutation(len(hold))]
    cfg["n_episodes"] = len(want)

    plan = []                                                    # (seed, start, n_swaps) reproducing each order
    for start, pairs in tqdm(want, desc="searching seeds"):
        while True:
            seed = int(rng.integers(0, 2**31 - 1))
            ep = sample_episode(cfg, np.random.default_rng(seed), start, len(pairs))
            if slot_pairs(ep) == tuple(pairs):
                plan.append((seed, start, len(pairs)))
                break

    model, data, handles = build_model(cfg, str(ROOT / "submodules" / "object_sim"))
    renderer = mujoco.Renderer(model, height=cfg["resolution"], width=cfg["resolution"])
    rec = Memtest4Recorder(str(ROOT / args.out), res=cfg["resolution"], append=False)
    try:
        for seed, start, n in tqdm(plan, desc="rendering"):
            ep = sample_episode(cfg, np.random.default_rng(seed), start, n)
            rec.add_episode(render_episode(model, data, handles, ep, cfg, renderer=renderer), ep, seed)
    finally:
        renderer.close()
    rec.finalize(cfg, args.seed)

    got = sequences(ROOT / args.out)
    assert got == [tuple(w) if isinstance(w, tuple) else w for w in want], "replay does not reproduce the plan"
    train = got[:len(got) - args.holdout]
    for d in range(args.max_depth + 1):
        full = set(all_sequences(d))
        have = [s for s in train if len(s[1]) == d]
        print(f"depth {d}: {len(have)} training episodes, {len(set(have) & full)}/{len(full)} orders covered")
    print(f"-> {args.out}: {len(got)} episodes ({len(got) - args.holdout} training + {args.holdout} held out at the end)")


if __name__ == "__main__":
    main()
