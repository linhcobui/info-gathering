"""ALIGN-SELF-PREDICTIONS labels: one class per still frame from the model's own predictions, or none.

    pixi run python scripts/build_psr_align_labels.py [--src <psr_labels npz>] [--out <npz>]

From scripts/psr_labels_phase1.py (frozen WINDOW60 teacher memory + latent-action prior), the two-action rule:

    0  "ball here"            the decoder says ball, no imagination
    1  "after the 0<->1 action"  it says no; pressing ONLY that swap action brings the ball out
    2  "after the 0<->2 action"  it says no; pressing ONLY that one brings it out
   -1  no class               neither, or both (conflict)

Each frame is in at most one class. `truth` is saved for diagnostics only; training reads `label`.
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/latent_action_prior/psr_labels_train.npz")
    ap.add_argument("--a1", default=None, help="the action that swaps slot 0 with one other slot (default: the "
                    "source's label-free pair_codes, else p_after_u9)")
    ap.add_argument("--a2", default=None, help="the action that swaps slot 0 with the remaining slot")
    ap.add_argument("--out", default="data/labels_psr_self_w60_memtest4.npz")
    args = ap.parse_args()
    d = np.load(ROOT / args.src)
    if args.a1 is None:
        c1, c2 = (int(x) for x in d["pair_codes"]) if "pair_codes" in d else (9, 8)
        args.a1, args.a2 = f"p_after_u{c1}", f"p_after_u{c2}"
        print(f"label actions: {args.a1}, {args.a2}")
    here, a, b = d["p_here"] > 0.5, d[args.a1] > 0.5, d[args.a2] > 0.5
    label = np.full(len(here), -1)
    label[here] = 0
    label[~here & a & ~b] = 1
    label[~here & b & ~a] = 2
    np.savez(ROOT / args.out, episode=d["episode"], t=d["t"], label=label, truth=d["truth"], swaps_done=d["swaps_done"],
             note=np.array("label: 0 here, 1 after 0<->1 action, 2 after 0<->2 action, -1 none; truth for diagnostics only"))
    has = label >= 0
    print(f"{len(label)} still frames from {len(set(d['episode'].tolist()))} episodes -> {args.out}")
    for k, nm in ((0, "here"), (1, f"after {args.a1[8:]}"), (2, f"after {args.a2[8:]}"), (-1, "no class")):
        m = label == k
        print(f"  {nm:12s} {m.sum():5d} ({m.mean():.3f})  true slot 0/1/2 (diagnostics): "
              + " / ".join(f"{np.mean(d['truth'][m] == j):.2f}" for j in range(3)))
    # a class is right when it holds its MAJORITY slot (class ids are arbitrary: which slot an action reveals
    # is not known without labels)
    right = sum(int(np.max(np.bincount(d["truth"][label == k], minlength=3))) for k in (0, 1, 2) if (label == k).any())
    print(f"  classed frames in their class's majority slot (diagnostics): {right / max(has.sum(), 1):.3f}")


if __name__ == "__main__":
    main()
