"""F1 vs number of swaps: Naive SMT, DreamerV3, Belief (Align-Walk) without / with DMT, Align-privileged.

    pixi run python scripts/plot_f1_by_swaps.py [--out figures/f1_by_swaps/f1_by_swaps.png]

F1 of "ball under the lifted cup" at the final lift, shown-start test sets (63 episodes per swap count).
SMT runs: posterior memory chained from frame 0, decoder read-out at lift offset 1 (onset probes, ALL3-U data,
tables smt_phase1_all3u/depth_table_alignwalk_dmt.md). DreamerV3: imagined pixels, trained on ALL3 (the shown-start
half of ALL3-U), logdir/world_model/dreamerv3_memtest4_all3/main/depth_table.md.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from figure_style import FAINT, HIGHLIGHT, INK, INTERESTING, MUTED, apply  # noqa: E402

SWAPS = list(range(9))
RUNS = [  # name, F1 at 0..8 swaps, colour, line style, marker
    ("Belief (w DMT)", [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00], HIGHLIGHT, "-", "o"),
    ("Belief (w/o DMT)", [1.00, 1.00, 1.00, 0.96, 0.90, 0.84, 0.67, 0.79, 0.60], INTERESTING[0], "-", "o"),
    ("Privileged", [1.00, 1.00, 1.00, 0.98, 0.94, 0.93, 0.86, 0.88, 0.87], INK, "--", "s"),
    ("DreamerV3*", [1.00, 0.95, 0.93, 1.00, 0.89, 0.67, 0.33, 0.12, 0.10], INTERESTING[1], ":", "D"),
    ("Naive SMT", [0.80, 0.94, 0.85, 0.62, 0.49, 0.27, 0.09, 0.14, 0.10], MUTED, "-", "o"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/f1_by_swaps/f1_by_swaps.png")
    args = ap.parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    apply(font_size=10)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axvspan(-0.4, 3.5, color=FAINT, alpha=0.25, lw=0)
    ax.text(1.55, 0.03, "training swaps (0–3)", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.text(6.0, 0.03, "unseen numbers of swaps", ha="center", va="bottom", fontsize=9, color=MUTED)
    for name, f1, c, ls, mk in RUNS:
        ax.plot(SWAPS, f1, ls=ls, color=c, lw=2, marker=mk, ms=6 if mk == "o" else 5, label=name,
                zorder=3 if c == HIGHLIGHT else 2)
    ax.set_xlim(-0.4, 8.4); ax.set_ylim(0, 1.05)
    ax.set_xticks(SWAPS)
    ax.set_xlabel("number of swaps in the test episode")
    ax.set_ylabel("F1 score")
    ax.yaxis.grid(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.08), frameon=False, fontsize=9)
    fig.text(0.01, 0.005, "F1 of \"ball under the lifted cup\" at the final lift, 63 shown-start test episodes per "
             "swap count. *DreamerV3: imagined pixels, trained on ALL3 (the shown-start half of the data the SMT runs "
             "used).", fontsize=7, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
