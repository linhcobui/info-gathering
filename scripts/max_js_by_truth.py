"""m = max JS over a pair's real-lift tests, split by what the pair truly is (truth used ONLY for this plot).

    python3 scripts/max_js_by_truth.py [data/pairs_jsw_all3u.npz] [--out figures/js_pairs/max_js_by_truth.png]

Pairs: soft_pairs_phase1.py --rule jsweight --tests lifts (frozen Naive-U judge, K = 64 borrowed futures).
  same cup         both shown-start episodes, ball in the same cup
  different cup    both shown-start episodes, ball in different cups
  same uncertain belief   both frames carry the same privileged belief class and it is uncertain:
                   {0,a,b} nothing known (honest p 1/3), {0,a}/{0,b} one of two incl. the lifted cup (1/2),
                   {a,b} one of the two others (0). Belief labels exist at still hold frames only.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import FAINT, HIGHLIGHT, INTERESTING, MUTED, apply  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "data/pairs_jsw_all3u.npz"
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figures/js_pairs/max_js_by_truth.png"
    d = np.load(ROOT / path)
    m, same, shown = d["max_js"], d["same_slot"], d["shown"]
    pr = np.load(ROOT / "data/labels_belief_all3u_privileged.npz")
    cl = {(int(e), int(t)): int(c) for e, t, c in zip(pr["episode"], pr["t"], pr["truth"])}
    cx = np.array([cl.get((int(e), int(t)), -1) for e, t in zip(d["x_episode"], d["x_t"])])
    cy = np.array([cl.get((int(e), int(t)), -1) for e, t in zip(d["y_episode"], d["y_t"])])
    unc = (cx == cy) & (cx >= 3)

    groups = [("same cup (shown starts)", shown & same, [("", np.ones_like(m, bool), HIGHLIGHT)]),
              ("different cup (shown starts)", shown & ~same, [("", np.ones_like(m, bool), INTERESTING[0])]),
              ("same uncertain belief (closed starts)", unc,
               [("{0,a,b}: nothing known, p* 1/3", cx == 3, INTERESTING[2]),
                ("{0,a} / {0,b}: one of two, p* 1/2", np.isin(cx, (5, 6)), INTERESTING[3]),
                ("{a,b}: not the lifted cup, p* 0", cx == 4, MUTED)])]
    print(f"{len(m)} pairs; m = max JS (bits) over each pair's real-lift tests")
    print(f"{'group':52s} {'n':>6s} {'median m':>9s} {'m < 0.05':>9s} {'m > 0.5':>8s} {'mean w=1-m':>11s}")

    apply(font_size=9)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    bins = np.linspace(0, 1, 41)
    for a, (title, g, subs) in zip(ax, groups):
        bottom = np.zeros(len(bins) - 1)
        for nm, sm, c in subs:
            sel = g & sm
            h = np.histogram(m[sel], bins)[0] / max(g.sum(), 1)
            a.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge", color=c, label=nm or None, lw=0)
            bottom += h
            print(f"{title + (' / ' + nm if nm else ''):52s} {sel.sum():6d} {np.median(m[sel]):9.3f} "
                  f"{np.mean(m[sel] < .05):9.2f} {np.mean(m[sel] > .5):8.2f} {np.mean(1 - m[sel]):11.3f}")
        a.set_title(f"{title}\nn {g.sum()}, median m {np.median(m[g]):.2f}", loc="left", fontsize=10)
        a.set_yscale("log"); a.set_ylim(1e-3, 1.2)
        a.set_xlabel("m = max JS over the pair's tests (bits)")
        for v in (0.05, 0.5):
            a.axvline(v, color=FAINT, lw=0.8, ls="--")
        if len(subs) > 1:
            a.legend(frameon=False, fontsize=7.5, loc="upper center")
    ax[0].set_ylabel("share of the group's pairs (log)")
    fig.suptitle("How different the decoder says two matched-window memories are, by what they truly are "
                 "(30,000 pairs, Naive-U judge, 64 borrowed futures)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    (ROOT / out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / out, dpi=170); fig.savefig((ROOT / out).with_suffix(".pdf"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
