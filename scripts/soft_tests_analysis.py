"""What do the soft-pair tests look like? Reads the per-test dump of scripts/soft_pairs_phase1.py --dump.

    python3 scripts/soft_tests_analysis.py figures/soft_tests/dump.npz [--out figures/soft_tests/soft_tests.png]

A test = one rest frame along one borrowed future, where both memories (x and y) are asked "lift on the next frame?";
d = the decoder's answer, mean of the two frozen models. s = min over a pair's tests of 1 - |d_x - d_y|.
Training pulls each pair with weight s and pushes it with weight 1 - s (equal at s = 0.5).
Truth (hidden ball slot; privileged belief class at still frames) is used only to score.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import FAINT, HIGHLIGHT, INK, INTERESTING, MUTED, apply  # noqa: E402

CERTAIN = (0, 1, 2)


def main():
    path = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else str(Path(path).with_name("soft_tests.png"))
    d = np.load(path)
    P, Jp, Js = d["P_all"], d["J_pair"], d["J_side"]
    D = np.nanmean(P, 0)                                            # (jobs, steps) mean over the two models
    rx = np.flatnonzero(Js == 0); ry = rx + 1
    assert (Jp[ry] == Jp[rx]).all() and (Js[ry] == 1).all()
    dx, dy, pid = D[rx], D[ry], Jp[rx]
    ok = ~np.isnan(dx) & ~np.isnan(dy)
    tx, ty = dx[ok], dy[ok]
    tp = np.repeat(pid, ok.sum(1))                                  # pair id of every test
    n_pairs = len(d["s"])
    n_test = np.bincount(tp, minlength=n_pairs)
    n_fut = np.bincount(pid, minlength=n_pairs)
    keep = (n_test > 0) & ~d["twin_bad"]
    s = d["s"]
    same = d["slot_x"] == d["slot_y"]
    shown = d["shown_x"] & d["shown_y"]
    cx, cy = d["cls_x"], d["cls_y"]
    lab = (cx >= 0) & (cy >= 0)

    print(f"{n_pairs} sampled pairs; futures per pair {np.median(n_fut):.0f}; tests per pair: median {np.median(n_test):.0f}, "
          f"10-90% {np.percentile(n_test, 10):.0f}-{np.percentile(n_test, 90):.0f}, max {n_test.max()}")
    a = np.r_[tx, ty]
    bins = [(0, .1, "< 0.1"), (.1, .4, "0.1-0.4"), (.4, .6, "0.4-0.6"), (.6, .9, "0.6-0.9"), (.9, 1.01, "> 0.9")]
    print("decoder answer d over all tests (x and y): " + ", ".join(f"{nm} {np.mean((a >= lo) & (a < hi)):.3f}" for lo, hi, nm in bins))
    both_no, both_yes = (tx < .1) & (ty < .1), (tx > .9) & (ty > .9)
    dis = np.abs(tx - ty) > .5
    print(f"per test: both say no-ball {both_no.mean():.3f}, both say ball {both_yes.mean():.3f}, "
          f"disagree by > 0.5 {dis.mean():.3f}, other {1 - both_no.mean() - both_yes.mean() - dis.mean():.3f}")
    inf = np.bincount(tp[dis], minlength=n_pairs)
    print(f"informative (disagreeing) tests per pair: 0 in {np.mean(inf == 0):.2f} of pairs")
    print(f"twin filter (models differ > {float(d['twin_tol'])} on any test) drops {np.mean(d['twin_bad']):.2f} of pairs; "
          f"dropped same-cup {np.mean(d['twin_bad'][same]):.2f}, different-cup {np.mean(d['twin_bad'][~same]):.2f}")
    print(f"\nkept pairs: {keep.sum()} | truly same cup {np.mean(same[keep]):.2f} (shown-start pairs: {np.mean(same[keep & shown]):.2f})")

    cats = [("same cup (shown starts)", keep & shown & same), ("different cup (shown starts)", keep & shown & ~same),
            ("same belief, both certain", keep & lab & (cx == cy) & np.isin(cx, CERTAIN)),
            ("same belief, both uncertain", keep & lab & (cx == cy) & ~np.isin(cx, CERTAIN)),
            ("different belief", keep & lab & (cx != cy))]
    print(f"\n{'truth':30s} {'n':>6s} {'mean s':>7s} {'s>0.5':>6s} {'pull share':>11s} {'inf. tests':>11s}")
    rows = []
    for nm, m in cats:
        if not m.any():
            continue
        pull = s[m].sum(); push = (1 - s[m]).sum()
        rows.append((nm, m.sum(), s[m], pull / (pull + push)))
        print(f"{nm:30s} {m.sum():6d} {s[m].mean():7.3f} {np.mean(s[m] > .5):6.2f} {pull / (pull + push):11.2f} "
              f"{np.median(inf[m]):11.0f}")
    tot_pull_diff = s[keep & ~same].sum() / s[keep].sum()
    print(f"\nshare of ALL pull weight that lands on different-cup pairs: {tot_pull_diff:.2f}; "
          f"share of ALL push weight on same-cup pairs: {(1 - s[keep & same]).sum() / (1 - s[keep]).sum():.2f}")

    apply(font_size=9)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].hist(a, bins=50, range=(0, 1), color=MUTED)
    ax[0].set_yscale("log"); ax[0].set_xlabel("decoder answer d at a test"); ax[0].set_ylabel("tests (log)")
    ax[0].set_title(f"{len(tx)} tests, {np.median(n_test):.0f} per pair (median)", loc="left", fontsize=9.5)
    b = np.linspace(0, 1, 26)
    ax[1].hist(s[keep & shown & same], bins=b, color=HIGHLIGHT, alpha=0.8, label="truly same cup")
    ax[1].hist(s[keep & shown & ~same], bins=b, color=INTERESTING[0], alpha=0.7, label="truly different cup")
    ax[1].axvline(0.5, color=INK, lw=1, ls="--"); ax[1].text(0.51, ax[1].get_ylim()[1] * 0.9, "pull = push", fontsize=8)
    ax[1].set_xlabel("weight s (pull s, push 1 - s)"); ax[1].set_ylabel("pairs (shown starts)"); ax[1].legend(frameon=False)
    ax[1].set_title("s by truth", loc="left", fontsize=9.5)
    names = [r[0] for r in rows]; share = [r[3] for r in rows]
    yy = np.arange(len(rows))[::-1]
    ax[2].barh(yy, share, color=HIGHLIGHT, label="pull (s)")
    ax[2].barh(yy, [1 - v for v in share], left=share, color=FAINT, label="push (1 - s)")
    for y_, v, r in zip(yy, share, rows):
        ax[2].text(0.01, y_, f"{v:.2f}  (n={r[1]})", va="center", fontsize=8, color=INK)
    ax[2].set_yticks(yy); ax[2].set_yticklabels(names, fontsize=8); ax[2].set_xlim(0, 1)
    ax[2].set_xlabel("share of the pair's total weight"); ax[2].legend(frameon=False, loc="lower right", fontsize=8)
    ax[2].set_title("how much is pull vs push, by truth", loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
