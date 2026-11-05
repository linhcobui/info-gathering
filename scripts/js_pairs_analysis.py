"""How different are the two memories of a pair, in bits? Jensen-Shannon divergence per test, from a test dump.

    python3 scripts/js_pairs_analysis.py figures/js_pairs/dump.npz [--out figures/js_pairs/js_pairs.png]

Dump: scripts/soft_pairs_phase1.py --dump (one model, K borrowed futures). A test = one rest frame along one future,
where both memories are asked "lift on the next frame?"; each answer p is a coin (ball with probability p).
JS(p_x, p_y) = H((p_x + p_y)/2) - (H(p_x) + H(p_y))/2 in bits, H = binary entropy: 0 when the two coins are
identical, 1 bit for 0 vs 1, ~0.03 bits for 0.5 vs 0.7. Per pair: max JS over its tests (the most decisive test)
and the sum (total evidence of a difference). Truth (hidden slot, privileged belief class) is used only to score.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import HIGHLIGHT, INTERESTING, MUTED, apply  # noqa: E402

CERTAIN = (0, 1, 2)


def H(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main():
    path = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else str(Path(path).with_name("js_pairs.png"))
    d = np.load(path)
    D = np.nanmean(d["P_all"], 0)
    Js, Jp = d["J_side"], d["J_pair"]
    rx = np.flatnonzero(Js == 0); ry = rx + 1
    dx, dy, pid = D[rx], D[ry], Jp[rx]
    ok = ~np.isnan(dx) & ~np.isnan(dy)
    tx, ty, tp = dx[ok], dy[ok], np.repeat(pid, ok.sum(1))
    js = H((tx + ty) / 2) - (H(tx) + H(ty)) / 2
    n = len(d["slot_x"])
    js_max = np.full(n, np.nan); js_sum = np.zeros(n); n_test = np.bincount(tp, minlength=n)
    np.maximum.at(js_max := np.zeros(n), tp, js)
    np.add.at(js_sum, tp, js)
    has = n_test > 0
    same = d["slot_x"] == d["slot_y"]; shown = d["shown_x"] & d["shown_y"]
    cx, cy = d["cls_x"], d["cls_y"]; lab = (cx >= 0) & (cy >= 0)
    print(f"{n} pairs, one model, {np.median(np.bincount(pid, minlength=n)):.0f} futures per pair, tests per pair median "
          f"{np.median(n_test[has]):.0f}; per-test JS: 0 bits {np.mean(js < 0.01):.2f} of tests, > 0.5 bits {np.mean(js > .5):.2f}")
    cats = [("same cup (both shown starts)", has & shown & same), ("different cup (both shown starts)", has & shown & ~same),
            ("same belief, both certain", has & lab & (cx == cy) & np.isin(cx, CERTAIN)),
            ("same belief, both uncertain", has & lab & (cx == cy) & ~np.isin(cx, CERTAIN)),
            ("different belief, certain vs certain", has & lab & (cx != cy) & np.isin(cx, CERTAIN) & np.isin(cy, CERTAIN)),
            ("different belief, certain vs uncertain", has & lab & (cx != cy) & (np.isin(cx, CERTAIN) != np.isin(cy, CERTAIN))),
            ("different belief, both uncertain", has & lab & (cx != cy) & ~np.isin(cx, CERTAIN) & ~np.isin(cy, CERTAIN))]
    print(f"\n{'truth':40s} {'n':>5s} | {'max JS (bits): median':>22s} {'< 0.05':>7s} {'> 0.5':>6s} | {'sum JS: median':>15s}")
    for nm, m in cats:
        if m.any():
            print(f"{nm:40s} {m.sum():5d} | {np.median(js_max[m]):22.3f} {np.mean(js_max[m] < .05):7.2f} {np.mean(js_max[m] > .5):6.2f} | "
                  f"{np.median(js_sum[m]):15.2f}")
    m_s, m_d = has & shown & same, has & shown & ~same
    for thr in (0.05, 0.1, 0.2, 0.5):
        print(f"  rule 'same if max JS < {thr}': same-cup pairs called same {np.mean(js_max[m_s] < thr):.2f}, "
              f"different-cup pairs called same {np.mean(js_max[m_d] < thr):.2f}")

    apply(font_size=9)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    b = np.linspace(0, 1, 41)
    ax[0].hist(js_max[m_s], bins=b, color=HIGHLIGHT, alpha=0.8, label="truly same cup")
    ax[0].hist(js_max[m_d], bins=b, color=INTERESTING[0], alpha=0.7, label="truly different cup")
    ax[0].set_xlabel("max JS over the pair's tests (bits)"); ax[0].set_ylabel("pairs (both shown starts)")
    ax[0].legend(frameon=False); ax[0].set_title("the most decisive test", loc="left", fontsize=10)
    bins2 = np.r_[0, np.logspace(-3, np.log10(max(js_sum[has].max(), 1)), 40)]
    for nm, m, c in (("same belief, both uncertain", cats[3][1], INTERESTING[2]), ("same cup", m_s, HIGHLIGHT),
                     ("different cup", m_d, INTERESTING[0])):
        ax[1].hist(js_sum[m] + 1e-3, bins=bins2, histtype="step", lw=2, color=c, label=nm)
    ax[1].set_xscale("log"); ax[1].set_xlabel("sum of JS over the pair's tests (bits, + 0.001)")
    ax[1].set_ylabel("pairs"); ax[1].legend(frameon=False); ax[1].set_title("total evidence of a difference", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=170)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
