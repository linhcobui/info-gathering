"""OGBar-SMT teacher belief at one model step: PC1 against the true c_view (system python3: needs sklearn).

    python3 scripts/plot_ogbar_belief_cview.py [--npz figures/ogbar_smt_belief_start_mid_s25000.npz] [--step 100]

Belief z*_t = teacher memory over the window of frames ending at model step t (scripts/belief_midframe_ogbar.py).
PCA (20 components) fitted on the seen episodes' beliefs at that step, each number standardised by its seen mean/std.
c_view: COM offset (cm), + = heavy end on the image's right (privileged; display only).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import FAINT, HIGHLIGHT, INK, INTERESTING, MUTED, apply  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="figures/ogbar_smt_belief_start_mid_s25000.npz")
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = np.load(ROOT / args.npz, allow_pickle=True)
    t = args.step
    Z, cv, pr = d[f"Z_{t}"], d[f"cview_{t}"], d[f"probed_{t}"].astype(bool)
    un = d["unseen"].astype(bool); se = ~un
    mu, sd = Z[se].mean(0), Z[se].std(0) + 1e-6
    P = PCA(20, random_state=0).fit((Z[se] - mu) / sd).transform((Z - mu) / sd)
    pc1 = P[:, 0] * np.sign(np.corrcoef(P[se, 0], cv[se])[0, 1])          # orient PC1 so + follows c_view
    r_se, r_un = np.corrcoef(pc1[se], cv[se])[0, 1], np.corrcoef(pc1[un], cv[un])[0, 1]
    k, b = np.polyfit(cv[se], pc1[se], 1)

    apply(font_size=10)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(cv[se & ~pr], pc1[se & ~pr], s=14, marker="^", color=FAINT, label="seen (training), not yet probed")
    ax.scatter(cv[se & pr], pc1[se & pr], s=14, marker="o", color=MUTED, label="seen (training), probed before this step")
    ax.scatter(cv[un], pc1[un], s=150, marker="*", color=HIGHLIGHT, edgecolor=INK, linewidth=0.8, zorder=4,
               label="unseen eval episode")
    xs = np.linspace(cv.min(), cv.max(), 2)
    ax.plot(xs, k * xs + b, ls="--", color=INK, lw=1, label="line fitted on seen episodes")
    ax.set_xlabel("true c_view: COM offset (cm), + = heavy end on the image's right")
    ax.set_ylabel(f"belief PC1 at model step {t}")
    ax.set_title(f"OGBar-SMT teacher belief at model step {t} (checkpoint step {int(d['step'])})\n"
                 f"r with c_view: seen {r_se:+.2f} (n={se.sum()}), unseen {r_un:+.2f} (n={un.sum()})", fontsize=10, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.yaxis.grid(True)
    fig.tight_layout()
    out = ROOT / (args.out or f"figures/ogbar_smt/belief_cview_step{t}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"-> {out}  r seen {r_se:+.3f}, unseen {r_un:+.3f}")


if __name__ == "__main__":
    main()
