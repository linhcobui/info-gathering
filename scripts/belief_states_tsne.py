"""t-SNE of chained-posterior states from scripts/belief_states_extract.py. Run with system python3 (scikit-learn).

    python3 scripts/belief_states_tsne.py <npz> <out.png>

Columns = runs. Row 1: colour = where the ball really is (left / middle / right). Row 2: colour = the honest belief
(known slot in the slot's colour; "any of the three" dark grey; two possible cups as grey triangles).
Dots = start shown, crosses = start closed. Neighbour scores: share of each state's 10 nearest states from OTHER
episodes (50-D PCA space) that agree.
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

SURF, INK, INK2, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]                 # left, middle, right
ANY3, TWO = "#3d3c39", "#9a9893"
SLOT_NAME = ["left", "middle", "right"]


def agree(X, lab, ep, mask, k=10):
    idx = NearestNeighbors(n_neighbors=min(200, len(X))).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    out = []
    for i in np.flatnonzero(mask):
        nb = [j for j in idx[i] if ep[j] != ep[i]][:k]
        out.append(np.mean(lab[nb] == lab[i]))
    return float(np.mean(out)) if out else float("nan")


def main():
    d = np.load(sys.argv[1], allow_pickle=True)
    out = sys.argv[2]
    names = list(d["names"])
    closed, slot, belief = d["closed"].astype(bool), d["slot"], d["belief"]
    ep = d["episode"] + 100000 * closed                     # episode ids unique across the two stores
    known = belief <= 2                                      # {0}, {a}, {b}
    fig, axes = plt.subplots(2, len(names), figsize=(7.2 * len(names), 13.2), squeeze=False)
    fig.patch.set_facecolor(SURF)
    for j, nm in enumerate(names):
        X = PCA(50, random_state=0).fit_transform(d["z"][j])
        Y = TSNE(2, perplexity=40, init="pca", random_state=0).fit_transform(X)
        s_shown = agree(X, slot, ep, ~closed)
        s_known = agree(X, slot, ep, closed & known)
        s_unk = agree(X, slot, ep, closed & ~known)
        b_closed = agree(X, belief, ep, closed)
        for r in range(2):
            ax = axes[r, j]
            ax.set_facecolor(SURF)
            for sp in ax.spines.values():
                sp.set_color(EDGE)
            ax.set_xticks([]); ax.set_yticks([])
            order = np.r_[np.flatnonzero(~closed), np.flatnonzero(closed)]
            if r == 0:
                col = np.array(SLOT)[slot]
                ax.scatter(Y[~closed, 0], Y[~closed, 1], s=7, c=col[~closed], linewidths=0, alpha=.85, zorder=3)
                ax.scatter(Y[closed, 0], Y[closed, 1], s=11, c=col[closed], marker="x", linewidths=.7, alpha=.75, zorder=2)
                txt = (f"same ball position among neighbours: start shown {s_shown:.2f} · closed, slot known {s_known:.2f}"
                       f"\n· closed, slot NOT knowable {s_unk:.2f} (honest ≈ chance)")
            else:
                for m, c, mk, sz in ((closed & known, None, "x", 11), (~closed, None, "o", 7)):
                    ax.scatter(Y[m, 0], Y[m, 1], s=sz, c=np.array(SLOT)[slot[m]], marker=mk,
                               linewidths=.7 if mk == "x" else 0, alpha=.8, zorder=2)
                m3 = closed & (belief == 3)
                ax.scatter(Y[m3, 0], Y[m3, 1], s=12, c=ANY3, marker="x", linewidths=.8, alpha=.8, zorder=3)
                m2a = closed & (belief == 4)
                ax.scatter(Y[m2a, 0], Y[m2a, 1], s=16, c=TWO, marker="v", linewidths=0, alpha=.9, zorder=3)
                m2b = closed & (belief >= 5)
                ax.scatter(Y[m2b, 0], Y[m2b, 1], s=16, c=TWO, marker="^", linewidths=0, alpha=.9, zorder=3)
                txt = f"same honest belief among neighbours (closed starts): {b_closed:.2f}"
            ax.text(0, 1.012, txt, transform=ax.transAxes, fontsize=9.4, color=INK2, va="bottom")
            del order
        axes[0, j].set_title(f"{nm}\n\n", loc="left", fontsize=14, color=INK, fontweight="bold")
    axes[0, 0].set_ylabel("colour = where the ball really is", fontsize=12.5, color=INK, labelpad=8, fontweight="bold")
    axes[1, 0].set_ylabel("colour = honest belief", fontsize=12.5, color=INK, labelpad=8, fontweight="bold")
    h = [Line2D([], [], ls="", marker="s", ms=9, color=SLOT[s], label=f"ball / known: {SLOT_NAME[s]}") for s in range(3)]
    h += [Line2D([], [], ls="", marker="x", ms=8, mew=1.6, color=ANY3, label="belief: any of the three (1/3)"),
          Line2D([], [], ls="", marker="v", ms=8, color=TWO, label="belief: middle or right (left seen empty)"),
          Line2D([], [], ls="", marker="^", ms=8, color=TWO, label="belief: left or one other (1/2)"),
          Line2D([], [], ls="", marker="o", ms=6, color=INK2, label="dot = start shown"),
          Line2D([], [], ls="", marker="x", ms=7, mew=1.4, color=INK2, label="cross = start closed")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=10.5, labelcolor=INK2,
               bbox_to_anchor=(.5, -.005))
    fig.suptitle("ALL3-U: chained-posterior states at resting frames (t-SNE of 50-D PCA)", x=.012, ha="left",
                 fontsize=15, color=INK, y=.995)
    fig.text(.012, .968, "Test sets: start shown (memtest4_eval, 0–4 swaps) and closed start (memtest4_evalu, 0–8 swaps). "
             "Neighbour scores: share of the 10 nearest states from OTHER episodes that agree.", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, .055, 1, .955))
    fig.savefig(out, dpi=110, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    main()
