"""t-SNE grid of rest states from scripts/rest_states_extract.py. Run with system python3 (scikit-learn).

    python3 scripts/rest_states_tsne.py <npz> <out.png> ["NAME=subtitle" ...]

Columns = runs. Rows = colour by the ball's slot now / swaps in the episode / where the ball started.
x = the episode's swap sequence is unseen by that run. Each panel reports, over 10 nearest neighbours
FROM OTHER EPISODES (50-D PCA space): same slot (seen / unseen), same swap count, same start slot.
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

SURF, INK, INK2, MUTED, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
SWAP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]       # ordinal blue ramp, 0..4 swaps


def purity(X, lab, ep, mask=None, k=10):
    idx = NearestNeighbors(n_neighbors=min(150, len(X))).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    h = []
    for i, row in enumerate(idx):
        if mask is not None and not mask[i]:
            continue
        keep = row[ep[row] != ep[i]][:k]
        if len(keep) == k:
            h.append((lab[keep] == lab[i]).mean())
    return float(np.mean(h)) if h else float("nan")


def main():
    d = np.load(sys.argv[1], allow_pickle=True)
    out = sys.argv[2]
    sub = dict(a.split("=", 1) for a in sys.argv[3:])
    names = list(d["names"])
    depth, start = d["depth"], d["start"]
    n = len(names)
    fig, axes = plt.subplots(3, n, figsize=(5.2 * n, 15.2), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURF)
    rng = np.random.default_rng(0)
    for j, nm in enumerate(names):
        Z, S, E = d[f"z{j}"], d[f"s{j}"].astype(int), d[f"e{j}"]
        X = PCA(50, random_state=0).fit_transform(Z)
        Em = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(X)
        sw, st, un = depth[E], start[E], ~d[f"seen{j}"][E]
        p_seen, p_un = purity(X, S, E, ~un), purity(X, S, E, un)
        p_sw, p_st = purity(X, sw, E), purity(X, st, E)
        order = rng.permutation(len(S))
        for i, (lab, cols, title, txt) in enumerate((
                (S, SLOT, "ball slot now", f"same-slot neighbours: seen {p_seen:.2f} · unseen {p_un:.2f}"),
                (sw, SWAP, "swaps in the episode", f"same-count neighbours {p_sw:.2f}  (history: lower is better)"),
                (st, SLOT, "where the ball STARTED", f"same-start neighbours {p_st:.2f}  (history: lower is better)"))):
            ax = axes[i, j]
            ax.set_facecolor(SURF)
            for mk, m, sz in (("o", ~un, 9), ("x", un, 22)):
                o = order[m[order]]
                ax.scatter(Em[o, 0], Em[o, 1], s=sz, marker=mk, c=[cols[v] for v in lab[o]], alpha=0.75,
                           linewidths=0.35 if mk == "o" else 1.0, edgecolor=SURF if mk == "o" else None, zorder=3)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(EDGE)
            ax.text(0, 1.015, txt, transform=ax.transAxes, fontsize=9.2, color=INK2)
            if j == 0:
                ax.set_ylabel(title, fontsize=13, color=INK, labelpad=10, fontweight="bold")
        axes[0, j].set_title(f"{nm}\n", loc="left", fontsize=13.5, color=INK, fontweight="bold", pad=16)
        axes[0, j].text(0, 1.075, sub.get(nm, ""), transform=axes[0, j].transAxes, fontsize=9.6, color=INK2)
        print(nm, f"slot seen {p_seen:.3f} unseen {p_un:.3f} | swaps {p_sw:.3f} | start {p_st:.3f}", flush=True)
    h = [Line2D([], [], ls="", marker="o", ms=8, color=SLOT[s], label=f"slot {s}") for s in range(3)]
    h += [Line2D([], [], ls="", marker="o", ms=8, color=SWAP[k], label=f"{k} swaps") for k in range(5)]
    h += [Line2D([], [], ls="", marker="o", ms=7, color=MUTED, label="sequence seen in that run's training"),
          Line2D([], [], ls="", marker="x", ms=8, mew=1.6, color=MUTED, label="sequence NOT seen")]
    fig.legend(handles=h, loc="lower center", ncol=10, frameon=False, fontsize=10.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Posterior memory at motionless hold frames — memtest4_eval (0–4 swaps)",
                 x=0.012, ha="left", fontsize=14.5, color=INK, y=0.995)
    fig.text(0.012, 0.028, "Neighbour scores: share of each state's 10 nearest states FROM OTHER EPISODES (50-D PCA "
             "space) with the same colour. Chance ≈ 0.33 for slot/start, ≈ 0.20 for swaps. The lifted cup is "
             "always slot 0 (blue).", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    fig.savefig(out, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    main()
