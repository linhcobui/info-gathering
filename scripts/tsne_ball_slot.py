"""t-SNE of posterior memories at rest frames, coloured ONLY by the hidden ball position (privileged, for display).

    python3 scripts/tsne_ball_slot.py <out.png> "ROW TITLE=a.npz[,b.npz...]" ["ROW 2=..."]      # system python3

Inputs come from scripts/rest_states_extract.py (one npz per test store; several npz in a row are pooled). Every
run (column) gets its own PCA(50) + t-SNE. Each panel prints the same-slot neighbour score: for each state, the
share of its 10 nearest states (in the PCA space) from OTHER episodes whose ball is in the same cup.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import FAINT, HIGHLIGHT, INK, INTERESTING, MUTED, apply  # noqa: E402

SLOT = {0: (HIGHLIGHT, "left cup (the lifted one)"), 1: (INTERESTING[0], "middle cup"), 2: (INTERESTING[2], "right cup")}


def purity(X, lab, ep, k=10):
    idx = NearestNeighbors(n_neighbors=min(150, len(X))).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    h = [(lab[r[ep[r] != ep[i]][:k]] == lab[i]).mean() for i, r in enumerate(idx) if (ep[r] != ep[i]).sum() >= k]
    return float(np.mean(h))


def load_row(paths):
    ds = [np.load(p, allow_pickle=True) for p in paths]
    names = list(ds[0]["names"])
    cols = []
    for j in range(len(names)):
        Z, S, E, off = [], [], [], 0
        for d in ds:
            Z.append(d[f"z{j}"]); S.append(d[f"s{j}"].astype(int)); E.append(d[f"e{j}"] + off)
            off += int(d[f"e{j}"].max()) + 1
        cols.append((names[j], np.concatenate(Z), np.concatenate(S), np.concatenate(E)))
    return cols


def main():
    out = sys.argv[1]
    rows = [(a.split("=", 1)[0], a.split("=", 1)[1].split(",")) for a in sys.argv[2:]]
    data = [load_row(p) for _, p in rows]
    n = len(data[0])
    apply(font_size=9)
    fig, axes = plt.subplots(len(rows), n, figsize=(4.3 * n, 4.3 * len(rows) + 0.9), squeeze=False)
    rng = np.random.default_rng(0)
    for i, ((title, _), cols) in enumerate(zip(rows, data)):
        for j, (nm, Z, S, E) in enumerate(cols):
            X = PCA(50, random_state=0).fit_transform(Z)
            Em = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(X)
            p = purity(X, S, E)
            ax = axes[i, j]
            o = rng.permutation(len(S))
            ax.scatter(Em[o, 0], Em[o, 1], s=7, c=[SLOT[s][0] for s in S[o]], linewidths=0, alpha=0.85)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(FAINT)
            ax.text(0.0, 1.01, f"same-cup neighbours {p:.2f}  ·  {len(S)} states, {len(np.unique(E))} episodes",
                    transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
            if i == 0:
                ax.set_title(nm, fontsize=12, color=INK, pad=16)
            if j == 0:
                ax.set_ylabel(title, fontsize=11, color=INK, labelpad=8)
            print(f"{title:28s} {nm:20s} same-cup neighbours {p:.3f}")
    h = [Line2D([], [], ls="", marker="o", ms=7, color=c, label=f"ball under the {lab}") for c, lab in SLOT.values()]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Posterior memory at rest frames of shown-start test episodes, t-SNE per run, coloured by the "
                 "hidden ball position", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=150)
    fig.savefig(str(Path(out).with_suffix(".pdf")))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
