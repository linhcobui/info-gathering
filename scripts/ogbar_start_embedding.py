"""Each OGBar episode's observable start state as one vector, projected to 2-D: how sparse are 400 starts jointly?

Start vector (simulator state at frame 0, privileged, used only to describe the data), each entry standardised:
  bar x, bar y, cos(2*bar yaw), sin(2*bar yaw)        (2*yaw: the bar's two ends look identical)
  gripper x, y, z, cos(2*gripper yaw), sin(2*gripper yaw)
Colour = c_view at the start (COM offset, + = heavy end on the image's right). Also: how many cells of a 10-bin-per-
feature grid the starts occupy, per feature and jointly.

    pixi run python scripts/ogbar_start_embedding.py --dump      # start vectors -> npz (zarr)
    python3 scripts/ogbar_start_embedding.py                     # figure (sklearn)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402

NPZ = ROOT / "figures/ogbar_smt/start_states.npz"
NAMES = ["bar x", "bar y", "bar cos 2yaw", "bar sin 2yaw", "grip x", "grip y", "grip z", "grip cos 2yaw", "grip sin 2yaw"]


def dump():
    import zarr
    out = {}
    def feats(z, idx):
        bar, eff = z["bar_pos"][:][idx], z["effector_pos"][:][idx]
        by, gy = z["bar_yaw_mod_pi"][:][idx], z["effector_yaw"][:][idx]
        return np.c_[bar[:, 0], bar[:, 1], np.cos(2 * by), np.sin(2 * by), eff[:, 0], eff[:, 1], eff[:, 2],
                     np.cos(2 * gy), np.sin(2 * gy)]

    zt = zarr.open_group(str(ROOT / "data/ogbar_v3_400.zarr"), "r")
    ends = zt["episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    fr = [np.arange(a, b, 5) for a, b in zip(starts, ends)]
    allf = feats(zt, np.concatenate(fr))
    ep_id = np.concatenate([np.full(len(f), i) for i, f in enumerate(fr)])
    out["X_steps"], out["ep_steps"] = allf, ep_id                    # every model step, in episode order
    for tag, store in (("train", "data/ogbar_v3_400.zarr"), ("eval", "data/ogbar_v3_eval5.zarr")):
        z = zarr.open_group(str(ROOT / store), "r")
        s0 = np.r_[0, z["episode_ends"][:][:-1]]
        bar, eff = z["bar_pos"][:][s0], z["effector_pos"][:][s0]
        by, gy = z["bar_yaw_mod_pi"][:][s0], z["effector_yaw"][:][s0]
        q = z["bar_quat"][:][s0]
        w, x, y, zz = q.T
        ax = 1 - 2 * (y * y + zz * zz)
        out[f"X_{tag}"] = np.c_[bar[:, 0], bar[:, 1], np.cos(2 * by), np.sin(2 * by), eff[:, 0], eff[:, 1], eff[:, 2],
                                np.cos(2 * gy), np.sin(2 * gy)]
        out[f"cview_{tag}"] = -z["episode/com_offset_c"][:] * 100 * np.sign(ax)
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ, **out)
    print(NPZ)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from figure_style import FAINT, HIGHLIGHT, INK, MUTED, apply
    apply()
    d = np.load(NPZ)
    Xt, Xe, ct, ce = d["X_train"], d["X_eval"], d["cview_train"], d["cview_eval"]
    mu, sd = Xt.mean(0), Xt.std(0)
    St, Se = (Xt - mu) / sd, (Xe - mu) / sd
    n, k = St.shape
    # occupancy of a 10-bins-per-feature grid (bins = deciles of the training starts)
    edges = [np.quantile(St[:, j], np.linspace(0, 1, 11)[1:-1]) for j in range(k)]
    cells = np.stack([np.searchsorted(edges[j], St[:, j]) for j in range(k)], 1)
    joint = len({tuple(c) for c in cells})
    print(f"{n} starts, {k} features; 10 decile bins per feature: every single feature fills 10/10 bins; jointly "
          f"{joint} of 10^{k} = {10 ** k:.0e} cells ({joint / 10 ** k:.1e}), i.e. every start alone in its cell")
    D = np.linalg.norm(St[:, None] - St[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    med_pair = np.median(D[np.isfinite(D)])
    print(f"standardised {k}-D distance: nearest other start median {np.median(nn):.2f}, typical pair {med_pair:.2f} "
          f"(ratio {np.median(nn) / med_pair:.2f}); in 2-D, 400 uniform points would give a ratio of about "
          f"{0.5 / np.sqrt(n) / 0.52:.2f}")
    De = np.linalg.norm(Se[:, None] - St[None], axis=-1).min(1)
    print("eval starts: distance to the nearest training start " + " ".join(f"{v:.2f}" for v in De)
          + f" (training NN median {np.median(nn):.2f})")
    pca = PCA(k).fit(St)
    print("PCA explained variance: " + " ".join(f"{v:.2f}" for v in pca.explained_variance_ratio_))
    P, Pe = pca.transform(St), pca.transform(Se)
    Y = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(np.r_[St, Se])
    cmap = LinearSegmentedColormap.from_list("cview", ["#0048a6", FAINT, "#850f67"])
    norm = Normalize(-12.6, 12.6)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    for ax, (a, b), (ea, eb), title in (
            (axes[0], (P[:, 0], P[:, 1]), (Pe[:, 0], Pe[:, 1]),
             f"PCA of the {k}-D start vector (PC1+PC2 = {pca.explained_variance_ratio_[:2].sum():.0%} of variance)"),
            (axes[1], (Y[:n, 0], Y[:n, 1]), (Y[n:, 0], Y[n:, 1]), "t-SNE of the same start vectors")):
        ax.scatter(a, b, c=[cmap(norm(v)) for v in ct], s=16, edgecolors="none", label="training start")
        ax.scatter(ea, eb, c=[cmap(norm(v)) for v in ce], marker="*", s=130, edgecolors=INK, linewidths=0.8,
                   label="eval start")
        ax.set_title(title, loc="left")
        ax.legend(loc="lower left", fontsize=7)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    ax = axes[2]
    Sx = (d["X_steps"] - mu) / sd                                       # same standardisation as the starts
    same = d["ep_steps"][1:] == d["ep_steps"][:-1]
    step = np.linalg.norm(Sx[1:] - Sx[:-1], axis=1)[same]               # consecutive model steps, same episode
    pairs = D[np.isfinite(D)]
    print(f"one model step (5 raw steps) moves median {np.median(step):.2f} (mean {step.mean():.2f}, 90% {np.percentile(step, 90):.2f}; "
          f"{(step < 1e-3).mean():.0%} of steps ~0); nearest other start median {np.median(nn):.2f}; random pair {np.median(pairs):.2f}; "
          f"steps to cross the median nearest-start gap at the median step size: {np.median(nn) / np.median(step):.0f}")
    bins = np.linspace(0, 7.5, 61)
    ax.hist(step, bins=bins, density=True, color=HIGHLIGHT, alpha=0.85,
            label=f"one model step within an episode (n = {len(step)})")
    ax.hist(nn, bins=bins, density=True, color=MUTED, alpha=0.8, label=f"nearest other start (n = {n})")
    ax.hist(pairs, bins=bins, density=True, color=FAINT, alpha=0.8, label=f"random pair of starts (n = {len(pairs) // 2})")
    ax.set_yscale("log")
    ax.set_xlabel(f"distance in the standardised {k}-D state space")
    ax.set_ylabel("density (log)")
    ax.set_title(f"every start alone: {joint} of {n} in its own 10-bin grid cell", loc="left")
    ax.legend(fontsize=7)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[:2], fraction=0.025, pad=0.01)
    cb.set_label("c_view at the start (cm), privileged, colour only")
    fig.suptitle(f"OGBar v3: each episode's observable start state as one point (N = {n} training + {len(Xe)} eval). "
                 f"Features: {', '.join(NAMES)}; each standardised by its mean and std over the training starts.",
                 x=0.01, ha="left", fontsize=9)
    out = ROOT / "figures/ogbar_smt/start_state_embedding.png"
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    dump() if "--dump" in sys.argv else plot()
