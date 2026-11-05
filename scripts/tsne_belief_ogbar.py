"""t-SNE of the OGBar-SMT belief z*_t, one point per model step, for a few training episodes.

Marker shape = episode, colour = that episode's centre of mass c (privileged; used for colouring only).
Left: the teacher's belief z*_t (window ending at t). Right, for reference: the frame itself (DINOv2 tokens,
averaged to a 4x4 patch grid). Episodes are chosen to span the COM range evenly.

    pixi run python scripts/tsne_belief_ogbar.py --ckpt <wm_latest.pt>        # beliefs -> <out>.npz (needs torch)
    python3 scripts/tsne_belief_ogbar.py --plot figures/ogbar_smt_tsne_belief.npz  # t-SNE + figure (needs sklearn)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#2b2a27", "#6b6a65", "#e7e6e2"
# diverging blue <-> red with a neutral grey midpoint (dataviz palette; midpoint darkened so c = 0 dots stay visible)
COLOURS = ["#184f95", "#2a78d6", "#c9c7c1", "#e34948", "#a32f2e"]


def compute(args):
    import torch
    import zarr
    from info_gathering.ogbar_smt_data import OgbarSmtAnchors
    from world_model.models.smt_ogbar import SmtOgbar
    from world_model.utils.checkpoint import CheckpointManager
    dev = torch.device(args.device)
    st = CheckpointManager.load(args.ckpt)
    m = SmtOgbar(action_dim=25, **st["config"]["model"]).to(dev).eval()
    m.load_state_dict(st["model"])
    data = OgbarSmtAnchors(ROOT / "data/ogbar_v3_400_dinov2", ROOT / "data/ogbar_v3_400.zarr", 5)
    z = zarr.open_group(str(ROOT / "data/ogbar_v3_400.zarr"), "r")
    c = z["episode/com_offset_c"][:] * 100
    full = np.where(z["episode/length"][:] >= 1000)[0]
    order = full[np.argsort(c[full])]
    eps = order[np.linspace(0, len(order) - 1, args.n_episodes).round().astype(int)]

    Z, F, E = [], [], []
    for e in eps:
        p, a = data.load(int(e), 0)
        p, a = p[None].to(dev).to(torch.bfloat16), m.norm_action(a[None].to(dev))
        S = p.shape[1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pooled = m.pool(p)
            zz = torch.cat([m.teacher_windows(pooled, a, torch.arange(i, min(i + 50, S), device=dev)[None])
                            for i in range(0, S, 50)], 1)[0]
        Z.append(zz.float().flatten(1).cpu().numpy())
        F.append(p[0].float().reshape(S, 4, 4, 4, 4, 384).mean((2, 4)).flatten(1).cpu().numpy())
        E += [int(e)] * S
    Z, F, E = np.concatenate(Z), np.concatenate(F), np.array(E)
    out = ROOT / args.out
    np.savez(out, Z=Z, F=F, E=E, eps=eps, c=c, step=st["step"])
    print(out, "episodes", eps.tolist(), "points", len(E))


def plot(npz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from sklearn.manifold import TSNE
    d = np.load(npz)
    Z, F, E, eps, c, step = d["Z"], d["F"], d["E"], d["eps"], d["c"], int(d["step"])
    CMAP = LinearSegmentedColormap.from_list("com", COLOURS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), facecolor=SURFACE)
    norm = Normalize(-12.6, 12.6)
    for ax, X, title in ((axes[0], Z, f"belief z*_t (teacher, step {step})"),
                         (axes[1], F, "frame (DINOv2 tokens, 4x4 grid) - reference")):
        Y = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(
            (X - X.mean(0)) / (X.std(0) + 1e-6))
        for k, e in enumerate(eps):
            sel = E == e
            ax.scatter(Y[sel, 0], Y[sel, 1], s=26, marker=MARKERS[k % len(MARKERS)], c=[CMAP(norm(c[e]))],
                       edgecolors="none", linewidths=0, alpha=0.9, label=f"ep {e}  (c {c[e]:+.1f} cm)")
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    leg = axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=5, frameon=False, fontsize=8,
                         labelcolor=MUTED, markerscale=1.3)
    for h in getattr(leg, "legend_handles", None) or leg.legendHandles:
        h.set_facecolor(MUTED); h.set_edgecolor(SURFACE)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("centre of mass c (cm), privileged - colour only", color=MUTED)
    cb.outline.set_edgecolor(GRID); cb.ax.tick_params(colors=MUTED)
    fig.suptitle("OGBar-SMT: t-SNE per model step - shape = training episode, colour = centre of mass",
                 color=INK, fontsize=12, x=0.01, ha="left")
    out = Path(npz).with_suffix(".png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--plot")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--out", default="figures/ogbar_smt_tsne_belief.npz")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    plot(a.plot) if a.plot else compute(a)
