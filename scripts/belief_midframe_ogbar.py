"""One belief per episode: OGBar-SMT teacher z*_t at the middle model step, t-SNE and PCA against the COM.

Training store (all 400 episodes) plus the unseen eval episodes. Labels (privileged, scoring/colour only):
  c       COM offset along the bar's body x axis (cm)
  c_view  the same, signed by which side of the side-view image the heavy end is on at that step (+ = image right;
          camera D's image-right is world -x). The ends look identical, so this is the version the camera can track
  |c|     distance of the COM from the middle
  probed  a probe lift happened before the middle step (the history could hold COM evidence)

    pixi run python scripts/belief_midframe_ogbar.py --ckpt <wm_latest.pt>       # beliefs -> figures/*.npz (torch)
    python3 scripts/belief_midframe_ogbar.py --plot figures/ogbar_smt_belief_midframe.npz   # t-SNE / PCA / probes (sklearn)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#2b2a27", "#6b6a65", "#e7e6e2"
COLOURS = ["#184f95", "#2a78d6", "#c9c7c1", "#e34948", "#a32f2e"]


def bar_axis_x(q):
    w, x, y, zz = q.T
    return 1 - 2 * (y * y + zz * zz)


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
    steps = [int(v) for v in str(args.steps).split(",")] if args.steps else [args.step]
    out = {"Z": [], "F": [], "c": [], "c_view": [], "probed": [], "unseen": [], "episode": []}
    for k in steps:
        out[f"Z_{k}"], out[f"cview_{k}"], out[f"probed_{k}"] = [], [], []
    for store, unseen in (("data/ogbar_v3_400.zarr", 0), ("data/ogbar_v3_eval5.zarr", 1)):
        data = OgbarSmtAnchors(ROOT / (store.removesuffix(".zarr") + "_dinov2"), ROOT / store, 5)
        z = zarr.open_group(str(ROOT / store), "r")
        ps = z["probe_stage"][:]
        for e in range(len(data.ends)):
            s0 = int(data.starts[e])
            S = (int(data.ends[e]) - s0) // 5
            t = min(args.step, S - 1)
            p, a = data.load(e, 0)
            p, a = p[None].to(dev).to(torch.bfloat16), m.norm_action(a[None].to(dev))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pooled = m.pool(p)
                zt = m.teacher_windows(pooled, a, torch.full((1, 1), t, dtype=torch.long, device=dev))[0, 0]
                for k in steps:
                    tk = min(k, S - 1)
                    zk = m.teacher_windows(pooled, a, torch.full((1, 1), tk, dtype=torch.long, device=dev))[0, 0]
                    out[f"Z_{k}"].append(zk.float().flatten().cpu().numpy())
                    axk = bar_axis_x(z["bar_quat"][s0 + tk * 5][None])[0]
                    out[f"cview_{k}"].append(-float(z["episode/com_offset_c"][e]) * 100 * np.sign(axk))
                    out[f"probed_{k}"].append(int((ps[s0:s0 + tk * 5] == 1).any()))
            out["Z"].append(zt.float().flatten().cpu().numpy())
            out["F"].append(p[0, t].float().reshape(4, 4, 4, 4, 384).mean((1, 3)).flatten().cpu().numpy())
            c = float(z["episode/com_offset_c"][e]) * 100
            ax = bar_axis_x(z["bar_quat"][s0 + t * 5][None])[0]
            out["c"].append(c)
            out["c_view"].append(-c * np.sign(ax))
            out["probed"].append(int((ps[s0:s0 + t * 5] == 1).any()))
            out["unseen"].append(unseen)
            out["episode"].append(e)
    np.savez(ROOT / args.out, step=st["step"], mid=args.step, steps=np.array(steps), **{k: np.array(v) for k, v in out.items()})
    print(ROOT / args.out, len(out["c"]), "episodes")


def cv_r2(X, y, lams=(1e-1, 1, 10, 100, 1e3, 1e4)):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    best = -np.inf
    for lam in lams:
        pred = cross_val_predict(Ridge(alpha=lam), X, y, cv=KFold(5, shuffle=True, random_state=0))
        best = max(best, 1 - ((pred - y) ** 2).mean() / y.var())
    return best


def plot(npz, colour="c"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    d = np.load(npz)
    Z, F, c, cv, probed, unseen = d["Z"], d["F"], d["c"], d["c_view"], d["probed"].astype(bool), d["unseen"].astype(bool)
    col = cv if colour == "c_view" else c
    Zs = (Z - Z[~unseen].mean(0)) / (Z[~unseen].std(0) + 1e-6)
    seen = ~unseen
    print(f"step {int(d['step'])}, belief at model step {int(d['mid'])}; {seen.sum()} seen + {unseen.sum()} unseen episodes; "
          f"probed before the middle: {probed[seen].mean():.0%} of seen")
    print("linear probe, 5-fold CV R^2 on the seen episodes (best ridge lambda):")
    for name, X in (("belief z*", Zs[seen]), ("frame tokens (4x4)", F[seen])):
        print(f"  {name:20s} c {cv_r2(X, c[seen]):+.3f}   c_view {cv_r2(X, cv[seen]):+.3f}   |c| {cv_r2(X, np.abs(c[seen])):+.3f}")
        if name.startswith("belief"):
            ps = probed & seen
            print(f"  {'  ...probed only':20s} c {cv_r2(Zs[ps], c[ps]):+.3f}   c_view {cv_r2(Zs[ps], cv[ps]):+.3f}   |c| {cv_r2(Zs[ps], np.abs(c[ps])):+.3f}   (n={ps.sum()})")
    pca = PCA(20).fit(Zs[seen])
    P = pca.transform(Zs)
    corr = {k: np.array([np.corrcoef(P[seen, i], y[seen])[0, 1] for i in range(20)]) for k, y in (("c", c), ("c_view", cv), ("|c|", np.abs(c)))}
    print("PCA (20 components, fitted on seen):  explained variance " + " ".join(f"{v:.2f}" for v in pca.explained_variance_ratio_[:6]) + " ...")
    for k, r in corr.items():
        i = int(np.argmax(np.abs(r)))
        print(f"  best single PC for {k:6s}: PC{i + 1} r = {r[i]:+.2f}   (|r| of PC1..5: " + " ".join(f"{abs(v):.2f}" for v in r[:5]) + ")")
    key = max(corr, key=lambda k: np.abs(corr[k]).max())
    best = int(np.argmax(np.abs(corr[key])))
    y_best = {"c": c, "c_view": cv, "|c|": np.abs(c)}[key]
    Y = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Zs)

    CMAP = LinearSegmentedColormap.from_list("com", COLOURS)
    norm = Normalize(-12.6, 12.6)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), facecolor=SURFACE)

    def dots(ax, x, y, col):
        for sel, mk, lab in ((seen & probed, "o", "seen, probed before the middle"), (seen & ~probed, "^", "seen, not yet probed"),
                             (unseen, "*", "unseen eval episode")):
            ax.scatter(x[sel], y[sel], c=[CMAP(norm(v)) for v in col[sel]], marker=mk, s=60 if mk == "*" else 26,
                       edgecolors=INK if mk == "*" else "none", linewidths=0.6, label=lab)

    dots(axes[0], Y[:, 0], Y[:, 1], col)
    axes[0].set_title("t-SNE of the belief at the middle step (one per episode)", color=INK, fontsize=10.5, loc="left")
    dots(axes[1], P[:, 0], P[:, 1], col)
    axes[1].set_title(f"PCA: PC1 vs PC2 ({pca.explained_variance_ratio_[:2].sum():.0%} of variance)", color=INK, fontsize=10.5, loc="left")
    axes[1].set_xlabel("PC1", color=MUTED); axes[1].set_ylabel("PC2", color=MUTED)
    dots(axes[2], y_best, P[:, best], col)
    axes[2].set_title(f"best single PC for {key}: PC{best + 1}, r = {corr[key][best]:+.2f} (seen)", color=INK, fontsize=10.5, loc="left")
    axes[2].set_xlabel(f"{key} (cm), privileged", color=MUTED); axes[2].set_ylabel(f"PC{best + 1}", color=MUTED)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    leg = axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.02), ncol=3, frameon=False, fontsize=8, labelcolor=MUTED)
    for h in getattr(leg, "legend_handles", None) or leg.legendHandles:
        h.set_facecolor(MUTED)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes, fraction=0.02, pad=0.01)
    cb.set_label(("c_view: COM, + = heavy end on image right" if colour == "c_view" else "centre of mass c, bar frame")
                 + " (cm), privileged - colour only", color=MUTED)
    cb.outline.set_edgecolor(GRID); cb.ax.tick_params(colors=MUTED)
    fig.suptitle(f"OGBar-SMT belief z* at model step {int(d['mid'])}, checkpoint step {int(d['step'])}: one point per episode, "
                 f"colour = {'COM side as seen by the camera (c_view)' if colour == 'c_view' else 'centre of mass (bar frame)'}", color=INK, fontsize=12, x=0.01, ha="left")
    out = Path(npz).with_name(Path(npz).stem + ("_cview" if colour == "c_view" else "") + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(out)


def plot_steps(npz, ref_step=100):
    """Rows = model steps (e.g. 0 and 100), same PCA basis (fitted on the seen episodes' beliefs at ref_step).
    Left: PC1 vs PC2; right: PC1 vs c_view, with the straight line fitted at ref_step."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from sklearn.decomposition import PCA
    from figure_style import FAINT, INK, MUTED, apply
    apply()
    d = np.load(npz)
    steps = [int(k) for k in d["steps"]]
    unseen = d["unseen"].astype(bool)
    seen = ~unseen
    Zr = d[f"Z_{ref_step}"][seen]
    mu, sd = Zr.mean(0), Zr.std(0) + 1e-6
    pca = PCA(20).fit((Zr - mu) / sd)
    cref = d[f"cview_{ref_step}"][seen]
    Pr = pca.transform((Zr - mu) / sd)
    k = np.polyfit(cref, Pr[:, 0], 1)
    cmap = LinearSegmentedColormap.from_list("cview", ["#0048a6", FAINT, "#850f67"])
    norm = Normalize(-12.6, 12.6)
    fig, axes = plt.subplots(len(steps), 2, figsize=(10.5, 4.3 * len(steps)), squeeze=False)
    print(f"checkpoint step {int(d['step'])}; PCA fitted on {seen.sum()} seen episodes at model step {ref_step}")
    for row, t in enumerate(steps):
        Z, cv, pr = d[f"Z_{t}"], d[f"cview_{t}"], d[f"probed_{t}"].astype(bool)
        P = pca.transform((Z - mu) / sd)
        r = np.corrcoef(P[seen, 0], cv[seen])[0, 1]
        est = (P[:, 0] - k[1]) / k[0]                                   # PC1 read as c_view with the ref-step line
        for ax, (x, y, xl, yl) in zip(axes[row], ((P[:, 0], P[:, 1], "PC1", "PC2"), (cv, P[:, 0], "true c_view (cm)", "PC1"))):
            for sel, mk, lab in ((seen & pr, "o", "seen, probed before this step"), (seen & ~pr, "^", "seen, not yet probed"),
                                 (unseen, "*", "unseen eval episode")):
                if sel.any():
                    ax.scatter(x[sel], y[sel], c=[cmap(norm(v)) for v in cv[sel]], marker=mk, s=70 if mk == "*" else 18,
                               edgecolors=INK if mk == "*" else "none", linewidths=0.6, label=lab)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
        lim = np.array([-12.6, 12.6])
        axes[row, 1].plot(lim, k[0] * lim + k[1], color=MUTED, lw=1, ls="--", label=f"line fitted at step {ref_step}")
        axes[row, 0].set_title(f"model step {t}: belief z*_t, PC1 vs PC2", loc="left")
        axes[row, 1].set_title(f"model step {t}: PC1 vs true c_view   r = {r:+.2f} (seen)", loc="left")
        axes[row, 0].set_xlim(*np.percentile(pca.transform((Zr - mu) / sd)[:, 0], [0.5, 99.5]) * 1.4)
        print(f"  step {t:3d}: r(PC1, c_view) seen {r:+.2f}; PC1 std seen {P[seen, 0].std():.1f} (step-{ref_step} std "
              f"{Pr[:, 0].std():.1f}); probed before this step {pr[seen].mean():.0%}; PC1 read as c_view, seen: "
              f"spread {est[seen].std():.1f} cm vs true {cv[seen].std():.1f} cm; unseen: "
              + " ".join(f"{a:+.1f}/{b:+.1f}" for a, b in zip(est[unseen], cv[unseen])) + " (estimate/true cm)")
    axes[0, 1].legend(loc="upper left", fontsize=7)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("c_view: COM offset (cm), + = heavy end on the image's right; privileged, colour only")
    fig.suptitle(f"OGBar-SMT teacher belief at the start vs the middle of each episode (checkpoint step {int(d['step'])}; "
                 f"N = {seen.sum()} seen + {unseen.sum()} unseen episodes)\nPCA (20 components) fitted on the seen episodes' "
                 f"beliefs at step {ref_step}, each number standardised by its mean and std there; the same basis for every row",
                 x=0.01, ha="left", fontsize=9)
    out = ROOT / "figures/ogbar_smt" / "belief_start_vs_mid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--plot")
    ap.add_argument("--colour", default="c", choices=["c", "c_view"])
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--steps", default=None, help="comma list of model steps to save (e.g. 0,100)")
    ap.add_argument("--plot-steps", default=None, help="npz from --steps: start-vs-mid figure")
    ap.add_argument("--out", default="figures/ogbar_smt_belief_midframe.npz")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    if a.plot_steps:
        plot_steps(a.plot_steps)
    elif a.plot:
        plot(a.plot, a.colour)
    else:
        compute(a)
