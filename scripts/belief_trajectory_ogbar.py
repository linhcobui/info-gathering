"""Belief trajectories of unseen episodes, in the space set by the seen episodes' mid-step beliefs.

Two ways to get the belief at every model step t of an episode:
  teacher   the teacher encoder re-run on the history up to t (its 200-step window ending at t), at every t
  updater   the teacher once at t = 0 (eval_warmup 1), then z_{t+1} = sample of Q(z_t, x_t, a_t, real x_{t+1})
Projected with the PCA fitted on the 400 seen episodes' beliefs at model step 100 (belief_midframe_ogbar.py), where
PC1 tracks c_view = the COM offset signed by which side of the image the heavy end is on (privileged, scoring only).
PC1 is turned into cm with the straight line fitted on the seen episodes.

    pixi run python scripts/belief_trajectory_ogbar.py --ckpt <wm_latest.pt>          # trajectories -> npz (torch)
    python3 scripts/belief_trajectory_ogbar.py --plot figures/ogbar_smt_belief_traj_eval5.npz \\
        --ref figures/ogbar_smt_belief_midframe_s25000.npz                              # figure (sklearn)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#2b2a27", "#6b6a65", "#e7e6e2"
COLOURS = ["#184f95", "#2a78d6", "#c9c7c1", "#e34948", "#a32f2e"]


def compute(args):
    import torch
    import zarr
    from info_gathering.ogbar_smt_data import OgbarSmtAnchors
    from world_model.models.smt_ogbar import SmtOgbar
    from world_model.utils.checkpoint import CheckpointManager
    torch.manual_seed(0)
    dev = torch.device(args.device)
    st = CheckpointManager.load(args.ckpt)
    m = SmtOgbar(action_dim=25, **st["config"]["model"]).to(dev).eval()
    m.load_state_dict(st["model"])
    data = OgbarSmtAnchors(ROOT / (args.zarr.removesuffix(".zarr") + "_dinov2"), ROOT / args.zarr, 5)
    z = zarr.open_group(str(ROOT / args.zarr), "r")
    out = {}
    for e in range(len(data.ends)):
        s0, e0 = int(data.starts[e]), int(data.ends[e])
        p, a = data.load(e, 0)
        p, a = p.to(dev).to(torch.bfloat16), m.norm_action(a.to(dev))
        T = len(p)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pooled = m.pool(p[None])
            teach = torch.cat([m.teacher_windows(pooled, a[None], torch.arange(i, min(i + 50, T), device=dev)[None])[0]
                               for i in range(0, T, 50)])
            zs = [teach[0]]
            for t in range(T - 1):
                mu, ls = m.posterior(zs[-1][None], p[t][None], a[t][None], p[t + 1][None])
                zs.append(m.sample(mu, ls)[0])
        q = z["bar_quat"][s0:e0][::5][:T]
        w, x, y, zz = q.T
        ax = 1 - 2 * (y * y + zz * zz)
        c = float(z["episode/com_offset_c"][e]) * 100
        ps = z["probe_stage"][s0:e0]
        lifts = [t for t in range(1, len(ps)) if ps[t] == 1 and ps[t - 1] != 1]
        out[f"teacher_{e}"] = teach.float().flatten(1).cpu().numpy()
        out[f"updater_{e}"] = torch.stack(zs).float().flatten(1).cpu().numpy()
        out[f"cview_{e}"] = -c * np.sign(ax)
        out[f"lifts_{e}"] = np.array(lifts) / 5.0
    np.savez(ROOT / args.out, n=len(data.ends), step=st["step"], **out)
    print(ROOT / args.out)


def plot(npz, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    d, r = np.load(npz), np.load(ref)
    seen = ~r["unseen"].astype(bool)
    Zr, cvr = r["Z"][seen], r["c_view"][seen]
    mu, sd = Zr.mean(0), Zr.std(0) + 1e-6
    pca = PCA(20).fit((Zr - mu) / sd)
    Pr = pca.transform((Zr - mu) / sd)
    k = np.polyfit(cvr, Pr[:, 0], 1)                         # PC1 = k0 * c_view + k1 on the seen episodes
    to_cm = lambda pc1: (pc1 - k[1]) / k[0]
    n = int(d["n"])
    CMAP = LinearSegmentedColormap.from_list("com", COLOURS)
    norm = Normalize(-12.6, 12.6)
    fig = plt.figure(figsize=(18, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, n, height_ratios=[1.35, 1], hspace=0.32, wspace=0.28)
    tops = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4])]
    tsne_ax = fig.add_subplot(gs[0, 4])
    proj = {}
    for kind, ax in zip(("teacher", "updater"), tops):
        ax.scatter(Pr[:, 0], Pr[:, 1], c=[CMAP(norm(v)) for v in cvr], s=10, alpha=0.35, edgecolors="none")
        for e in range(n):
            P = pca.transform((d[f"{kind}_{e}"] - mu) / sd)
            proj[(kind, e)] = P
            col = CMAP(norm(d[f"cview_{e}"][0]))
            ax.plot(P[:, 0], P[:, 1], color=col, lw=1.6)
            ax.scatter(P[0, 0], P[0, 1], color=col, marker="o", s=40, edgecolors=INK, linewidths=0.8, zorder=3)
            ax.scatter(P[-1, 0], P[-1, 1], color=col, marker="*", s=140, edgecolors=INK, linewidths=0.8, zorder=3)
        ax.set_title(f"{kind} belief, unseen episodes (o start, * end) on seen mid-step beliefs; PCA", color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel("PC1 (tracks c_view on seen episodes, r = 0.92)", color=MUTED)
        ax.set_ylabel("PC2", color=MUTED)
    # joint t-SNE: seen mid-step beliefs + every 2nd step of the teacher trajectories
    traj = [d[f"teacher_{e}"][::2] for e in range(n)]
    X = np.concatenate([Zr] + traj)
    Y = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform((X - mu) / sd)
    tsne_ax.scatter(Y[:len(Zr), 0], Y[:len(Zr), 1], c=[CMAP(norm(v)) for v in cvr], s=8, alpha=0.4, edgecolors="none")
    o = len(Zr)
    for e in range(n):
        Yt = Y[o:o + len(traj[e])]
        o += len(traj[e])
        col = CMAP(norm(d[f"cview_{e}"][0]))
        tsne_ax.plot(Yt[:, 0], Yt[:, 1], color=col, lw=1.4)
        tsne_ax.scatter(Yt[-1, 0], Yt[-1, 1], color=col, marker="*", s=120, edgecolors=INK, linewidths=0.8, zorder=3)
    tsne_ax.set_title("joint t-SNE (teacher)", color=INK, fontsize=10.5, loc="left")
    tsne_ax.set_xticks([]); tsne_ax.set_yticks([])
    for e in range(n):
        ax = fig.add_subplot(gs[1, e])
        t = np.arange(len(d[f"teacher_{e}"]))
        ax.plot(t, d[f"cview_{e}"], color=INK, lw=1.2, ls="--", label="true c_view")
        ax.plot(t, to_cm(proj[("teacher", e)][:, 0]), color="#2a78d6", lw=1.6, label="teacher")
        ax.plot(t, to_cm(proj[("updater", e)][:, 0]), color="#eb6834", lw=1.6, label="updater (Q chain)")
        for l in d[f"lifts_{e}"]:
            ax.axvline(l, color=GRID, lw=1.0, zorder=0)
        ax.set_ylim(-20, 20)
        ax.set_title(f"eval ep {e}: c_view {d[f'cview_{e}'][0]:+.1f} cm", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("model step (grey = probe lift starts)", color=MUTED, fontsize=8)
        if e == 0:
            ax.set_ylabel("belief PC1 read as c_view (cm)", color=MUTED)
            ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle(f"OGBar-SMT (checkpoint step {int(d['step'])}): belief trajectories of the 5 unseen eval episodes, "
                 "colour = c_view (COM side as seen by the camera)", color=INK, fontsize=12, x=0.01, ha="left")
    out = Path(npz).with_suffix(".png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    print(out)
    for e in range(n):
        tt, uu = to_cm(proj[("teacher", e)][:, 0]), to_cm(proj[("updater", e)][:, 0])
        print(f"ep {e} true {d[f'cview_{e}'][0]:+5.1f} | teacher at steps 0/50/100/150/end: "
              + " ".join(f"{tt[min(i, len(tt) - 1)]:+5.1f}" for i in (0, 50, 100, 150, 10 ** 6))
              + " | updater: " + " ".join(f"{uu[min(i, len(uu) - 1)]:+5.1f}" for i in (0, 50, 100, 150, 10 ** 6))
              + f" | lifts at {[int(l) for l in d[f'lifts_{e}']]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--plot")
    ap.add_argument("--ref", default="figures/ogbar_smt_belief_midframe_s25000.npz")
    ap.add_argument("--zarr", default="data/ogbar_v3_eval5.zarr")
    ap.add_argument("--out", default="figures/ogbar_smt_belief_traj_eval5.npz")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    plot(a.plot, a.ref) if a.plot else compute(a)
