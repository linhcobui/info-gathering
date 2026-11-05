"""Histograms of the frozen DINO-WM's embedding prediction error on a sample of episodes.

Error of one patch at one step: L2 distance between the WM's teacher-forced one-step prediction of a DINO patch token
(visual part, 384 numbers) and the real token of the next frame (context: the 3 real frames before it). Reference:
the copy-last baseline (distance between the real tokens of consecutive frames, i.e. how much the embedding changes),
and the median norm of a real token.

    error_hist_pooled.png    per-patch error over all sampled steps and patches, held-out vs WM-train episodes,
                             plus the per-step summaries err_l2_mean (mean over 256 patches) and err_l2_topk (16 worst)
    error_hist_episodes.png  one panel per sampled episode: per-patch error vs the copy-last baseline

    pixi run python scripts/seg_error_hist.py --n-per-group 6
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from figure_style import HIGHLIGHT, INK, MUTED, apply  # noqa: E402
from seg.config import load_config, out_dir  # noqa: E402
from seg.episodes import load_episodes  # noqa: E402
from seg.metrics import context_index, to_device  # noqa: E402
from seg.wm import DinoWMAdapter  # noqa: E402


@torch.no_grad()
def patch_errors(wm, ep, batch=64):
    """(steps, 256) per-patch L2 of the WM prediction and of copy-last, and (steps, 256) real token norms."""
    H = wm.history_len
    vis, prop, act = to_device(ep, wm.device)
    ts = np.arange(H - 1, ep.T - 1)
    err, copy = [], []
    for b in range(0, len(ts), batch):
        t = torch.as_tensor(ts[b:b + batch], device=wm.device)
        pred = wm.predict_next(*(x[context_index(t, H, wm.device)] for x in (vis, prop, act)))
        real = vis[t + 1].float()
        err.append((pred - real).norm(dim=-1).cpu().numpy())
        copy.append((vis[t].float() - real).norm(dim=-1).cpu().numpy())
    return np.concatenate(err), np.concatenate(copy), vis[ts + 1].float().norm(dim=-1).cpu().numpy()


def step_hist(ax, values, bins, colour, label, lw=1.3):
    ax.hist(values, bins=bins, density=True, histtype="step", color=colour, lw=lw, label=label)
    ax.axvline(np.median(values), color=colour, lw=0.8, ls=":")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-group", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(overrides=args.overrides)
    out = out_dir(cfg) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    episodes = load_episodes(cfg)
    pick = []
    for flag in (False, True):
        eps = [e for e in episodes if e.wm_train == flag and e.T == 200]
        pick += [eps[i] for i in sorted(rng.choice(len(eps), args.n_per_group, replace=False))]
    wm = DinoWMAdapter(ROOT / cfg.paths.wm_ckpt, cfg.device, cfg.errors.bf16)
    res = {e.episode_id: patch_errors(wm, e) for e in pick}
    steps = pd.read_parquet(out_dir(cfg) / "wm_errors.parquet")
    steps = steps[steps.episode_id.isin(res)]
    norm_med = float(np.median(np.concatenate([r[2].ravel() for r in res.values()])))

    apply()
    groups = ((False, HIGHLIGHT, "held-out"), (True, MUTED, "WM-train (in-sample)"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    allp = np.concatenate([res[e.episode_id][0].ravel() for e in pick])
    bins = np.linspace(0, np.quantile(allp, 0.999), 90)
    for flag, c, name in groups:
        ids = [e.episode_id for e in pick if e.wm_train == flag]
        v = np.concatenate([res[i][0].ravel() for i in ids])
        step_hist(axes[0], v, bins, c, f"{name}: median {np.median(v):.1f}, N = {len(ids)} eps / {v.size:,} patch-steps")
    cp = np.concatenate([r[1].ravel() for r in res.values()])
    step_hist(axes[0], cp, bins, INK, f"copy-last baseline (all sampled): median {np.median(cp):.1f}", lw=0.9)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(f"per-patch L2 error ‖ê − e‖ (median token norm ‖e‖ = {norm_med:.0f})")
    axes[0].set_ylabel("density (log)")
    axes[0].set_title("one patch at one step", fontsize=9)
    for ax, m, title in ((axes[1], "err_l2_mean", "per step: mean over 256 patches"),
                         (axes[2], "err_l2_topk", f"per step: mean of the {cfg.errors.topk} worst patches")):
        b = np.linspace(steps[m].quantile(0.001), steps[m].quantile(0.999), 60)
        for flag, c, name in groups:
            v = steps.loc[steps.wm_train_episode == flag, m].to_numpy()
            step_hist(ax, v, b, c, f"{name}: median {np.median(v):.1f}, N = {v.size} steps")
        ax.set_xlabel(f"{m}")
        ax.set_title(title, fontsize=9)
    for ax in axes:
        ax.legend(fontsize=6.5)
    fig.suptitle("DINO-WM one-step embedding prediction error (teacher-forced; DINOv2 patch tokens, visual part; "
                 "dotted = medians)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "error_hist_pooled.png")
    plt.close(fig)

    n = len(pick)
    cols = args.n_per_group
    fig, axes = plt.subplots(2, cols, figsize=(2.6 * cols, 5.0), sharex=True, sharey=True, squeeze=False)
    for ax, e in zip(axes.ravel(), pick):
        err, copy, _ = res[e.episode_id]
        c = MUTED if e.wm_train else HIGHLIGHT
        step_hist(ax, err.ravel(), bins, c, f"WM error, median {np.median(err):.1f}")
        step_hist(ax, copy.ravel(), bins, INK, f"copy-last, median {np.median(copy):.1f}", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{e.episode_id} ({'WM-train' if e.wm_train else 'held-out'})", fontsize=7.5)
        ax.legend(fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("per-patch L2 error")
    for ax in axes[:, 0]:
        ax.set_ylabel("density (log)")
    fig.suptitle(f"Per-patch DINO-WM error per episode ({n} random full-length episodes; each panel 197 steps x 256 patches)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "error_hist_episodes.png")
    plt.close(fig)
    print(out / "error_hist_pooled.png")
    print(out / "error_hist_episodes.png")


if __name__ == "__main__":
    main()
