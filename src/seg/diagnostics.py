"""Step 6b: diagnostics under <out>/diagnostics/ (house figure style, docs/FIGURES.md).

Privileged per-episode events (Episode.events, e.g. grasp onsets) appear here only, to check where windows land;
nothing upstream reads them.
"""
import sys

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .config import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import FAINT, HIGHLIGHT, INK, INTERESTING, MUTED, apply  # noqa: E402


def episode_plot(ep, err, scores, segs, lam, cfg, path):
    """err, scores: this episode's rows (t, err_value) and (t, g); both drawn at the frame they score, t + 1."""
    apply()
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True)
    a0.plot(err.t + 1, err.err_value, color=INK, lw=0.9, label=f"WM error ({cfg.error_metric}, {cfg.normalise})")
    a0.axhline(lam, color=MUTED, ls="--", lw=0.8, label="λ")
    a1.plot(scores.t + 1, scores.g, color=HIGHLIGHT, lw=1.1, label="g (calibrated P(err > λ))")
    a1.axhline(cfg.on_thresh, color=MUTED, ls="--", lw=0.8, label="on / off threshold")
    a1.axhline(cfg.off_thresh, color=MUTED, ls=":", lw=0.8)
    for ax in (a0, a1):
        for sg in segs:
            ax.axvspan(sg.t_start - 0.5, sg.t_end + 0.5, color=HIGHLIGHT, alpha=0.18, lw=0)
        for (name, ts), c in zip(sorted(ep.events.items()), INTERESTING):
            for i, t in enumerate(ts):
                ax.axvline(t, color=c, lw=0.8, alpha=0.7, label=f"{name} (privileged)" if (ax is a0 and i == 0) else None)
    a1.set_ylim(-0.02, 1.02)
    a0.set_ylabel("error")
    a1.set_ylabel("g")
    a1.set_xlabel("predicted frame τ (model step, 5 raw steps)")
    a0.set_title(f"{ep.episode_id}  ({'WM-train' if ep.wm_train else 'held-out'}; shaded = candidate windows, N = {len(segs)})")
    a0.legend(fontsize=7, loc="upper right")
    a1.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def montage(records, episodes, cfg, path, title):
    """One row per window: frames from t_start - 1 on; the window's frames boxed in orange."""
    by_id = {e.episode_id: e for e in episodes}
    n_f, th = cfg.diagnostics.montage_frames, cfg.diagnostics.montage_thumb
    label_w, rows = 260, []
    for r in records:
        ep = by_id[r["episode_id"]]
        ts = [t for t in range(r["t_start"] - 1, r["t_start"] - 1 + n_f) if 0 <= t < ep.T]
        row = Image.new("RGB", (label_w + n_f * th, th), "white")
        d = ImageDraw.Draw(row)
        res = r["resolvability"]
        d.text((4, 4), f"{r['episode_id']}\nframes {r['t_start']}-{r['t_end']}\nres {res:+.3f}\npct {r['resolvability_pct']:.0f}"
               if res is not None else f"{r['episode_id']}\n{r['t_start']}-{r['t_end']}\n{r['skipped']}", fill="black")
        for j, t in enumerate(ts):
            img = ep.frame_rgb(t)
            if img is None:
                continue
            im = Image.fromarray(np.asarray(img, dtype=np.uint8)).resize((th, th), Image.NEAREST if img.shape[0] < th else Image.BILINEAR)
            row.paste(im, (label_w + j * th, 0))
            if r["t_start"] <= t <= r["t_end"]:
                d.rectangle([label_w + j * th, 0, label_w + (j + 1) * th - 1, th - 1], outline=HIGHLIGHT, width=3)
        rows.append(row)
    if not rows:
        return
    head = 22
    out = Image.new("RGB", (rows[0].width, head + th * len(rows)), "white")
    ImageDraw.Draw(out).text((4, 4), title, fill="black")
    for i, row in enumerate(rows):
        out.paste(row, (0, head + i * th))
    out.save(path)


def error_histogram(errors, labels, lam, cfg, path):
    apply()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    v = labels.err_value.to_numpy()
    finite = np.isfinite(v)
    lo, hi = np.quantile(v[finite], [0.001, 0.999]) if finite.any() else (0, 1)
    bins = np.linspace(lo, hi, 80)
    for flag, colour, name in ((True, MUTED, "WM-train episodes (in-sample)"), (False, HIGHLIGHT, "held-out episodes")):
        m = (labels.wm_train_episode == flag).to_numpy() & finite
        if m.any():
            ax.hist(v[m], bins=bins, density=True, histtype="step", color=colour, lw=1.3,
                    label=f"{name}, N = {labels.episode_id[m].nunique()} eps / {m.sum()} steps")
    ax.axvline(lam, color=INK, ls="--", lw=0.8, label=f"λ = {lam:.3g}")
    ax.set_xlabel(f"one-step error {cfg.error_metric} ({cfg.normalise}), DINO patch tokens e, visual part")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def reliability_plot(reports, cfg, path):
    """reports: {name: classification_report}; reliability curves of g on the test split."""
    apply()
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], color=FAINT, lw=0.8, label="perfect calibration")
    colours = [HIGHLIGHT] + INTERESTING
    for (name, rep), c in zip(reports.items(), colours):
        b = rep["reliability"]
        m = np.asarray(b["count"]) > 0
        ax.plot(np.asarray(b["mean_p"])[m], np.asarray(b["frac_pos"])[m], "o-", ms=3, color=c, lw=1,
                label=f"{name}: ECE {rep['ece']:.3f}, N = {rep['n']}")
    ax.set_xlabel("predicted P(err > λ) (bin mean)")
    ax.set_ylabel("observed fraction err > λ")
    ax.set_title(f"g calibration, {cfg.g.ece_bins} equal-width bins", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def resolvability_histogram(records, controls, path):
    apply()
    groups = sorted({r["wm_train_episode"] for r in records if r["gap"] is not None})
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 3.2), squeeze=False)
    for ax, flag in zip(axes[0], groups):
        w = np.array([r["gap"] for r in records if r["gap"] is not None and r["wm_train_episode"] == flag])
        c = np.array([c["gap"] for c in controls if c["group"] in (flag, "all", "heldout")])
        allv = np.concatenate([w, c]) if len(c) else w
        bins = np.linspace(*np.quantile(allv, [0.002, 0.998]), 60) if len(allv) > 1 else 10
        if len(c):
            ax.hist(c, bins=bins, density=True, histtype="step", color=MUTED, lw=1.2, label=f"control windows, N = {len(c)}")
        ax.hist(w, bins=bins, density=True, histtype="step", color=HIGHLIGHT, lw=1.4, label=f"candidate windows, N = {len(w)}")
        ax.axvline(0, color=FAINT, lw=0.8)
        ax.set_xlabel("gap = blind − informed error over the K frames after the window")
        ax.set_ylabel("density")
        ax.set_title("WM-train episodes" if flag is True else "held-out episodes", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summary_table(records, episodes, cfg):
    """Per episode group: window counts and lengths, resolvability, and how often windows meet a privileged event."""
    rows = []
    tol = cfg.diagnostics.event_tolerance
    for flag, name in ((False, "held-out"), (True, "WM-train")):
        eps = [e for e in episodes if e.wm_train == flag]
        if not eps:
            continue
        ids = {e.episode_id for e in eps}
        rs = [r for r in records if r["episode_id"] in ids]
        scored = [r for r in rs if r["resolvability"] is not None]
        by = {e.episode_id: e for e in eps}
        meets = {}
        for ev in sorted({k for e in eps for k in e.events}):
            hit = [any(r["t_start"] - tol <= t <= r["t_end"] + tol for t in by[r["episode_id"]].events.get(ev, ()))
                   for r in rs]
            meets[f"frac_windows_near_{ev}"] = float(np.mean(hit)) if hit else float("nan")
        rows.append(dict(group=name, n_episodes=len(eps), n_windows=len(rs),
                         mean_len=float(np.mean([r["length"] for r in rs])) if rs else float("nan"),
                         frac_episodes_with_window=len({r["episode_id"] for r in rs}) / len(eps),
                         n_scored=len(scored),
                         median_resolvability=float(np.median([r["resolvability"] for r in scored])) if scored else float("nan"),
                         frac_pct_ge_95=float(np.mean([r["resolvability_pct"] >= 95 for r in scored])) if scored else float("nan"),
                         **meets))
    return pd.DataFrame(rows)
