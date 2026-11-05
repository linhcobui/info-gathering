"""Videos of the error-spike pipeline on whole episodes: what the WM sees, what it predicts, and g's running output.

Layout (one video frame = one model step tau = 5 raw steps):
    real frame tau | WM prediction of frame tau from h = error_horizon steps back (decoded) | text: step, err, g, labels
    strip plot over the whole episode, cursor at tau:
        top     realised WM error err_{tau-h} (error_metric of that prediction of frame tau) and lambda
        bottom  g_{tau-h}: g's probability that frame tau is mispredicted, computed BEFORE frames tau-h+1 .. tau exist
                (from frames tau-h-2 .. tau-h and the logged actions); on / off thresholds; candidate windows shaded
Privileged labels (play segment, probe stage, grasped, true COM offset c, grasp point s) are shown for the viewer only;
neither the WM nor g reads them.

Needs outputs of compute_wm_errors.py, make_labels.py and train_g.py (paths.out_dir).
    pixi run python scripts/seg_video.py --auto 4
    pixi run python scripts/seg_video.py --episodes heldout200/ep012 v3_400/ep033
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from figure_style import HIGHLIGHT, INK, INTERESTING, MUTED, apply  # noqa: E402
from seg.config import load_config, out_dir  # noqa: E402
from seg.episodes import load_episodes  # noqa: E402
from seg.metrics import context_index, to_device  # noqa: E402
from seg.segment import segment_all  # noqa: E402
from seg.wm import DinoWMAdapter  # noqa: E402
from train_dino_wm import to_u8  # noqa: E402

S = 224          # panel size (px)
PLOT_H = 250


@torch.no_grad()
def decoded_predictions(wm, ep, h, batch=32):
    """(T, S, S, 3) uint8: decoder(WM prediction of frame tau, open loop from the context ending at tau - h)."""
    H = wm.history_len
    vis, prop, act = to_device(ep, wm.device)
    out = np.zeros((ep.T, S, S, 3), np.uint8)
    taus = np.arange(H - 1 + h, ep.T)
    for b in range(0, len(taus), batch):
        tau = taus[b:b + batch]
        t = torch.as_tensor(tau - h, device=wm.device)
        idx = context_index(t, H, wm.device)
        aidx = t[:, None] + torch.arange(-H + 1, h, device=wm.device)
        e = wm.rollout(vis[idx], prop[idx], act[aidx])[:, -1]
        with torch.autocast(wm.device.type, dtype=torch.bfloat16):
            out[tau] = to_u8(wm.model.decoder(e))
    return out


def strip_plot(ep, lab, sc, segs, lam, cfg, width, signal_label):
    """Background image of the two-row strip plot and a function tau -> cursor x pixel."""
    apply(font_size=8)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(width / 100, PLOT_H / 100), dpi=100, sharex=True)
    h = cfg.error_horizon
    a0.plot(lab.t + h, lab.err_value, color=INK, lw=0.8)
    a0.axhline(lam, color=MUTED, ls="--", lw=0.7)
    a0.text(1.0, 1.0, f"realised WM error ({cfg.error_metric}, {h}-step open loop); dashed = λ", transform=a0.transAxes, ha="right",
            va="bottom", fontsize=7, color=MUTED)
    a1.plot(sc.t + h, sc.g, color=HIGHLIGHT, lw=1.0)
    a1.axhline(cfg.on_thresh, color=MUTED, ls="--", lw=0.7)
    a1.axhline(cfg.off_thresh, color=MUTED, ls=":", lw=0.7)
    a1.set_ylim(-0.03, 1.03)
    a1.text(1.0, 1.0, f"{signal_label}, before the frame; shaded = candidate windows", transform=a1.transAxes,
            ha="right", va="bottom", fontsize=7, color=MUTED)
    for ax in (a0, a1):
        ax.set_xlim(0, ep.T)
        for s in segs:
            ax.axvspan(s.t_start - 0.5, s.t_end + 0.5, color=HIGHLIGHT, alpha=0.18, lw=0)
        for t in ep.events.get("grasp", ()):
            ax.axvline(t, color=INTERESTING[0], lw=0.6, alpha=0.6)
    a1.set_xlabel("model step τ (blue ticks: grasp onsets, privileged)", fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.17, hspace=0.45)
    fig.canvas.draw()
    bg = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    h = bg.shape[0]
    y0, y1 = h - a0.transData.transform((0, a0.get_ylim()[1]))[1], h - a1.transData.transform((0, a1.get_ylim()[0]))[1]
    x_of = lambda tau: float(a1.transData.transform((tau, 0))[0])
    plt.close(fig)
    return bg, x_of, (int(y0), int(y1))


def label(img, text, colour=(255, 255, 0)):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((4, 3), text, fill=colour)
    return np.asarray(im)


def privileged(z, attrs, raw, ep_c):
    seg = attrs["play_segments"][int(z["segment_type"][raw])]
    stage = attrs["probe_stages"][int(z["probe_stage"][raw])]
    grasped = bool(z["grasped"][raw])
    s = float(z["grasp_offset_s"][raw])
    lines = [f"segment: {seg}" + (f" / {stage}" if seg == "probe" and stage else ""),
             f"grasped: {'yes' if grasped else 'no'}" + (f"  s = {s * 100:+.1f} cm" if grasped else ""),
             f"true COM c = {ep_c * 100:+.1f} cm"]
    if grasped:
        lines.append(f"|s - c| = {abs(s - ep_c) * 100:.1f} cm")
    return lines


def auto_pick(episodes, scores, stores, n):
    """n episodes: mostly held-out TEST-split ones (g never trained on them) rich in probes and slide-ins, one WM-train."""
    test = set(scores.loc[scores.split == "test", "episode_id"])
    rows = []
    for e in episodes:
        if not (e.episode_id in test or e.wm_train):
            continue
        tab = stores[e.store]["episode"]
        rows.append((e.episode_id, e.wm_train, int(tab["n_probes"][e.index]) + 3 * int(tab["n_slide_in_seated"][e.index] > 0)))
    held = sorted([r for r in rows if not r[1]], key=lambda r: -r[2])[:max(n - 1, 1)]
    wmt = sorted([r for r in rows if r[1]], key=lambda r: -r[2])[:1]
    return [r[0] for r in held + wmt][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--auto", type=int, default=4, help="pick this many episodes when --episodes is not given")
    ap.add_argument("--fps", type=int, default=6, help="model steps per second (real time is 4)")
    ap.add_argument("--signal-label", default="g: predicted P(err > λ)", help="name of the g column in the plot")
    ap.add_argument("--tag", default="", help="suffix of the video file names")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    out = out_dir(cfg)
    scores = pd.read_parquet(out / "g_scores.parquet")
    labels = pd.read_parquet(out / "labels.parquet")
    lam = json.loads((out / "threshold.json").read_text())["lambda_"]
    segs = segment_all(scores, cfg)
    episodes = {e.episode_id: e for e in load_episodes(cfg)}
    stores = {s.name: zarr.open_group(str(ROOT / s.zarr), mode="r") for s in cfg.stores}
    ids = args.episodes or auto_pick(list(episodes.values()), scores, stores, args.auto)
    wm = DinoWMAdapter(ROOT / cfg.paths.wm_ckpt, cfg.device, cfg.errors.bf16, cfg.errors.wm_feature_layer)
    vdir = out / "videos"
    vdir.mkdir(exist_ok=True)
    width = 3 * S
    for eid in ids:
        ep = episodes[eid]
        z = stores[ep.store]
        attrs = dict(z.attrs)
        c = float(z["episode"]["com_offset_c"][ep.index])
        lab, sc = labels[labels.episode_id == eid], scores[scores.episode_id == eid]
        split = sc.split.iloc[0]
        esegs = [s for s in segs if s.episode_id == eid]
        bg, x_of, (y0, y1) = strip_plot(ep, lab, sc, esegs, lam, cfg, width, args.signal_label)
        h = cfg.error_horizon
        pred = decoded_predictions(wm, ep, h)
        err_at = dict(zip(lab.t + h, lab.err_value))
        g_at = dict(zip(sc.t + h, sc.g))
        frames = []
        for tau in range(ep.T):
            real = np.asarray(ep.frame_rgb(tau))
            p = label(pred[tau], f"WM prediction from {h} step{'s' if h > 1 else ''} back" if tau in g_at else "(context)")
            in_win = any(s.t_start <= tau <= s.t_end for s in esegs)
            panel = Image.new("RGB", (S, S), "black")
            d = ImageDraw.Draw(panel)
            txt = [f"{eid}  ({'WM-train' if ep.wm_train else 'held-out'}, g split: {split})", f"step tau = {tau} / {ep.T - 1}", ""]
            if tau in g_at:
                txt += [f"signal (predicted) = {g_at[tau]:.2f}", f"err (realised) = {err_at[tau]:.2f}",
                        f"err > lambda: {'YES' if err_at[tau] > lam else 'no'}", "in candidate window" if in_win else ""]
            else:
                txt += [f"(no prediction: needs 3 context frames + {h} steps)", "", "", ""]
            txt += ["", "privileged (WM and g never see):"] + privileged(z, attrs, int(ep.frame_index[tau]), c)
            d.text((6, 6), "\n".join(txt), fill=(255, 255, 255), spacing=3)
            if tau in g_at:
                gx = int(6 + g_at[tau] * (S - 12))
                d.rectangle([6, S - 16, S - 6, S - 6], outline=(120, 120, 120))
                d.rectangle([6, S - 16, gx, S - 6], fill=(255, 149, 0))
            top = np.concatenate([label(real, "real"), p, np.asarray(panel)], 1)
            plot = Image.fromarray(bg.copy())
            x = x_of(tau)
            ImageDraw.Draw(plot).line([(x, y0), (x, y1)], fill=(0, 0, 0), width=2)
            frames.append(np.concatenate([top, np.asarray(plot)[:, :width]], 0))
        path = vdir / f"{eid.replace('/', '_')}_g{args.tag}.mp4"
        imageio.mimwrite(path, frames, fps=args.fps, macro_block_size=8)
        print(f"{path}  ({len(frames)} steps, {len(esegs)} windows, c = {c * 100:+.1f} cm)", flush=True)


if __name__ == "__main__":
    main()
