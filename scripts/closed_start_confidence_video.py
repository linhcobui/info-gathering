"""When does Naive-U say "ball here" with confidence in a closed-start episode, before anything was seen?

    pixi run python scripts/closed_start_confidence_video.py [--run <ckpt>] [--n_videos 2] [--out figures/closed_start_confidence]

For closed-start TRAINING episodes (384..767 of memtest4_all3u), at every frame t:
  p_teacher(t)    decoder "lift now?" on the teacher memory: the encoder over the window of frames ending at t
  p_posterior(t)  the same question on the posterior chained frame by frame from frame 0 (the eval's memory)
"lift now?" = decoder read-out at lift offset 1 of the future (frame t, rest, rest, rest, lift). It is only a
valid question at rest frames (the canned future starts from cups at rest), so only those are scored.
Before the episode's first lift action nothing about the ball has been shown: the honest answer is 1/3.
Hidden ball slot: scoring and the caption only.
"""
import argparse
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
import torch  # noqa: E402
import zarr  # noqa: E402
from torch.amp import autocast  # noqa: E402

from figure_style import FAINT, HIGHLIGHT, INK, INTERESTING, MUTED, apply  # noqa: E402
from train_mcl_prior_phase1 import load_model  # noqa: E402

CUP = {0: "left (the lifted cup)", 1: "middle", 2: "right"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_all3u/naive/checkpoints/wm_latest.pt")
    ap.add_argument("--zarr", default="data/memtest4_all3u.zarr")
    ap.add_argument("--eps", type=int, nargs=2, default=[384, 768], help="closed-start training episodes")
    ap.add_argument("--n_videos", type=int, default=1)
    ap.add_argument("--pick", choices=["canned", "real_right", "real_wrong"], default="canned")
    ap.add_argument("--episodes", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="figures/closed_start_confidence")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.run, dev)
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    g = zarr.open_group(str(ROOT / args.zarr), mode="r")
    ends = g["episode_ends"][:]; starts = np.r_[0, ends[:-1]]
    HOLD, LIFT = np.array([1., 0.], np.float32), np.array([0., 1.], np.float32)
    F_, dec = int(model.fut_len), model.reward_decoder
    A_q = torch.from_numpy(np.stack([HOLD] * 4 + [LIFT] * (F_ - 4))).to(dev)

    @torch.no_grad()
    def lift_now(z, cur, wait=4):
        """Canned future: the current frame held, the lift action after `wait` frames; read at lift offset 1."""
        A = torch.from_numpy(np.stack([HOLD] * wait + [LIFT] * (F_ - wait))).to(dev)
        X = cur[:, None].expand(-1, F_, -1)
        with autocast(**amp):
            return torch.sigmoid(dec.ball_logit(dec(z, X, A[None].expand(len(z), -1, -1))).float()[:, wait + 1]).cpu().numpy()

    @torch.no_grad()
    def real_read(z, E, act, fl):
        """The decoder's TRAINING input: the real future frames + actions, read at the real first lift."""
        T = len(E); p = np.full(T, np.nan)
        ts = np.array([t for t in range(T) if t <= fl and fl - t + 1 <= F_])
        idx = np.minimum(np.arange(F_)[None] + ts[:, None], T - 1)
        with autocast(**amp):
            out = torch.sigmoid(dec.ball_logit(dec(z[ts], E[torch.from_numpy(idx).to(dev)],
                                                   torch.from_numpy(act[idx]).to(dev))).float()).cpu().numpy()
        p[ts] = out[np.arange(len(ts)), fl - ts + 1]
        return p

    def episode(e):
        s, en = int(starts[e]), int(ends[e])
        act = g["action"][s:en].astype(np.float32)
        E = (torch.from_numpy(np.asarray(g[model.embed_key][s:en], np.float32)).to(dev) - model.embed_mean) / model.embed_std
        PA = np.zeros_like(act); PA[1:] = act[:-1]
        PAt = torch.from_numpy(PA).to(dev)
        first = torch.zeros(1, en - s, dtype=torch.bool, device=dev); first[0, 0] = True
        with torch.no_grad():
            Zt = model.encode_windows(E[None], PAt[None], first)[0].float()
            zs, z = [Zt[0]], Zt[0][None]
            with autocast(**amp):
                for t in range(1, en - s):
                    z = model.posterior.step(z, E[t][None], PAt[t][None]).float()
                    zs.append(z[0])
            Zp = torch.stack(zs)
        still = np.r_[False, (E[1:] - E[:-1]).abs().amax(-1).cpu().numpy() < 1e-6]
        hold = (PA == HOLD).all(1)
        lift = (act == LIFT).all(1)
        fl = int(np.flatnonzero(lift)[0])
        return dict(e=e, pt=lift_now(Zt, E), pp=lift_now(Zp, E), p1=lift_now(Zt, E, wait=1),
                    pr=real_read(Zt, E, act, fl), rest=still & hold, lift=lift, first_lift=fl, slot=g["ball_slot"][s:en].astype(int),
                    rgb=g["rgb"][s:en], final=int(g["ball_under_target"][en - 1]))

    split = "training" if "all3u.zarr" in args.zarr else "test"
    closed = np.flatnonzero(g["episode_closed"][:] == 1)
    ep_ids = closed[(closed >= args.eps[0]) & (closed < args.eps[1])] if split == "training" else closed
    eps = [episode(e) for e in ep_ids]
    before = lambda d: d["rest"] & (np.arange(len(d["pt"])) < d["first_lift"])
    print(f"{len(eps)} closed-start {split} episodes; rest frames before the first lift (nothing seen, honest p = 1/3):")
    for name, k in (("canned: lift after 4 frames, teacher", "pt"), ("canned: lift after 4 frames, posterior", "pp"),
                    ("canned: lift NEXT frame, teacher", "p1"), ("real future frames (training input)", "pr")):
        p = np.concatenate([d[k][before(d)] for d in eps])
        ok = ~np.isnan(p)
        print(f"  {name:40s} {ok.sum()} frames | p > 0.9: {np.mean(p[ok] > 0.9):.3f} | p < 0.1: {np.mean(p[ok] < 0.1):.3f}")

    def score(d):
        m = before(d)
        if args.pick == "canned":       # most confident wrong answer to my old canned query
            m = m & (d["slot"] != 0); return d["pt"][m].max() if m.any() else 0.0
        conf = np.nan_to_num(np.abs(d["pr"] - 0.5))[m]            # most confident real-input read before any lift
        right = ((d["pr"][m] > 0.5) == bool(d["slot"][d["first_lift"]] == 0))
        want = right if args.pick == "real_right" else ~right
        return conf[want].max() if want.any() else 0.0
    picks = ([d for d in eps if d["e"] in args.episodes] if args.episodes
             else sorted(eps, key=score, reverse=True)[:args.n_videos])

    apply(font_size=9)
    for d in picks:
        T = len(d["pt"])
        frames = []
        for t in range(T):
            fig = plt.figure(figsize=(10.4, 4.4), dpi=100)
            ax_img = fig.add_axes([0.02, 0.12, 0.28, 0.70])
            ax = fig.add_axes([0.37, 0.14, 0.60, 0.64])
            ax_img.imshow(np.kron(d["rgb"][t], np.ones((5, 5, 1), np.uint8)), interpolation="nearest")
            ax_img.set_axis_off()
            a = "lift" if d["lift"][t] else "hold" if t > 0 else "start"
            ax_img.set_title(f"frame {t}  ·  action {a}", fontsize=9.5)
            ax.axhspan(0.9, 1.02, color=FAINT, alpha=0.25, lw=0)
            ax.axhline(1 / 3, color=INK, lw=1, ls="--")
            ax.text(T - 0.5, 1 / 3 + 0.02, "honest: 1/3", ha="right", va="bottom", fontsize=8, color=INK)
            for tt in np.flatnonzero(d["lift"]):
                ax.axvspan(tt - 0.5, tt + 0.5, color=INTERESTING[5], alpha=0.35, lw=0)
            rr = np.flatnonzero(d["rest"][:t + 1])
            for k, c, lab in (("pt", INTERESTING[0], "my canned query: hold 4 frames, then lift"),
                              ("p1", INTERESTING[1], "canned query: lift on the NEXT frame"),
                              ("pr", HIGHLIGHT, "training input: the real future frames, read at the real lift")):
                ax.plot(np.arange(t + 1), d[k][:t + 1], color=c, lw=1, alpha=0.35)
                ax.plot(rr, d[k][rr], "o", color=c, ms=4.5, label=lab)
            ax.axvline(t, color=MUTED, lw=0.8)
            ax.set_xlim(-0.5, T - 0.5); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("frame"); ax.set_ylabel('decoder: p("ball under the left cup at the lift")')
            ax.legend(loc="upper left", fontsize=7.5, frameon=False)
            seen = "nothing about the ball shown yet" if t < d["first_lift"] else "a lift has happened"
            ax.set_title(f"closed-start {split} episode {d['e']}  ·  hidden: ball under the {CUP[d['slot'][t]]} cup  ·  {seen}",
                         fontsize=9, loc="left")
            fig.text(0.37, 0.015, "teacher memory (window of frames ending at t) · dots = rest frames · "
                     "purple = lift action · grey band = p > 0.9", fontsize=7.5, color=MUTED)
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if t == T - 1:
                fig.savefig(out / f"{split}_ep{d['e']:03d}_summary.png", dpi=150)
            plt.close(fig)
        path = out / f"{split}_ep{d['e']:03d}.mp4"
        imageio.mimwrite(path, frames, fps=3, codec="libx264", quality=8, macro_block_size=1)
        print(f"-> {path}")

if __name__ == "__main__":
    main()
