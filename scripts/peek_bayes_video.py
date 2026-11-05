"""Closed-start test episodes with one peek: the decoder's answer frame by frame against the exact Bayes answer.

    pixi run python scripts/peek_bayes_video.py [--out figures/peek_bayes]

On every frame t:
  Bayes p*(t)   P(ball under the left cup | everything shown up to t), from the hidden truth (display only):
                1/3 before anything is seen; the peek collapses it to 1 (ball) or 0 (empty, ball in one of the other
                two); swaps then move that set (scripts/belief_labels_phase1.truth_labels); at the final reveal, once the
                cup has visibly risen, p* = what it shows.
  model p(t)    the frozen decoder's "lift on the NEXT frame?" read-out on the posterior memory chained from frame 0
                (teacher only on the first frame, as the eval). Valid at rest frames (dots); mid-swap values are lines.
Picks one episode where the peek shows the ball and one where it is empty and a later swap makes the answer 1/2.
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
import torch  # noqa: E402
import zarr  # noqa: E402
from torch.amp import autocast  # noqa: E402

from belief_eval_phase1 import PSTAR  # noqa: E402
from belief_labels_phase1 import truth_labels  # noqa: E402
from figure_style import FAINT, HIGHLIGHT, INK, INTERESTING, MUTED, apply  # noqa: E402
from info_gathering.swap_sequences import sequences  # noqa: E402
from train_mcl_prior_phase1 import load_model  # noqa: E402

A = "logdir/world_model/smt_phase1_all3u"
RUNS = [("Belief", f"{A}/alignsplice_unifneg/dmt/main", HIGHLIGHT)]          # Align-Walk + label-free DMT
CUP = {0: "left (lifted)", 1: "middle", 2: "right"}
HOLD, LIFT = np.array([1., 0.], np.float32), np.array([0., 1.], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/memtest4_evalu.zarr")
    ap.add_argument("--out", default="figures/peek_bayes")
    ap.add_argument("--episodes", type=int, nargs="*", default=[])
    ap.add_argument("--temp", type=float, default=1.0, help="p = sigmoid(logit / temp); 1 = the trained read-out")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev, out = args.device, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    z = zarr.open_group(str(ROOT / args.zarr), mode="r")
    cfg = json.loads(z.attrs["config"])
    peek_len = 2 * int(cfg["move_frames"]) + int(cfg.get("peek_up_frames", 2))
    ends = z["episode_ends"][:]; st = np.r_[0, ends[:-1]]
    seqs = sequences(ROOT / args.zarr)
    cup_rest = float(z["cup_z_max"][:].min())
    closed, peek, swaps = z["episode_closed"][:], z["episode_peek"][:], z["episode_swapped"][:]

    def pstar(e):
        s, n = int(st[e]), int(ends[e] - st[e])
        return np.array([PSTAR[c] for c in truth_labels(z, int(e), s, n, seqs, cup_rest, peek_len)])

    if args.episodes:
        picks = args.episodes
    else:                                    # one "peek shows the ball", one "empty peek, then 1/2"
        cand = [e for e in range(len(ends)) if closed[e] and peek[e] >= 0 and swaps[e] - peek[e] >= 2]
        ball = next(e for e in cand if pstar(e).max() == 1.0 and len(np.unique(pstar(e))) >= 3)
        half = next(e for e in cand if np.any(np.isclose(pstar(e), 0.5)) and np.isclose(pstar(e)[-1], 0.5))
        picks = [ball, half]
    print("episodes:", picks)
    tag = "" if args.temp == 1.0 else f"_T{args.temp:g}"

    models = [(nm, load_model(f"{r}/checkpoints/wm_latest.pt", dev), c) for nm, r, c in RUNS]
    apply(font_size=9)
    for e in picks:
        s, n = int(st[e]), int(ends[e] - st[e])
        act = z["action"][s:s + n].astype(np.float32)
        PA = np.zeros_like(act); PA[1:] = act[:-1]
        ps, curves = pstar(e), {}
        # the final reveal is evidence too: once the lifted cup has visibly risen, p* = what it shows (1 ball, 0 empty)
        lifts = np.flatnonzero((act == LIFT).all(1))
        final_on = int(lifts[np.r_[True, np.diff(lifts) > 1]][-1])
        up = (np.arange(n) >= final_on) & (z["cup_z_max"][s:s + n] > cup_rest + 1e-3)
        ps = np.where(up, (z["ball_slot"][s:s + n] == 0).astype(float), ps)
        for nm, m, _ in models:
            E = (torch.from_numpy(np.asarray(z[m.embed_key][s:s + n], np.float32)).to(dev) - m.embed_mean) / m.embed_std
            PAt = torch.from_numpy(PA).to(dev)
            amp = dict(device_type=dev.split(":")[0], dtype=m.amp_dtype)
            F_ = int(m.fut_len)
            Aq = torch.from_numpy(np.stack([HOLD] + [LIFT] * (F_ - 1))).to(dev)
            with torch.no_grad():
                f0 = torch.zeros(1, 1, dtype=torch.bool, device=dev); f0[0, 0] = True
                zc = m.encode_windows(E[:1][None], PAt[:1][None], f0)[0, 0][None]
                zs = [zc[0]]
                with autocast(**amp):
                    for t in range(1, n):
                        zc = m.posterior.step(zc, E[t][None], PAt[t][None]).float(); zs.append(zc[0])
                Z = torch.stack(zs)
                with autocast(**amp):
                    p = torch.sigmoid(m.reward_decoder.ball_logit(m.reward_decoder(
                        Z, E[:, None].expand(-1, F_, -1), Aq[None].expand(n, -1, -1))).float()[:, 2]
                        / args.temp).cpu().numpy()
            curves[nm] = p
            still = np.r_[False, (E[1:] - E[:-1]).abs().amax(-1).cpu().numpy() < 1e-6]
        lift = (act == LIFT).all(1)
        rest = still & (PA == HOLD).all(1)
        slot = z["ball_slot"][s:s + n].astype(int)
        rgb = z["rgb"][s:s + n]
        frames = []
        for t in range(n):
            fig = plt.figure(figsize=(10.4, 4.4), dpi=100)
            ax_img = fig.add_axes([0.02, 0.12, 0.28, 0.70]); ax = fig.add_axes([0.37, 0.14, 0.60, 0.64])
            ax_img.imshow(np.kron(rgb[t], np.ones((5, 5, 1), np.uint8)), interpolation="nearest"); ax_img.set_axis_off()
            ax_img.set_title(f"frame {t}  ·  action {'lift' if lift[t] else 'hold' if t else 'start'}", fontsize=9.5)
            for tt in np.flatnonzero(lift):
                ax.axvspan(tt - 0.5, tt + 0.5, color=INTERESTING[5], alpha=0.35, lw=0)
            ax.step(np.arange(t + 1), ps[:t + 1], where="post", color=INK, lw=0.9, label="Bayes p* (exact)")
            rr = np.flatnonzero(rest[:t + 1]); mm = np.flatnonzero(~rest[:t + 1])
            for nm, _, c in models:
                ax.plot(np.arange(t + 1), curves[nm][:t + 1], color=c, lw=1.6, alpha=0.6, zorder=3)
                ax.plot(rr, curves[nm][rr], "o", color=c, ms=5.5, zorder=4, label=f"{nm}: p at rest frames")
                ax.plot(mm, curves[nm][mm], "o", mfc="white", mec=c, mew=1.0, ms=3.5, zorder=4,
                        label=f"{nm}: p mid-motion (question not physical)")
            ax.axvline(t, color=FAINT, lw=0.8)
            ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 1 / 3, 0.5, 1])
            ax.set_yticklabels(["0", "1/3", "1/2", "1"])
            ax.set_xlabel("frame"); ax.set_ylabel('p("ball under the left cup")')
            ax.legend(loc="upper left", fontsize=7.5, frameon=False)
            ax.set_title(f"closed-start test episode {e}  ·  hidden: ball under the {CUP[slot[t]]} cup  ·  Bayes p* = "
                         f"{ps[t]:.2f}  ·  Belief p = {curves[RUNS[0][0]][t]:.2f}" + (f"  ·  T = {args.temp:g}" if args.temp != 1.0 else ""), fontsize=8.5, loc="left")
            fig.text(0.37, 0.015, "start never shown · purple = lift action (the peek, then the final reveal) · "
                     "decoder asked 'lift on the next frame?' on the posterior chained from frame 0",
                     fontsize=7.5, color=MUTED)
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if t == n - 1:
                fig.savefig(out / f"ep{e:03d}{tag}_summary.png", dpi=150)
            plt.close(fig)
        imageio.mimwrite(out / f"ep{e:03d}{tag}.mp4", frames, fps=3, codec="libx264", quality=8, macro_block_size=1)
        m = rest & ~lift
        print(f"-> {out / f'ep{e:03d}{tag}.mp4'}  | at rest frames: mean |p - p*| " + ", ".join(
            f"{nm} {np.abs(curves[nm][m] - ps[m]).mean():.3f}" for nm, _, _ in models)
              + f" | final lift onset p* {ps[np.flatnonzero(lift)[-1] - 9]:.2f}")


if __name__ == "__main__":
    main()
