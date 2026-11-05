"""Videos of the model's confidence on closed-start episodes: the frames next to a live curve of the frozen decoder's
"ball under the lifted cup if we lift now?" (chained posterior) against the honest Bayes answer.

    pixi run python scripts/belief_videos_phase1.py --run logdir/world_model/smt_phase1_all3u/handcrafted \
        --episodes 181 217 173 --out figures/belief_videos

The honest answer comes from the true information seen so far (belief_labels_phase1.truth_labels); it is shown for
comparison only. Slots: 0 = left (the cup that is lifted), 1 = middle, 2 = right.
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
from belief_labels_phase1 import SETS, truth_labels  # noqa: E402
from info_gathering.swap_sequences import sequences  # noqa: E402
from train_mcl_prior_phase1 import load_model  # noqa: E402

SLOT_NAME = {0: "left", 1: "middle", 2: "right"}
BLUE, ORANGE, INK, MUTED, GRID, BG = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_all3u/handcrafted")
    ap.add_argument("--zarr", default="data/memtest4_evalu.zarr")
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--out", default="figures/belief_videos")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(args.run) / "checkpoints" / "wm_latest.pt", dev)
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    dec, F_ = model.reward_decoder, int(model.fut_len)
    key = model.embed_key
    norm = lambda v: (torch.from_numpy(np.asarray(v, np.float32)).to(dev) - model.embed_mean) / model.embed_std
    A_LIFT = torch.tensor([0., 1.], device=dev)

    # "ball if we lift now?": exactly the scored read-out (posterior_r1 in onset_probe_phase1.py / belief_eval_phase1.py),
    # moved to frame t: the decoder sees the memory plus one token (x_t, a = lift) and answers for the next frame.
    # Later tokens cannot reach that read-out (causal), so they are just x_t repeated. At t = onset it IS the eval.
    @torch.no_grad()
    def lift_now(zz, e_now):
        X = e_now[:, None].expand(-1, F_, -1)
        A = A_LIFT.expand(len(zz), F_, -1)
        with autocast(**amp):
            return dec.ball_logit(dec(zz, X, A)).float()[:, 1].cpu().numpy()          # the raw logit

    z = zarr.open_group(str(ROOT / args.zarr), mode="r")
    cfg = json.loads(z.attrs["config"])
    peek_len = 2 * int(cfg["move_frames"]) + int(cfg.get("peek_up_frames", 2))
    ends = z["episode_ends"][:]
    st = np.r_[0, ends[:-1]]
    seqs = sequences(ROOT / args.zarr)
    cup_rest = float(z["cup_z_max"][:].min())

    for e in args.episodes:
        s, n = int(st[e]), int(ends[e] - st[e])
        rgb = z["rgb"][s:s + n]
        ph, act = z["phase"][s:s + n], z["action"][s:s + n].astype(np.float32)
        peek = int(z["episode_peek"][e])
        onset = n - int(cfg["show_frames"])
        label = bool(z["ball_under_target"][s + n - 1])

        # chained posterior (warm-up on frame 0 only), then the query at every frame up to the onset
        E = norm(z[key][s:s + n])[None]
        PA = torch.zeros(1, n, 2, device=dev)
        PA[0, 1:] = torch.from_numpy(act[:n - 1]).to(dev)
        first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
        with torch.no_grad():
            zw = model.encode_windows(E[:, :1], PA[:, :1], first[:, :1])
            zs, zz = [zw[:, 0]], zw[:, 0]
            with autocast(**amp):
                for t in range(1, n):
                    zz = model.posterior.step(zz, E[:, t], PA[:, t]).float()
                    zs.append(zz)
        Z = torch.cat(zs, 0)
        logit = lift_now(Z, E[0])                                                # every frame, the final lift included
        p_model = 1.0 / (1.0 + np.exp(-logit))                                   # probability = sigmoid(logit)
        cls = truth_labels(z, int(e), s, n, seqs, cup_rest, peek_len)
        cz = z["cup_z_max"][s:s + n]
        shown_now = (np.arange(n) > onset) & (cz > cup_rest + 1e-3)           # the final lift has revealed the answer
        p_true = np.array([float(label) if shown_now[t] else PSTAR[cls[t]] for t in range(n)])
        # resting frames (identical to the previous one): the only kind the decoder was ever asked this question at
        rest = np.r_[True, (rgb[1:] == rgb[:-1]).reshape(n - 1, -1).all(1)]

        # what is happening at each frame
        start, pairs = seqs[e]
        h0 = int(np.flatnonzero(ph == 1)[0])
        swap_start = [h0 + 4 + 6 * j + (peek_len if 0 <= peek <= j else 0) for j in range(len(pairs))]
        lifts = np.flatnonzero(ph == 2)
        peek_frames = [t for t in lifts if t < onset]

        def what(t):
            if t < h0 + 4:
                return "start: cups already down, ball never shown"
            for j, t0 in enumerate(swap_start):
                if t0 <= t < t0 + 6:
                    a, b = pairs[j]
                    return f"swap {j + 1}/{len(pairs)}: {SLOT_NAME[a]} <-> {SLOT_NAME[b]}"
            if peek_frames and peek_frames[0] <= t < peek_frames[0] + peek_len:
                seen = "BALL" if z["ball_slot"][s + peek_frames[0] + 1] == 0 else "EMPTY"
                return f"peek: left cup lifted -> {seen}"
            if t >= onset:
                return f"final lift: {'BALL' if label else 'EMPTY'}"
            return "waiting"

        def belief_words(c):
            return "ball under " + " / ".join(SLOT_NAME[k] for k in sorted(SETS[c]))

        frames_out = []
        tail = int(round(args.fps * 2))                      # hold the last frame ~2 s
        for t in list(range(n)) + [n - 1] * tail:
            fig = plt.figure(figsize=(10, 4.2), facecolor=BG, dpi=100)
            ax_im = fig.add_axes([0.02, 0.08, 0.36, 0.78])
            ax = fig.add_axes([0.46, 0.18, 0.51, 0.62], facecolor=BG)
            ax_im.imshow(rgb[t], interpolation="nearest"); ax_im.axis("off")
            ax_im.set_title(what(t), fontsize=10.5, color=INK, loc="left")
            for j, t0 in enumerate(swap_start):
                ax.axvspan(t0 - .5, t0 + 5.5, color=GRID if j % 2 == 0 else "#d6d5cf", lw=0, zorder=0)
            if peek_frames:
                ax.axvspan(peek_frames[0] - .5, peek_frames[0] + peek_len - .5, color="#fbe3d6", lw=0, zorder=0)
            ax.axvspan(onset - .5, n - .5, color="#dbe8f8", lw=0, zorder=0)
            ax.axhline(1 / 3, color=MUTED, lw=.8, ls=":", zorder=1)
            ax.axhline(.5, color=MUTED, lw=.8, ls=":", zorder=1)
            k = t
            ax.plot(np.arange(k + 1), p_true[:k + 1], color=ORANGE, lw=2, ls=(0, (4, 2)), zorder=3)
            ax.plot(np.arange(k + 1), p_model[:k + 1], color=BLUE, lw=1.2, alpha=.35, zorder=4)
            rk = np.flatnonzero(rest[:k + 1])
            ax.plot(rk, p_model[rk], "o", color=BLUE, ms=3.5, zorder=5)
            ax.plot([k], [p_model[k]], "o", color=BLUE, ms=7, mfc=BG if not rest[k] else BLUE, mew=2, zorder=6)
            ax.set_xlim(-.5, n - .5); ax.set_ylim(-.04, 1.04)
            ax.set_yticks([0, 1 / 3, .5, 1]); ax.set_yticklabels(["0", "1/3", "1/2", "1"])
            ax.set_xlabel("frame", color=MUTED, fontsize=9)
            ax.tick_params(colors=MUTED, labelsize=8.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.set_title("p(ball under the LEFT cup if we lift now)", fontsize=10, color=INK, loc="left")
            ax.text(0.0, 1.13, f"model {p_model[k]:.2f} (logit {logit[k]:+.2f})", transform=ax.transAxes, color=BLUE,
                    fontsize=11, weight="bold")
            hw = f"revealed: {'ball' if label else 'empty'}" if shown_now[k] else belief_words(cls[k])
            ax.text(0.44, 1.13, f"honest {p_true[k]:.2f}  ({hw})", transform=ax.transAxes, color=ORANGE, fontsize=10)
            if t >= onset:
                ok = abs(p_model[onset] - p_true[onset]) <= 0.1
                fig.text(0.02, 0.015, f"lift onset (before the cup moves): model {p_model[onset]:.2f} vs honest "
                         f"{p_true[onset]:.2f} -> {'PASS' if ok else 'FAIL'}   ·   after the reveal: model {p_model[k]:.2f} "
                         f"(revealed: {'ball' if label else 'empty'})", color=INK, fontsize=9.5, weight="bold")
            fig.text(0.46, 0.93, "● model at resting frames (faint: cups moving)", color=BLUE, fontsize=8.5)
            fig.text(0.76, 0.93, "- - honest Bayes answer", color=ORANGE, fontsize=8.5)
            if t < onset:
                fig.text(0.46, 0.03, "grey: swaps   orange: peek   blue: final lift", color=MUTED, fontsize=8.5)
            fig.canvas.draw()
            frames_out.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
        tag = {3: "nothing_seen", 5: "half", 6: "half", 4: "other_two"}.get(cls[onset], "certain")
        path = out / f"ep{e:03d}_{len(pairs)}swaps_{tag}.mp4"
        imageio.mimwrite(path, frames_out, fps=args.fps, quality=8, macro_block_size=1)
        sheet = np.concatenate([frames_out[i] for i in np.linspace(0, n - 1, 4).astype(int)], 0)
        imageio.imwrite(out / f"ep{e:03d}_sheet.png", sheet)
        print(f"{path}  ({n} frames, model {p_model[onset]:.2f} vs honest {p_true[onset]:.2f} at the lift)")
        rest_t = np.flatnonzero(rest)
        print("   resting frames  t:     " + " ".join(f"{t:6d}" for t in rest_t))
        print("                   logit: " + " ".join(f"{logit[t]:+6.2f}" for t in rest_t))
        print("                   p:     " + " ".join(f"{p_model[t]:6.2f}" for t in rest_t))
        for nm, tt in (("peek", peek_frames), ("final lift", list(range(onset, n)))):
            if len(tt):
                print(f"   {nm:10s} frames {int(tt[0])}..{int(tt[-1])}  logit: " + " ".join(f"{logit[t]:+6.2f}" for t in tt)
                      + "\n" + " " * 30 + "p:     " + " ".join(f"{p_model[t]:6.2f}" for t in tt))


if __name__ == "__main__":
    main()
