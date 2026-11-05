"""Imagination from one cut, side by side: ground truth | SMT no memory | SMT posterior / teacher then prior | DINO-WM.

All panels show the real frame at the cut and predict every later step from the recorded actions alone:
  SMT prior from cut        z_cut = teacher on a window holding ONLY x_cut (nothing before it is seen)
  SMT teacher then prior    z_cut = teacher on the whole history up to the cut (the training target z*_cut)
  SMT posterior then prior  z_0 = teacher on x_0; z_{t+1} = sample of Q(z_t, x_t, a_t, real x_{t+1}) up to the cut
  both SMT panels then imagine with the PRIOR, as docs/OGBAR_SMT.md's eval loop:
      z_{t+1} = sample of P(z_t, x_t, a_t)        x_{t+1} = frame read-out from (z_t, x_t, a_t)
  The read-out R (stage 2) is not built; the stand-in is the stage-1 decoder's first output (z_t, one token
  (x_t, a_t) -> x_{t+1}), so the frame generator is Markov and z is the only memory.
  DINO-WM                   open-loop rollout with its 3-frame history ending at the cut (real frames).
Frames are rendered with the DINO-WM pixel decoder. The cut is the start of probe `--cut-probe`'s lift.

    pixi run python scripts/compare_from_cut.py --episode 3 --cut-probe 2
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from dino_wm_videos import decode_all  # noqa: E402
from dino_wm_watch_imagine import img_tilt, panel  # noqa: E402
from smt_ogbar_watch_imagine import decoder_rollout  # noqa: E402
from info_gathering.ogbar_dino import OgbarDinoWindows  # noqa: E402
from info_gathering.ogbar_smt_data import OgbarSmtAnchors  # noqa: E402
from train_dino_wm import build_model  # noqa: E402
from world_model.models.smt_ogbar import SmtOgbar  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

LIFT, HOLD = 1, 2   # play.PROBE_STAGES indices
AMP = dict(device_type="cuda", dtype=torch.bfloat16)


@torch.no_grad()
def readout(m, z, x, a):
    """Stand-in for R: the decoder's first output from z_t and the token (x_t, a_t) -> x_{t+1}."""
    with torch.autocast(**AMP):
        tok = m.dec_step_in(torch.cat([m.pool(x[None]), a[None].to(x.dtype)], -1))
        return m.head(m.decoder(torch.cat([z[None], tok[:, None]], 1))[:, m.K])[0]


@torch.no_grad()
def prior_rollout(m, z, x, actions):
    """From z_cut and the real x_cut: frames x_cut+1 .. for each action a_cut .. (P chains z, readout makes x)."""
    out = []
    for a in actions:
        x_next = readout(m, z, x, a)
        with torch.autocast(**AMP):
            mu, ls = m.prior(z[None], x[None], a[None])
            z = m.sample(mu, ls)[0]
        x = x_next
        out.append(x.float())
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smt", default="logdir/world_model/smt_ogbar/v3_400_gaussian/checkpoints/wm_latest.pt")
    ap.add_argument("--dino-wm", default="logdir/world_model/dino_wm_ogbar/v3_400_proprio/checkpoints/wm_latest.pt")
    ap.add_argument("--zarr", default="data/ogbar_v3_eval5.zarr")
    ap.add_argument("--episode", type=int, default=3)
    ap.add_argument("--cut-probe", type=int, default=2)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generator", choices=["prior", "decoder"], default="prior",
                    help="prior: P chains z, the decoder's first output makes each frame (Markov). decoder: z stays at "
                         "its cut value and the stage-1 decoder rolls out autoregressively on its own predicted tokens "
                         "with the recorded actions (no P)")
    ap.add_argument("--out", default="logdir/world_model/smt_ogbar/v3_400_gaussian/videos")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    st = CheckpointManager.load(args.smt)
    m = SmtOgbar(action_dim=25, **st["config"]["model"]).to(dev).eval()
    m.load_state_dict(st["model"])
    dst = CheckpointManager.load(args.dino_wm)
    dcfg = OmegaConf.create(dst["config"])
    dw = build_model(dcfg, dst["stats"], 25, 19).to(dev).eval()
    dw.load_state_dict(dst["model"])

    cache = ROOT / (args.zarr.removesuffix(".zarr") + "_dinov2")
    data = OgbarSmtAnchors(cache, ROOT / args.zarr, 5)
    ddata = OgbarDinoWindows(cache, ROOT / args.zarr, num_frames=dw.num_hist + 1, frameskip=5)
    z = zarr.open_group(str(ROOT / args.zarr), "r")
    e = args.episode
    s0, e0 = int(data.starts[e]), int(data.ends[e])
    ps = z["probe_stage"][s0:e0]
    lift_starts = [t for t in range(len(ps)) if ps[t] == LIFT and (t == 0 or ps[t - 1] != LIFT)]
    cut = (lift_starts[args.cut_probe - 1] - 1) // 5          # last model step BEFORE the lift starts: no tilt in x_cut
    c_cm = float(z["episode/com_offset_c"][e]) * 100
    patch, action = data.load(e, 0)
    patch, action = patch.to(dev).to(torch.bfloat16), m.norm_action(action.to(dev))   # model's action scaling
    T = len(patch)
    real = z["rgb"][s0:e0][::5][:T]
    acts = action[cut:T - 1]                                       # a_cut .. a_{T-2} -> x_cut+1 .. x_{T-1}

    with torch.no_grad(), torch.autocast(**AMP):
        pooled = m.pool(patch[None])
        # no memory: the teacher's usual 200-step window ending at the cut, with every step but x_cut masked out
        W = m.cfg["window"]
        idx = torch.arange(cut - W + 1, cut + 1, device=dev)
        prev = torch.cat([torch.zeros_like(action[:1]), action[:-1]], 0)              # a_{s-1}, as the teacher pairs them
        z_blind = m.teacher.tokens(pooled[:, idx.clamp(min=0)], prev[idx.clamp(min=0)][None].to(pooled.dtype),
                                   (idx == cut)[None])[0]
        z_teacher = m.teacher_windows(pooled, action[None], torch.full((1, 1), cut, dtype=torch.long, device=dev))[0, 0]
        zc = m.teacher_windows(pooled, action[None], torch.zeros(1, 1, dtype=torch.long, device=dev))[0, 0]
        for t in range(cut):                                       # posterior chain over the real frames
            mu, ls = m.posterior(zc[None], patch[t][None], action[t][None], patch[t + 1][None])
            zc = m.sample(mu, ls)[0]
    if args.generator == "decoder":
        H = min(len(acts), m.cfg["future"])
        roll = lambda zz: decoder_rollout(m, zz.to(torch.bfloat16), patch[cut], acts[:H])
    else:
        roll = lambda zz: prior_rollout(m, zz, patch[cut], acts)
    blind, post, teach = roll(z_blind), roll(zc), roll(z_teacher)

    seq = ddata.episode_sequence(e, load_rgb=False)
    h = dw.num_hist
    with torch.no_grad(), torch.autocast(**AMP):
        v, _ = dw.rollout(seq["visual"][None, cut - h + 1:cut + 1].to(dev), seq["proprio"][None, cut - h + 1:cut + 1].to(dev),
                          seq["action"][None, cut - h + 1:T - 1].to(dev))
    dino = v[0, h:].float()                                        # x_cut+1 .. x_{T-1}

    dec = lambda x: decode_all(dw, x.to(dev))
    imgs = [dec(blind), dec(post), dec(teach), dec(dino)]
    g = "prior" if args.generator == "prior" else "decoder AR"
    titles = [f"SMT no memory -> {g}", f"SMT posterior -> {g}", f"SMT teacher to cut -> {g}", "DINO-WM from cut"]
    frames = []
    for t in range(T):
        row = [panel(real[t], f"ground truth  ep {e}  c {c_cm:+.1f} cm", f"step {t}  ")]
        for im, title in zip(imgs, titles):
            row.append(panel(real[t], title, "real  ") if t <= cut else
                       panel(im[t - cut - 1], title, f"+{t - cut}  ") if t - cut - 1 < len(im) else
                       panel(np.zeros_like(real[t]), title, "past the decoder's 200 steps"))
        frames.append(np.concatenate(row, axis=1))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    out = out / f"{Path(args.zarr).stem}_ep{e:03d}_from_probe{args.cut_probe}_{args.generator}_smt_vs_dinowm.mp4"
    imageio.mimwrite(out, frames, fps=args.fps, macro_block_size=8)
    holds = [t for t in range(len(ps)) if ps[t] == HOLD and (t == 0 or ps[t - 1] != HOLD)]
    mids = [min((t + 20) // 5, T - 1) for t in holds if (t + 20) // 5 > cut]
    imageio.imwrite(out.with_suffix(".png"), np.concatenate([frames[i] for i in mids], 0))
    print(f"{out}  ({T} steps, cut at step {cut}; SMT checkpoint step {st['step']})")
    print(f"probe holds after the cut (image tilt, deg), generator {args.generator}:  GT | SMT no memory | SMT posterior | SMT teacher | DINO-WM")
    for i in mids:
        k = i - cut - 1
        print(f"  step {i}: {img_tilt(real[i][27:]):+4.0f} | " + " | ".join(f"{img_tilt(im[k][27:]) if k < len(im) else np.nan:+4.0f}" for im in imgs)
              + f"   (privileged {np.degrees(z['bar_tilt'][s0 + i * 5]):+.0f})")
    zd = float((z_blind.float() - zc.float()).pow(2).mean())
    zt = float((z_teacher.float() - zc.float()).pow(2).mean())
    print(f"belief at the cut, mean squared difference: no-memory vs posterior chain {zd:.2e}; teacher vs posterior chain {zt:.2e}")


if __name__ == "__main__":
    main()
