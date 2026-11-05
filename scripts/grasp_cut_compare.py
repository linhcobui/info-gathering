"""From the moment of grasp: ground truth vs OGBar-SMT (teacher to the cut, then prior) vs DINO-WM.

The cut is the last model step before the pads close on the bar for probe `--probe` (the first frame of that probe's
segment where both pads touch the bar). Nothing has been lifted yet, so the coming tilt is not in the cut frame.
  SMT      z_cut = teacher on the whole history up to the cut; then z_{t+1} = sample of P(z_t, x_t, a_t), and each frame
           from the stage-1 decoder's first output (stand-in for R, which is not built)
  DINO-WM  open-loop rollout from its 3 real frames ending at the cut
Both are driven by the recorded actions. Tilt = the bar's angle in the side-view image (line through its red pixels).
Writes a table, a frame strip and a 3-panel video per episode.

    pixi run python scripts/grasp_cut_compare.py --zarr data/ogbar_v3_400.zarr --episodes 156 198
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

from compare_from_cut import prior_rollout  # noqa: E402
from dino_wm_videos import decode_all  # noqa: E402
from dino_wm_watch_imagine import img_tilt, panel  # noqa: E402
from info_gathering.ogbar_dino import OgbarDinoWindows  # noqa: E402
from info_gathering.ogbar_smt_data import OgbarSmtAnchors  # noqa: E402
from train_dino_wm import build_model  # noqa: E402
from world_model.models.smt_ogbar import SmtOgbar  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

AMP = dict(device_type="cuda", dtype=torch.bfloat16)
SEGMENTS = ("probe", "slide_in", "slide_out", "smash", "free", "clear")   # play.SEGMENTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smt", default="logdir/world_model/smt_ogbar/v3_400_unif0.1/checkpoints/wm_latest.pt")
    ap.add_argument("--dino-wm", default="logdir/world_model/dino_wm_ogbar/v3_400_proprio/checkpoints/wm_latest.pt")
    ap.add_argument("--zarr", default="data/ogbar_v3_400.zarr")
    ap.add_argument("--episodes", type=int, nargs="+", default=[156, 198])
    ap.add_argument("--probe", type=int, default=2, help="which segment of that type in the episode (1-based)")
    ap.add_argument("--segment", default="probe", choices=["probe", "smash", "free", "slide_in"],
                    help="pickup segment type to cut at (default probe)")
    ap.add_argument("--horizon", type=int, default=32, help="model steps to imagine after the cut")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)
    st = CheckpointManager.load(args.smt)
    m = SmtOgbar(action_dim=25, **st["config"]["model"]).to(dev).eval()
    m.load_state_dict(st["model"])
    dst = CheckpointManager.load(args.dino_wm)
    dw = build_model(OmegaConf.create(dst["config"]), dst["stats"], 25, 19).to(dev).eval()
    dw.load_state_dict(dst["model"])
    cache = ROOT / (args.zarr.removesuffix(".zarr") + "_dinov2")
    data = OgbarSmtAnchors(cache, ROOT / args.zarr, 5)
    ddata = OgbarDinoWindows(cache, ROOT / args.zarr, num_frames=dw.num_hist + 1, frameskip=5)
    z = zarr.open_group(str(ROOT / args.zarr), "r")
    out_dir = Path(args.smt).parents[1] / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    tilt = lambda im: img_tilt(im[27:])
    print(f"SMT checkpoint step {st['step']}; image tilt (deg); columns = model steps after the cut (x5 raw steps)")
    for e in args.episodes:
        s0, e0 = int(data.starts[e]), int(data.ends[e])
        seg_i = np.where((z["segment/episode"][:] == e) & (z["segment/type"][:] == SEGMENTS.index(args.segment)))[0][args.probe - 1]
        t0, t1 = int(z["segment/t_start"][seg_i]), int(z["segment/t_end"][seg_i])
        grasped = z["grasped"][s0 + t0:s0 + t1]
        t_grasp = t0 + int(np.argmax(grasped))
        bz = z["bar_pos"][s0 + t_grasp:s0 + t1, 2]
        lift = t_grasp + int(np.argmax(bz > bz[0] + 0.01))                 # bar 1 cm above where it was gripped
        cut = (t_grasp - 1) // 5                                 # last model step before the pads close
        patch, action = data.load(e, 0)
        patch, action = patch.to(dev).to(torch.bfloat16), m.norm_action(action.to(dev))
        T = len(patch)
        H = min(args.horizon, T - 1 - cut)
        real = z["rgb"][s0:e0][::5][:T]
        c_cm = float(z["episode/com_offset_c"][e]) * 100
        with torch.no_grad(), torch.autocast(**AMP):
            z_teacher = m.teacher_windows(m.pool(patch[None]), action[None],
                                          torch.full((1, 1), cut, dtype=torch.long, device=dev))[0, 0]
            W = m.cfg["window"]
            idx = torch.arange(cut - W + 1, cut + 1, device=dev)
            prev = torch.cat([torch.zeros_like(action[:1]), action[:-1]], 0)
            pooled = m.pool(patch[None])
            z_blind = m.teacher.tokens(pooled[:, idx.clamp(min=0)], prev[idx.clamp(min=0)][None].to(pooled.dtype),
                                       (idx == cut)[None])[0]                    # teacher sees only x_cut
            z_upd = m.teacher_windows(pooled, action[None], torch.zeros(1, 1, dtype=torch.long, device=dev))[0, 0]
            for t in range(cut):                                   # updater: posterior Q chain over the real frames
                mu, ls = m.posterior(z_upd[None], patch[t][None], action[t][None], patch[t + 1][None])
                z_upd = m.sample(mu, ls)[0]
        smt = decode_all(dw, prior_rollout(m, z_teacher, patch[cut], action[cut:cut + H]).to(dev))
        smt0 = decode_all(dw, prior_rollout(m, z_blind, patch[cut], action[cut:cut + H]).to(dev))
        smtu = decode_all(dw, prior_rollout(m, z_upd, patch[cut], action[cut:cut + H]).to(dev))
        seq = ddata.episode_sequence(e, load_rgb=False)
        h = dw.num_hist
        with torch.no_grad(), torch.autocast(**AMP):
            v, _ = dw.rollout(seq["visual"][None, cut - h + 1:cut + 1].to(dev), seq["proprio"][None, cut - h + 1:cut + 1].to(dev),
                              seq["action"][None, cut - h + 1:cut + H].to(dev))
        dino = decode_all(dw, v[0, h:h + H].float().to(dev))
        cols = [k for k in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40) if k <= H]
        print(f"\nep {e} (c {c_cm:+.1f} cm), {args.segment} {args.probe} (raw {t0}-{t1}, grasp s {float(z['segment/s_last'][seg_i]) * 100:+.1f} cm): "
              f"pads close at raw {t_grasp}, bar 1 cm up at raw {lift}; "
              f"cut = model step {cut} (raw {cut * 5}); lift starts {(lift - cut * 5) / 5:.1f} model steps after the cut")
        print(f"  {'':14s}" + "".join(f"{'+' + str(k):>7s}" for k in cols))
        print(f"  {'GT privileged':14s}" + "".join(f"{np.degrees(z['bar_tilt'][s0 + (cut + k) * 5]):+7.1f}" for k in cols))
        print(f"  {'GT image':14s}" + "".join(f"{tilt(real[cut + k]):+7.1f}" for k in cols))
        for name, im in (("SMT updater", smtu), ("SMT teacher", smt), ("SMT no memory", smt0), ("DINO-WM", dino)):
            print(f"  {name:14s}" + "".join(f"{(tilt(real[cut]) if k == 0 else tilt(im[k - 1])):+7.1f}" for k in cols))
        strip_cols = [k for k in (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40) if k <= H]
        rows = []
        for name, im in (("ground truth", None), ("SMT updater->prior", smtu), ("SMT teacher->prior", smt), ("DINO-WM", dino)):
            rows.append(np.concatenate([panel(real[cut + k] if (im is None or k == 0) else im[k - 1], name, f"+{k}  ")
                                        for k in strip_cols], 1))
        stem = f"{Path(args.zarr).stem}_ep{e:03d}_{args.segment}{args.probe}_from_grasp"
        imageio.imwrite(out_dir / f"{stem}.png", np.concatenate(rows, 0))
        frames = [np.concatenate([panel(real[cut + k], f"ground truth ep {e} c {c_cm:+.1f}", f"+{k}  "),
                                  panel(real[cut] if k == 0 else smtu[k - 1], "SMT updater -> prior", f"+{k}  "),
                                  panel(real[cut] if k == 0 else smt[k - 1], "SMT teacher -> prior", f"+{k}  "),
                                  panel(real[cut] if k == 0 else dino[k - 1], "DINO-WM", f"+{k}  ")], 1) for k in range(H + 1)]
        imageio.mimwrite(out_dir / f"{stem}.mp4", frames, fps=4, macro_block_size=8)
        print(f"  strip {out_dir / (stem + '.png')}   video {out_dir / (stem + '.mp4')}")


if __name__ == "__main__":
    main()
