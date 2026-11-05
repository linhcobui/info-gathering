"""Every probe of the given episodes: tilt at the hold, imagined from the moment of grasp.

For each probe: the cut is the last model step before the pads close. From there, with the recorded actions:
  SMT teacher    z_cut = teacher on the whole history up to the cut, then the prior (frames from the decoder's
                 first output, the stand-in for R)
  SMT no memory  the same from a teacher window holding only x_cut
  DINO-WM        open-loop from its 3 real frames ending at the cut
Score: the bar's angle in the side-view image (line through its red pixels), averaged over the model steps where the
real bar is in the probe's 'hold' stage (privileged timing, scoring only). Probes are flagged side-on when the bar's
yaw is within 30 deg of the camera's image plane (privileged, scoring only): only there is the image angle the tilt.

    pixi run python scripts/probe_hold_eval.py --smt <ckpt> --episodes-file logdir/world_model/smt_ogbar/overfit32_episodes.txt
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from compare_from_cut import prior_rollout  # noqa: E402
from dino_wm_videos import decode_all  # noqa: E402
from dino_wm_watch_imagine import img_tilt  # noqa: E402
from info_gathering.ogbar_dino import OgbarDinoWindows  # noqa: E402
from info_gathering.ogbar_smt_data import OgbarSmtAnchors  # noqa: E402
from train_dino_wm import build_model  # noqa: E402
from world_model.models.smt_ogbar import SmtOgbar  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

AMP = dict(device_type="cuda", dtype=torch.bfloat16)
PROBE, HOLD = 0, 2   # play.SEGMENTS 'probe', play.PROBE_STAGES 'hold'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smt", required=True)
    ap.add_argument("--dino-wm", default="logdir/world_model/dino_wm_ogbar/v3_400_proprio/checkpoints/wm_latest.pt")
    ap.add_argument("--zarr", default="data/ogbar_v3_400.zarr")
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--episodes-file")
    ap.add_argument("--horizon", type=int, default=34)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=None, help="csv; default <run>/probe_hold_eval_<zarr stem>.csv")
    args = ap.parse_args()
    eps = args.episodes or [int(e) for e in open(ROOT / args.episodes_file).read().strip().split(",")]
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
    seg_ep, seg_ty = z["segment/episode"][:], z["segment/type"][:]
    seg_t0, seg_t1, seg_s = z["segment/t_start"][:], z["segment/t_end"][:], z["segment/s_last"][:]
    tilt = lambda im: img_tilt(im[27:])
    rows = []
    for e in eps:
        s0, e0 = int(data.starts[e]), int(data.ends[e])
        patch, action = data.load(e, 0)
        patch, action = patch.to(dev).to(torch.bfloat16), m.norm_action(action.to(dev))
        T = len(patch)
        real = z["rgb"][s0:e0][::5][:T]
        grasped, ps = z["grasped"][s0:e0], z["probe_stage"][s0:e0]
        yaw = z["bar_yaw_mod_pi"][s0:e0]
        seq = ddata.episode_sequence(e, load_rgb=False)
        with torch.no_grad(), torch.autocast(**AMP):
            pooled = m.pool(patch[None])
        prev = torch.cat([torch.zeros_like(action[:1]), action[:-1]], 0)
        for k, si in enumerate(np.where((seg_ep == e) & (seg_ty == PROBE))[0]):
            t0, t1 = int(seg_t0[si]), int(seg_t1[si])
            if not grasped[t0:t1].any():
                continue
            t_grasp = t0 + int(np.argmax(grasped[t0:t1]))
            cut = (t_grasp - 1) // 5
            hold = [t for t in range(t_grasp, t1) if ps[t] == HOLD]
            if cut < dw.num_hist or not hold:
                continue
            steps = sorted({t // 5 for t in hold if t // 5 > cut and t // 5 < T})
            H = min(args.horizon, T - 1 - cut)
            steps = [s for s in steps if s - cut <= H]
            if not steps:
                continue
            with torch.no_grad(), torch.autocast(**AMP):
                z_t = m.teacher_windows(pooled, action[None], torch.full((1, 1), cut, dtype=torch.long, device=dev))[0, 0]
                W = m.cfg["window"]
                idx = torch.arange(cut - W + 1, cut + 1, device=dev)
                z_b = m.teacher.tokens(pooled[:, idx.clamp(min=0)], prev[idx.clamp(min=0)][None].to(pooled.dtype),
                                       (idx == cut)[None])[0]
                h = dw.num_hist
                v, _ = dw.rollout(seq["visual"][None, cut - h + 1:cut + 1].to(dev), seq["proprio"][None, cut - h + 1:cut + 1].to(dev),
                                  seq["action"][None, cut - h + 1:cut + H].to(dev))
            ims = {"teacher": decode_all(dw, prior_rollout(m, z_t, patch[cut], action[cut:cut + H]).to(dev)),
                   "nomem": decode_all(dw, prior_rollout(m, z_b, patch[cut], action[cut:cut + H]).to(dev)),
                   "dino": decode_all(dw, v[0, h:h + H].float().to(dev))}
            r = dict(episode=e, probe=k + 1, s_cm=float(seg_s[si]) * 100, c_cm=float(z["episode/com_offset_c"][e]) * 100,
                     priv_tilt=float(np.degrees(np.mean([abs(z["bar_tilt"][s0 + t]) for t in hold]))),
                     side_on=bool(np.mean([min(abs(np.degrees(yaw[t])), abs(180 - np.degrees(yaw[t]))) for t in hold]) < 30),
                     gt=float(np.nanmean([tilt(real[s]) for s in steps])))
            for name, im in ims.items():
                r[name] = float(np.nanmean([tilt(im[s - cut - 1]) for s in steps]))
            rows.append(r)
            print(f"ep {e:3d} probe {k + 1}  s {r['s_cm']:+5.1f}  c {r['c_cm']:+5.1f}  side-on {str(r['side_on']):5s}  hold tilt: "
                  f"GT {r['gt']:+6.1f} | teacher {r['teacher']:+6.1f} | no memory {r['nomem']:+6.1f} | DINO-WM {r['dino']:+6.1f}", flush=True)
    out = Path(args.out) if args.out else Path(args.smt).parents[1] / f"probe_hold_eval_{Path(args.zarr).stem}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} probes -> {out}")
    for subset, sel in (("side-on", [r for r in rows if r["side_on"]]), ("all", rows)):
        sel = [r for r in sel if np.isfinite(r["gt"])]
        if not sel:
            continue
        print(f"{subset} probes (n={len(sel)}): mean |error| deg / sign right when |GT| > 10 deg")
        for name in ("teacher", "nomem", "dino"):
            err = np.array([abs(r[name] - r["gt"]) for r in sel if np.isfinite(r[name])])
            big = [r for r in sel if abs(r["gt"]) > 10 and np.isfinite(r[name])]
            sign = np.mean([np.sign(r[name]) == np.sign(r["gt"]) for r in big]) if big else np.nan
            print(f"  {name:8s} |err| {err.mean():5.1f}  median {np.median(err):5.1f}   sign right {sign:.2f} (n={len(big)})")
        better = np.mean([abs(r["teacher"] - r["gt"]) < abs(r["nomem"] - r["gt"]) for r in sel])
        print(f"  teacher closer to GT than no memory in {better:.0%} of probes")


if __name__ == "__main__":
    main()
