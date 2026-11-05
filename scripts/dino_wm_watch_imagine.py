"""Ground truth | open-loop from the start | watch real frames up to a cut, then imagine (DINO-WM).

DINO-WM has no separate posterior and prior: "watching" = its 3-frame history holds REAL frames (shown: its
1-step predictions), "imagining" = open-loop rollout from the last 3 real frames before the cut, driven by the
recorded actions. The cut is the first model step at or after the start of probe `--cut-probe`'s lift.
Every panel carries the bar's tilt measured in that image (line fit to its red pixels).

    pixi run python scripts/dino_wm_watch_imagine.py --ckpt <wm_latest.pt> --zarr data/ogbar_v3_eval5.zarr --episode 3
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
from PIL import Image, ImageDraw  # noqa: E402

from dino_wm_videos import decode_all, one_step, open_loop  # noqa: E402
from info_gathering.ogbar_dino import OgbarDinoWindows  # noqa: E402
from train_dino_wm import build_model  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

LIFT = 1   # play.PROBE_STAGES index of 'lift'


def img_tilt(im):
    """Bar angle in the image (deg, + = image-right end lower) from a line through its red pixels."""
    r, g, b = (im[..., i].astype(int) for i in range(3))
    ys, xs = np.nonzero((r > 150) & (g < 110) & (b < 130))
    if len(xs) < 50:
        return np.nan
    w, v = np.linalg.eigh(np.cov(np.stack([xs - xs.mean(), ys - ys.mean()])))
    d = v[:, -1] if v[0, -1] > 0 else -v[:, -1]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def panel(img, title, sub=""):
    im = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 223, 26), fill=(0, 0, 0))
    d.text((4, 2), title, fill=(255, 255, 0))
    t = img_tilt(img[27:])
    d.text((4, 14), f"{sub}image tilt {t:+.0f} deg" if np.isfinite(t) else sub, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--zarr", default="data/ogbar_v3_eval5.zarr")
    ap.add_argument("--episode", type=int, default=3)
    ap.add_argument("--cut-probe", type=int, default=2, help="imagine from the start of this probe's lift (1-based)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    state = CheckpointManager.load(args.ckpt)
    cfg = OmegaConf.create(state["config"])
    device = torch.device(args.device)
    data = OgbarDinoWindows(ROOT / (args.zarr.removesuffix(".zarr") + "_dinov2"), ROOT / args.zarr,
                            num_frames=cfg.model.num_hist + 1, frameskip=cfg.data.frameskip)
    model = build_model(cfg, state["stats"], data.action_dim, data.proprio_dim).to(device).eval()
    model.load_state_dict(state["model"])
    h, k = model.num_hist, cfg.data.frameskip

    e = args.episode
    z = zarr.open_group(str(ROOT / args.zarr), "r")
    s0, e0 = int(data.episode_starts[e]), int(data.episode_ends[e])
    ps, seg = z["probe_stage"][s0:e0], z["segment_idx"][s0:e0]
    lift_starts = [t for t in range(len(ps)) if ps[t] == LIFT and (t == 0 or ps[t - 1] != LIFT)]
    cut = -(-lift_starts[args.cut_probe - 1] // k)              # first model step at/after that lift's first raw frame
    c_cm = float(z["episode/com_offset_c"][e]) * 100

    seq = data.episode_sequence(e)
    real = seq["rgb"].numpy()
    T = len(real)
    ol = decode_all(model, open_loop(model, seq, device, 10 ** 6)[0])                  # steps h .. T-1
    one = decode_all(model, one_step(model, seq, device))                               # steps h .. T-1
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vis, prop, act = (seq[x].to(device) for x in ("visual", "proprio", "action"))
        v, _ = model.rollout(vis[None, cut - h:cut], prop[None, cut - h:cut], act[None, cut - h:T - 1])
    imag = decode_all(model, v[0, h:].float())                                          # steps cut .. T-1

    frames = []
    for t in range(T):
        stage = ("probe %d" % (seg[t * k] + 1)) + ("" if ps[t * k] == 0 else "")
        gt = panel(real[t], f"ground truth  ep {e}  c {c_cm:+.1f} cm", f"step {t}  {stage}  ")
        a = panel(real[t] if t < h else ol[t - h], "open-loop from start", "real context  " if t < h else f"+{t - h + 1}  ")
        if t < h:
            b = panel(real[t], "watch, then imagine", "real context  ")
        elif t < cut:
            b = panel(one[t - h], "WATCHING (real history)", "1-step pred  ")
        else:
            b = panel(imag[t - cut], f"IMAGINING from step {cut}", f"+{t - cut + 1}  ")
        frames.append(np.concatenate([gt, a, b], axis=1))
    out = Path(args.ckpt).parents[1] / "videos" / f"{Path(args.zarr).stem}_ep{e:03d}_watch{args.cut_probe}_imagine.mp4"
    imageio.mimwrite(out, frames, fps=args.fps, macro_block_size=8)

    # probe holds: middle of each 40-step hold
    holds = [t for t in range(len(ps)) if ps[t] == 2 and (t == 0 or ps[t - 1] != 2)]
    mids = [min((t + 20) // k, T - 1) for t in holds]
    grid = np.concatenate([frames[m] for m in mids], axis=0)
    imageio.imwrite(out.with_suffix(".png"), grid)
    print(f"{out}  ({T} steps, cut at step {cut} = raw {cut * k}, lift of probe {args.cut_probe} starts raw {lift_starts[args.cut_probe - 1]})")
    print(f"probe-hold frames {mids} -> {out.with_suffix('.png')}")
    for p, m in enumerate(mids):
        gt_t = img_tilt(real[m][27:])
        print(f"  probe {p + 1} (step {m}): GT {gt_t:+.0f}  open-loop {img_tilt(ol[m - h][27:]):+.0f}  "
              f"{'watch 1-step' if m < cut else 'imagine'} {img_tilt((one[m - h] if m < cut else imag[m - cut])[27:]):+.0f}  "
              f"(privileged tilt {np.degrees(z['bar_tilt'][s0 + m * k]):+.0f})")


if __name__ == "__main__":
    main()
