"""Can the OGBar-SMT prior imagine a whole pickup? Open loop with the RECORDED actions, from cuts before probe 1's grasp.

For each eval episode and each cut (model steps before the pads close: --before), z_cut = teacher on the real
history up to the cut, then z_{k+1} = P(z_k, a_k) and frames from the decoder's first output (compare_from_cut's
prior_rollout), with the episode's real actions, through the grasp and the lift. Frames are drawn with the DINO-WM
pixel decoder. Bar height = mean image row of the bar's red pixels (smaller = higher in the image); "lifted" = more than
15 px above its row at the cut. Prints GT vs imagined bar row every 4 model steps; writes a video per episode and cut.

    pixi run python scripts/prior_pickup_check_ogbar.py
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
from dino_wm_watch_imagine import panel  # noqa: E402
from info_gathering.ogbar_smt_data import OgbarSmtAnchors  # noqa: E402
from train_dino_wm import build_model  # noqa: E402
from world_model.models.smt_ogbar import SmtOgbar  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

AMP = dict(device_type="cuda", dtype=torch.bfloat16)


def bar_row(img):
    """Mean row of the bar's red pixels below the 27-px title strip (nan if none)."""
    im = img[27:].astype(int)
    m = (im[..., 0] > 150) & (im[..., 1] < 110) & (im[..., 2] < 120)
    return float(np.nonzero(m)[0].mean()) + 27 if m.sum() > 30 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smt", default="logdir/world_model/smt_ogbar/2026-10-30_17-37-45_privalign_push/checkpoints/wm_latest.pt")
    ap.add_argument("--dino-wm", default="logdir/world_model/dino_wm_ogbar/v3_400_proprio/checkpoints/wm_latest.pt")
    ap.add_argument("--zarr", default="data/ogbar_v3_eval5.zarr")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--before", type=int, nargs="+", default=[0, 2, 4, 8], help="cut = this many model steps before the grasp step (cut at step 0 at most)")
    ap.add_argument("--after", type=int, default=16, help="model steps imagined past the grasp step")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    st = CheckpointManager.load(args.smt)
    m = SmtOgbar(action_dim=25, **st["config"]["model"]).to(dev).eval()
    m.load_state_dict(st["model"])
    dst = CheckpointManager.load(args.dino_wm)
    dw = build_model(OmegaConf.create(dst["config"]), dst["stats"], 25, 19).to(dev).eval()
    dw.load_state_dict(dst["model"])
    data = OgbarSmtAnchors(ROOT / (args.zarr.removesuffix(".zarr") + "_dinov2"), ROOT / args.zarr, 5)
    z = zarr.open_group(str(ROOT / args.zarr), "r")
    out_dir = Path(args.smt).parents[1] / "prior_pickup"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for e in args.episodes:
        s0, e0 = int(data.starts[e]), int(data.ends[e])
        g = z["grasped"][s0:e0]
        t_grasp = int(np.argmax(g))                                         # raw step the pads first hold the bar
        g_step = t_grasp // 5
        patch, action = data.load(e, 0)
        patch, action = patch.to(dev).to(torch.bfloat16), m.norm_action(action.to(dev))
        real = z["rgb"][s0:e0][::5][:len(patch)]
        bz = z["bar_pos"][s0:e0, 2][::5][:len(patch)]
        with torch.no_grad(), torch.autocast(**AMP):
            pooled = m.pool(patch[None])
        for b in args.before:
            cut = max(0, g_step - b)
            H = g_step + args.after - cut
            with torch.no_grad(), torch.autocast(**AMP):
                z_cut = m.teacher_windows(pooled, action[None], torch.full((1, 1), cut, dtype=torch.long, device=dev))[0, 0]
            im = decode_all(dw, prior_rollout(m, z_cut, patch[cut], action[cut:cut + H]).to(dev))
            gt_rows = np.array([bar_row(real[cut + k]) for k in range(H + 1)])
            im_rows = np.array([gt_rows[0]] + [bar_row(im[k - 1]) for k in range(1, H + 1)])
            gt_lift = np.nanmin(gt_rows) < gt_rows[0] - 15
            im_lift = np.nanmin(im_rows) < im_rows[0] - 15
            summary.append((e, b, cut, g_step, gt_rows[0] - np.nanmin(gt_rows), im_rows[0] - np.nanmin(im_rows), gt_lift, im_lift))
            cols = list(range(0, H + 1, 4))
            print(f"\nep {e}: grasp at model step {g_step}; cut {cut} ({g_step - cut} before); real bar z at the end {bz[cut + H]:.3f} m")
            print("  step      " + "".join(f"{cut + k:>6d}" for k in cols))
            print("  GT row    " + "".join(f"{gt_rows[k]:6.0f}" for k in cols))
            print("  prior row " + "".join(f"{im_rows[k]:6.0f}" for k in cols))
            frames = [np.concatenate([panel(real[cut + k], f"GT ep{e} step {cut + k}", f"{'GRASP ' if cut + k >= g_step else ''}"),
                                      panel(real[cut] if k == 0 else im[k - 1], f"prior from step {cut}", f"+{k}  ")], 1)
                      for k in range(H + 1)]
            imageio.mimwrite(out_dir / f"ep{e}_cut{b}before.mp4", frames, fps=4, macro_block_size=8)
    print("\nep  before  cut  grasp  GT rise(px)  prior rise(px)  GT lifted  prior lifted")
    for r in summary:
        print(f"{r[0]:2d}  {r[1]:6d}  {r[2]:3d}  {r[3]:5d}  {r[4]:11.0f}  {r[5]:14.0f}  {str(r[6]):>9s}  {str(r[7]):>12s}")
    print(f"videos in {out_dir}")


if __name__ == "__main__":
    main()
