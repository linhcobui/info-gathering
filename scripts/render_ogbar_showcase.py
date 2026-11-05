"""Showcase video of one OGBar play episode: the 224 px model input next to a high-resolution view from another angle.

The episode is generated exactly like the training data (play policy, physics, noise), with a fixed mass split and
start pose. Its recorded joint positions are then replayed through `render_debug` with a free camera at 1024 px:
the bar shell semi-transparent, the two hidden end masses drawn as balls sized by mass, the centre of mass as a green
ring. The model input (left) is the recorded `rgb`, untouched.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 pixi run python scripts/render_ogbar_showcase.py --w-right 0.03 --yaw 3.1416
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from ogbar.generate import Rejected, get_env, run_play_episode  # noqa: E402
from ogbar.play import SEGMENTS  # noqa: E402

RES = 1024


def font(size):
    try:
        return ImageFont.truetype(font_manager.findfont("P052", fallback_to_default=False), size)
    except Exception:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/ogbar_slot_v3_flat_noslip.yaml")
    ap.add_argument("--w-right", type=float, default=0.03, help="kg of 0.2; 0.03 -> COM at -12.6 cm (heavy end = body -x)")
    ap.add_argument("--x", type=float, default=0.44)
    ap.add_argument("--y", type=float, default=0.10)
    ap.add_argument("--yaw", type=float, default=np.pi, help="pi puts body -x (the heavy end) on camera D's image left")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--min-probes", type=int, default=2, help="(sampled plans only) first seed with this many probes")
    ap.add_argument("--plan", default=None, help="explicit plan, e.g. smash,probe,pair (pair = slide-in then slide-out)")
    ap.add_argument("--smash-s", type=float, default=None, help="grasp point of the smash (m from the bar's middle)")
    ap.add_argument("--max-steps", type=int, default=None, help="cap; default config episode_length (explicit plans: 2000)")
    ap.add_argument("--azimuth", type=float, default=135)
    ap.add_argument("--elevation", type=float, default=-25)
    ap.add_argument("--out-dir", default="data/videos_ogbar_showcase")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    cfg["debug"]["resolution"] = RES
    pose = (args.x, args.y, args.yaw)

    seed = args.seed
    while True:                                             # first seed whose plan has enough probes
        try:
            plan = None if args.plan is None else args.plan.split(",")
            res = run_play_episode(cfg, seed, w_right=args.w_right, bar_pose=pose, plan=plan,
                                   smash_s=None if args.smash_s is None else [args.smash_s],
                                   max_steps=args.max_steps or (2000 if plan else None))
        except Rejected:
            seed += 1
            continue
        n_probes = int((res["tables"]["segment"]["type"] == SEGMENTS.index("probe")).sum())
        if args.plan or n_probes >= args.min_probes:
            break
        seed += 1
    F, ep = res["frames"], res["episode"]
    c_cm = float(ep["com_offset_c"]) * 100
    segs = res["tables"]["segment"]
    print("segments: " + ", ".join(f"{SEGMENTS[int(t)]} {int(a)}-{int(b)} (grasp s {v * 100:+.1f} cm)" for t, a, b, v in
                                   zip(segs["type"], segs["t_start"], segs["t_end"], segs["s_planned"])))   # first attempt's point
    print(f"seed {seed}: {len(F['rgb'])} steps, {n_probes} probes, W_L {float(ep['w_left']) * 1000:.0f} g, "
          f"W_R {float(ep['w_right']) * 1000:.0f} g, c {c_cm:+.1f} cm; {res['summary']}")

    env = get_env(cfg).unwrapped                            # replay the recorded joint positions at high resolution
    env.reset(seed=seed, options={"w_right": args.w_right, "bar_pose": pose})
    env._model.vis.global_.offwidth = env._model.vis.global_.offheight = RES
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.42, -0.02, 0.10]
    cam.distance, cam.azimuth, cam.elevation = 1.05, args.azimuth, args.elevation
    seg_names = {i: s for i, s in enumerate(SEGMENTS)}
    f_title, f_small = font(30), font(24)
    frames = []
    for t in range(len(F["rgb"])):
        env._data.qpos[:] = F["qpos"][t]
        mujoco.mj_forward(env._model, env._data)
        hi = env.render_debug(camera=cam)
        small = np.asarray(Image.fromarray(F["rgb"][t]).resize((4 * 224, 4 * 224), Image.NEAREST))
        left = np.full((RES, 4 * 224, 3), 252, np.uint8)
        left[(RES - 896) // 2:(RES - 896) // 2 + 896] = small
        frame = Image.fromarray(np.concatenate([left, hi], 1))
        d = ImageDraw.Draw(frame)
        d.text((16, 14), "model input: side camera D, 224 x 224 (shown x4)", fill=(20, 20, 20), font=f_title)
        d.text((16, RES - 48), f"t = {t:4d}   {seg_names[int(F['segment_type'][t])]}   "
               f"tilt {np.degrees(F['bar_tilt'][t]):+5.1f} deg", fill=(20, 20, 20), font=f_small)
        d.text((4 * 224 + 16, 14), f"another angle, {RES} px - hidden masses drawn (not seen by the model)",
               fill=(255, 255, 255), font=f_title)
        d.text((4 * 224 + 16, 54), f"W_L {float(ep['w_left']) * 1000:.0f} g   W_R {float(ep['w_right']) * 1000:.0f} g   "
               f"centre of mass {c_cm:+.1f} cm (green ring)", fill=(255, 255, 255), font=f_small)
        frames.append(np.asarray(frame))
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = f"ogbar_play_c{c_cm:+.1f}cm_seed{seed}" + (f"_{args.plan.replace(',', '-')}" if args.plan else "")
    imageio.mimwrite(out / f"{stem}.mp4", frames, fps=20, quality=8, macro_block_size=8)
    pick = np.linspace(0, len(frames) - 1, 6).astype(int)
    sheet = np.concatenate([np.concatenate([frames[i] for i in pick[:3]], 1), np.concatenate([frames[i] for i in pick[3:]], 1)], 0)
    Image.fromarray(sheet).resize((sheet.shape[1] // 3, sheet.shape[0] // 3)).save(out / f"{stem}_sheet.png")
    print(out / f"{stem}.mp4")
    print(out / f"{stem}_sheet.png")


if __name__ == "__main__":
    main()
