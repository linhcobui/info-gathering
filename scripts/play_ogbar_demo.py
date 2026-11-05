"""Run one OGBar play episode (src/ogbar/play.py) and render it: debug view | side camera | 64 px model input.

  pixi run python scripts/play_ogbar_demo.py --config configs/data/ogbar_slot_v3.yaml --seed 0
"""
import argparse
import json
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ogbar.envs.bar_env import BarEnv            # noqa: E402
from ogbar.oracle import PHASE_NAMES             # noqa: E402
from ogbar.play import PlayEpisode               # noqa: E402

R = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/data/ogbar_slot_v3.yaml')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise', type=float, default=0.0, help='per-step action noise level xi (0 = clean)')
    ap.add_argument('--max-steps', type=int, default=1000)
    ap.add_argument('--out', default='data/videos_ogbar_play')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg['debug']['resolution'] = R
    cfg['oracle']['precise_grasp_axis_tol'] = 0.0015       # close only on the chosen grasp point (not hidden info)
    env = BarEnv(cfg, ob_type='pixels', width=64, height=64, visualize_info=False, terminate_at_goal=False)
    ob, info = env.reset(seed=args.seed)
    m, dd = env.model, env.data
    side = mujoco.MjvCamera()
    side.type = mujoco.mjtCamera.mjCAMERA_FREE
    side.lookat[:] = [0.425, 0.02, 0.03]
    side.distance, side.azimuth, side.elevation = 0.72, -90, -28
    rend = mujoco.Renderer(m, R, R)
    opt = mujoco.MjvOption()
    frames, labels = [], []
    c_true = env.mass.com_offset

    def on_step(ob, info, a, lab):
        labels.append(lab)
        if len(labels) % 2:
            return
        env.set_debug_grasp(lab['grasp_s'] if np.isfinite(lab['grasp_s']) else None)
        dbg = np.asarray(env.render_debug())
        rend.update_scene(dd, camera=side, scene_option=opt)
        sv = rend.render()
        obs = np.asarray(Image.fromarray(ob).resize((R, R), Image.NEAREST))
        img = Image.fromarray(np.concatenate([dbg, sv, obs], 1))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 3 * R, 38], fill=(0, 0, 0))
        ce = '' if not np.isfinite(lab['c_est']) else f"  COM estimate {lab['c_est']*100:+.1f} cm"
        d.text((4, 2), f"segment {lab['segment_idx'] + 1}: {lab['segment'].upper()}{' / ' + lab['probe_stage'] if 'probe_stage' in lab else ''}"
                       f"  | phase {lab['oracle_phase']}: {PHASE_NAMES.get(lab['oracle_phase'], '')}", fill=(255, 255, 0))
        d.text((4, 14), f"t={len(labels)}  grasp s {lab['grasp_s']*100:+.1f} cm{ce}  (true COM {c_true*100:+.1f} cm, debug only)", fill=(255, 255, 255))
        d.text((4, 26), f"tilt {np.degrees(info['privileged/bar_tilt'][0]):+5.1f} deg | seated {env.bar_in_slot()} | touching slot "
                        f"{bool(info['privileged/bar_slot_contact'][0])}", fill=(255, 255, 255))
        d.text((2 * R + 4, R - 12), 'model input (64x64)', fill=(255, 255, 255))
        frames.append(np.asarray(img))

    play = PlayEpisode(env, cfg, np.random.default_rng(args.seed + 1000), noise_xi=args.noise, max_steps=args.max_steps, on_step=on_step)
    log = play.run(ob, info)
    env.close()

    os.makedirs(args.out, exist_ok=True)
    stem = f'{args.out}/play_seed{args.seed}'
    imageio.mimwrite(f'{stem}.mp4', frames, fps=20, quality=8, macro_block_size=2)
    summary = dict(seed=args.seed, plan=play.plan, steps=len(labels), w_left=env.mass.w_left, w_right=env.mass.w_right,
                   true_com_cm=c_true * 100, com_estimate_cm=None if play.c_est is None else play.c_est * 100,
                   probes=[dict(s_cm=round(seg['s'] * 100, 1), lift_cm=round(seg['lift'] * 100, 1), tilt_deg=round(seg.get('tilt', float('nan')), 1))
                           for seg in log if seg['segment'] == 'probe'],
                   segments=[{k: (round(v * 100, 1) if k == 's' else v) for k, v in seg.items()} for seg in log])
    json.dump(summary, open(f'{stem}.json', 'w'), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))
    print('video:', f'{stem}.mp4')


if __name__ == '__main__':
    main()
