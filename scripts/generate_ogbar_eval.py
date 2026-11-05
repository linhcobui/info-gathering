"""OGBar eval episodes: 5 probes each, bar start across the side camera's view, COM spread end to end.

Same env, physics and recorder as the training store (configs/data/ogbar_slot_v3_flat_noslip.yaml), with every
privileged key. What is fixed instead of sampled (user 2026-10-15):
  - one episode per centre of mass, W_R = 30 .. 170 g -> c = -12.6, -6.3, 0, +6.3, +12.6 cm (the training range);
  - start pose: bar lying along world x (yaw 0), i.e. side-on to camera D (which looks along -y), in the middle of
    the free floor, so its tilt shows in the image (it is put back down where it was after every probe);
  - plan: 5 probes and nothing else; the k-th probe grasps at PROBE_S[k] (fixed, also on retries), lifted LIFT m;
  - no action noise (xi = 0). The episode ends when the 5th probe has finished (no refill), capped at MAX_STEPS.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 pixi run python scripts/generate_ogbar_eval.py
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import imageio.v2 as imageio  # noqa: E402
import yaml  # noqa: E402

from ogbar.generate import Rejected, close_envs, episode_seed, finalize_store, run_play_episode  # noqa: E402
from ogbar.recorder import OgbarRecorder  # noqa: E402

W_RIGHT = [0.030, 0.065, 0.100, 0.135, 0.170]     # kg of 200 g -> c = (W_R - W_L)/M * 0.18 m
PROBE_S = [-0.12, 0.12, 0.0, -0.06, 0.06]          # grasp points along the bar (m from its middle), in order
LIFT = 0.15
BAR_POSE = (0.44, 0.10, 0.0)                       # x, y, yaw: grasp points x 0.32-0.56 all within reach
MAX_STEPS = 3000
MASTER_SEED = 20261015


def job(args):
    cfg, index, w_right = args
    for retry in range(cfg['health']['max_retries'] + 1):
        seed = episode_seed(MASTER_SEED, index, retry)
        try:
            res = run_play_episode(cfg, seed, w_right=w_right, debug=True, plan=['probe'] * len(PROBE_S),
                                   bar_pose=BAR_POSE, noise_xi=0.0, max_steps=MAX_STEPS, probe_s=PROBE_S,
                                   probe_lift=LIFT)
            close_envs()
            return seed, res, retry
        except Rejected as e:
            print(f'ep {index}: rejected ({e}), retrying', flush=True)
    raise RuntimeError(f'episode {index}: too many rejections')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/ogbar_slot_v3_flat_noslip.yaml")
    ap.add_argument("--zarr", default="data/ogbar_v3_eval5.zarr")
    ap.add_argument("--video-dir", default="data/videos_ogbar_v3_eval5")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    cfg['eval'] = dict(w_right=W_RIGHT, probe_s=PROBE_S, probe_lift=LIFT, bar_pose=list(BAR_POSE), noise_xi=0.0,
                       max_steps=MAX_STEPS, plan=['probe'] * len(PROBE_S), master_seed=MASTER_SEED,
                       source_config=args.config)
    zarr_path, video_dir = ROOT / args.zarr, ROOT / args.video_dir
    if zarr_path.exists():
        raise SystemExit(f'{zarr_path} exists; remove it first')
    video_dir.mkdir(parents=True, exist_ok=True)
    rec = OgbarRecorder(str(zarr_path), cfg['resolution'], append=False)
    with mp.get_context('spawn').Pool(len(W_RIGHT)) as pool:
        results = pool.map(job, [(cfg, i, w) for i, w in enumerate(W_RIGHT)])
    try:
        for i, (seed, res, retries) in enumerate(results):
            rec.add_episode(res['frames'], res['episode'], res['tables'], seed)
            ep = res['episode']
            c = float(ep['com_offset_c']) * 100
            name = f"eval_{i}_c{c:+.1f}cm_WL{float(ep['w_left'])*1000:.0f}_WR{float(ep['w_right'])*1000:.0f}.mp4"
            imageio.mimwrite(video_dir / name, res['debug_frames'], fps=cfg['fps'], quality=8, macro_block_size=8)
            seg = res['tables']['segment']
            tilts = ' '.join(f"{s*100:+.0f}cm:{t:+.1f}deg" for s, t in zip(seg['s_last'], seg['probe_tilt_deg']))
            print(f"ep {i}: c={c:+.1f}cm T={int(ep['length'])} probes [{tilts}] slipped={int(seg['probe_slipped'].sum())} "
                  f"retries={retries}", flush=True)
    finally:
        finalize_store(rec, cfg, MASTER_SEED)
    print(f'wrote {len(results)} episodes to {zarr_path}; videos in {video_dir}')


if __name__ == "__main__":
    main()
