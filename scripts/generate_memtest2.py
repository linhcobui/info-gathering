"""Generate memtest2: two cups, ball under one of them, 50/50 swap during the hold.

    pixi run generate-memtest2 [--n N] [--config ...] [--append] [--preview-every K]
"""
import os, sys, yaml, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from info_gathering.rendering import build_model, render_episode
from info_gathering.memtest2 import Memtest2Recorder, hold_base, sample_episode
from tqdm import tqdm

def combinations(cfg):
    """(ball start slot, number of swaps) for every combination the config asks for."""
    counts = [int(n) for n in cfg.get("swap_counts", [0, 1])]
    return [(start, n) for n in counts for start in (0, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "data", "memtest2.yaml"))
    ap.add_argument("--n", type=int, default=None, help="override n_episodes")
    ap.add_argument("--preview-every", type=int, default=25,
                    help="dump an MP4 every K episodes (0 = none)")
    ap.add_argument("--append", action="store_true",
                    help="add these episodes to an existing zarr instead of overwriting it")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    n_episodes = args.n if args.n is not None else cfg["n_episodes"]

    meshdir_abs = os.path.join(ROOT, "submodules", "object_sim")
    model, data, handles = build_model(cfg, meshdir_abs)
    renderer = mujoco.Renderer(model, height=cfg["resolution"], width=cfg["resolution"])

    master_seed = cfg["seed"]
    master_rng = np.random.default_rng(master_seed)
    hold_values = cfg.get("hold_values")
    if not hold_values and cfg.get("hold_range"):
        lo, hi = cfg["hold_range"]
        hold_values = list(range(int(lo), int(hi) + 1))
    slack_values = cfg.get("slack_values")
    if not slack_values and cfg.get("slack_range"):
        lo, hi = cfg["slack_range"]
        slack_values = list(range(int(lo), int(hi) + 1))
    combos = combinations(cfg)
    if slack_values:
        # eval set with cumulative holds: sweep the slack, so each swap count gets its own hold
        plan = [(hold_base(cfg, n) + s, start, n) for s in slack_values for start, n in combos]
    elif hold_values:
        # eval set: every hold length x each (ball side, swap count) combination
        plan = [(h, start, swap) for h in hold_values for start, swap in combos]
    else:
        # training set: the combinations in equal numbers, shuffled
        picked = [combos[i % len(combos)] for i in range(n_episodes)]
        order = master_rng.permutation(n_episodes)
        plan = [(None, *picked[i]) for i in order]
    n_episodes = len(plan)
    episode_seeds = master_rng.integers(0, 2**31 - 1, size=n_episodes)

    zarr_path = os.path.join(ROOT, cfg["zarr_path"])
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
    rec = Memtest2Recorder(zarr_path, res=cfg["resolution"], append=args.append)
    first_new = len(rec.episodes["episode_ends"])

    video_dir = os.path.join(ROOT, cfg["video_dir"])
    os.makedirs(video_dir, exist_ok=True)

    try:
        for i, (hold_frames, ball_start, n_swaps) in enumerate(tqdm(plan, desc="episodes")):
            seed = int(episode_seeds[i])
            ep = sample_episode(cfg, np.random.default_rng(seed), ball_start, n_swaps, hold_frames=hold_frames)
            rgb = render_episode(model, data, handles, ep, cfg, renderer=renderer)
            rec.add_episode(rgb, ep, seed)

            if args.preview_every and (i % args.preview_every == 0):
                tag = f"start{ball_start}_{int(n_swaps)}swap_{'ball' if ep['ball_under_target'][-1] else 'empty'}"
                out = os.path.join(video_dir, f"ep_{first_new + i:05d}_{tag}.mp4")
                imageio.mimwrite(out, list(rgb), fps=cfg["fps"], quality=8)
    finally:
        renderer.close()

    rec.finalize(cfg, master_seed)
    ends = np.array(rec.episodes["episode_ends"])
    lengths = np.diff(np.concatenate([[0], ends]))
    swapped = np.array(rec.episodes["episode_swapped"])
    print(f"done. zarr -> {zarr_path}  ({rec.frames} frames, {len(ends)} episodes total, "
          f"{n_episodes} new from this run starting at index {first_new}, {int(swapped.sum())} swapped, "
          f"length {lengths.min()}-{lengths.max()} mean {lengths.mean():.1f})")
    print(f"previews -> {video_dir}")


if __name__ == "__main__":
    main()
