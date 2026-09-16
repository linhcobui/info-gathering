"""Generate memtest1: single-cup memory episodes -> zarr + sample MP4 previews.

    pixi run generate-memtest1 [--n N] [--preview-every K]
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
from info_gathering.memtest1 import Memtest1Recorder, sample_episode
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "data", "memtest1.yaml"))
    ap.add_argument("--n", type=int, default=None, help="override n_episodes")
    ap.add_argument("--preview-every", type=int, default=50,
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
    if hold_values:
        # eval set: enumerate the holds, one ball and one empty episode each
        holds = [int(h) for h in hold_values for _ in (1, 0)]
        has_ball = np.array([b for _ in hold_values for b in (True, False)])
        n_episodes = len(holds)
    else:
        # training set: sampled holds, exactly ball_fraction of the episodes get a ball, shuffled
        holds = [None] * n_episodes
        n_ball = int(round(n_episodes * cfg["ball_fraction"]))
        has_ball = master_rng.permutation(np.arange(n_episodes) < n_ball)
    episode_seeds = master_rng.integers(0, 2**31 - 1, size=n_episodes)

    zarr_path = os.path.join(ROOT, cfg["zarr_path"])
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
    rec = Memtest1Recorder(zarr_path, res=cfg["resolution"], append=args.append)
    first_new = len(rec.episode_ends)

    video_dir = os.path.join(ROOT, cfg["video_dir"])
    os.makedirs(video_dir, exist_ok=True)

    try:
        for i in tqdm(range(n_episodes), desc="episodes"):
            seed = int(episode_seeds[i])
            ep = sample_episode(cfg, np.random.default_rng(seed), bool(has_ball[i]), hold_frames=holds[i])
            rgb = render_episode(model, data, handles, ep, cfg, renderer=renderer)
            rec.add_episode(rgb, ep, seed)

            if args.preview_every and (i % args.preview_every == 0):
                tag = "ball" if has_ball[i] else "empty"
                out = os.path.join(video_dir, f"ep_{i:05d}_{tag}.mp4")
                imageio.mimwrite(out, list(rgb), fps=cfg["fps"], quality=8)
    finally:
        renderer.close()

    rec.finalize(cfg, master_seed)
    lengths = np.diff(np.concatenate([[0], rec.episode_ends]))
    print(f"done. zarr -> {zarr_path}  ({rec.frames} frames, {len(rec.episode_ends)} episodes total, "
          f"{n_episodes} new from this run starting at index {first_new}, "
          f"{int(np.array(rec.episode_ball).sum())} with ball, length {lengths.min()}-{lengths.max()} "
          f"mean {lengths.mean():.1f})")
    print(f"previews -> {video_dir}")


if __name__ == "__main__":
    main()
