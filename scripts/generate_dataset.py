"""Generate the full dataset: N episodes -> zarr + sample MP4 previews."""
import os, sys, yaml, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from info_gathering.choreography import generate_plan, choreograph
from info_gathering.rendering import build_model, render_episode
from info_gathering.recorder import ZarrRecorder
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "memtest0.yaml"))
    ap.add_argument("--n", type=int, default=None, help="override n_episodes")
    ap.add_argument("--preview-every", type=int, default=20,
                    help="dump an MP4 every K episodes (0 = none)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    n_episodes = args.n if args.n is not None else cfg["n_episodes"]

    meshdir_abs = os.path.join(ROOT, "submodules", "object_sim")
    model, data, handles = build_model(cfg, meshdir_abs)
    renderer = mujoco.Renderer(model, height=cfg["resolution"], width=cfg["resolution"])

    # master -> per-episode seeds
    master_seed = cfg["seed"]
    master_rng = np.random.default_rng(master_seed)
    episode_seeds = master_rng.integers(0, 2**31 - 1, size=n_episodes)

    zarr_path = os.path.join(ROOT, cfg.get("zarr_path", "data/memtest0.zarr"))
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
    rec = ZarrRecorder(zarr_path, res=cfg["resolution"])

    video_dir = os.path.join(ROOT, cfg.get("video_dir", "data/videos"))
    os.makedirs(video_dir, exist_ok=True)

    try:
        for i in tqdm(range(n_episodes), desc='episodes'):
            seed = int(episode_seeds[i])
            ep_rng = np.random.default_rng(seed)
            plan = generate_plan(cfg, ep_rng)
            ep = choreograph(cfg, plan, ep_rng)
            rgb = render_episode(model, data, handles, ep, cfg, renderer=renderer)
            rec.add_episode(rgb, ep, seed)

            if args.preview_every and (i % args.preview_every == 0):
                out = os.path.join(video_dir, f"ep_{i:05d}.mp4")
                imageio.mimwrite(out, list(rgb), fps=cfg["fps"], quality=8)

            if (i + 1) % 10 == 0 or i == n_episodes - 1:
                print(f"  {i+1}/{n_episodes} episodes  ({rec._cursor} frames)")
    finally:
        renderer.close()

    rec.finalize(cfg, master_seed)
    print(f"done. zarr -> {zarr_path}  ({rec._cursor} frames, {n_episodes} episodes)")
    print(f"previews -> {video_dir}")


if __name__ == "__main__":
    main()
