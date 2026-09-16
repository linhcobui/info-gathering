"""Rollout videos: ground truth, and ground truth that switches to imagination at each action onset.

Rows, in one mp4 per episode and side by side for all episodes in all_episodes.mp4:
    1  ground truth
    2  ground truth up to and including the first action onset (lift 1), then imagined to the end
    3  ground truth up to and including the second action onset (lift 2), then imagined to the end
Imagined frames are outlined in red.

    pixi run rollout-videos decoder_run=logdir/world_model/r2dreamer/<wm run>/decoder/<timestamp>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from info_gathering.dataset import EpisodeWindows  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402
from world_model.training.rollout import action_onsets  # noqa: E402
from world_model.utils.checkpoint import load_frozen_models  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402

RED, GREY, WHITE = (230, 60, 60), (150, 150, 150), (235, 235, 235)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _even(n):
    return n + n % 2


def episode_video(episode, rows, action, ball, frameskip, scale):
    """rows: [(title, subtitle, frames (T, H, W, 3) uint8, first imagined step or None)] -> list of video frames."""
    T, H, W = rows[0][2].shape[:3]
    ch, cw = H * scale, W * scale
    label_w, header, gap = 240, 52, 10
    width, height = _even(label_w + cw + gap), _even(header + len(rows) * (ch + gap))
    big, small = _font(16), _font(13)
    video = []
    for t in range(T):
        canvas = Image.new("RGB", (width, height), (18, 18, 22))
        d = ImageDraw.Draw(canvas)
        lifted = np.flatnonzero(action[t].reshape(frameskip, -1).any(0))
        act = f"lift slot {', '.join(map(str, lifted))}" if len(lifted) else "none"
        d.text((gap, 6), f"ep {episode}   step {t}/{T - 1}   frame {t * frameskip}", font=big, fill=WHITE)
        d.text((gap, 28), f"action: {act}   |   ball under slot {ball[t]}", font=small, fill=(190, 190, 190))
        for r, (title, subtitle, frames, switch) in enumerate(rows):
            y = header + r * (ch + gap)
            imagined = switch is not None and t >= switch
            d.text((gap, y + 6), title, font=small, fill=(220, 220, 220))
            d.text((gap, y + 24), subtitle, font=small, fill=GREY)
            d.text((gap, y + 50), "IMAGINED" if imagined else "real", font=big, fill=RED if imagined else GREY)
            canvas.paste(Image.fromarray(frames[t]).resize((cw, ch), Image.Resampling.NEAREST), (label_w, y))
            if imagined:
                d.rectangle([label_w - 4, y - 4, label_w + cw + 3, y + ch + 3], outline=RED, width=3)
        video.append(np.asarray(canvas))
    return video


def grid_video(episodes, rows, frameskip, scale=2):
    """All episodes side by side (columns), rows stacked; imagined frames outlined in red."""
    T, H, W = rows[0][2].shape[1:4]
    ch, cw, gap, header = H * scale, W * scale, 8, 46
    width = _even(gap + len(episodes) * (cw + gap))
    height = _even(header + len(rows) * (ch + gap))
    big, small = _font(15), _font(12)
    video = []
    for t in range(T):
        canvas = Image.new("RGB", (width, height), (18, 18, 22))
        d = ImageDraw.Draw(canvas)
        d.text((gap, 4), f"step {t}/{T - 1}  frame {t * frameskip}   rows: real | real, then imagined from lift 1 | "
                         f"real, then imagined from lift 2   (red = imagined)", font=big, fill=WHITE)
        for c, episode in enumerate(episodes):
            x = gap + c * (cw + gap)
            d.text((x, 26), f"ep {episode}", font=small, fill=(190, 190, 190))
            for r, (_, _, frames, switch) in enumerate(rows):
                y = header + r * (ch + gap)
                canvas.paste(Image.fromarray(frames[c, t]).resize((cw, ch), Image.Resampling.NEAREST), (x, y))
                if switch is not None and t >= switch:
                    d.rectangle([x - 3, y - 3, x + cw + 2, y + ch + 2], outline=RED, width=3)
        video.append(np.asarray(canvas))
    return video


@hydra.main(version_base=None, config_path="../configs/dreamer", config_name="rollout_videos")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    outdir = Path(cfg.outdir)
    model, decoder, train_set, _, dec_state, wm_state = load_frozen_models(cfg.decoder_checkpoint, cfg.device)
    arrays = train_set.arrays
    episodes = list(range(int(cfg.episodes.start), int(cfg.episodes.start) + int(cfg.episodes.count)))
    clips = EpisodeWindows(arrays, episodes, window=None)
    batch = to_device(default_collate([clips[i] for i in range(len(clips))]), cfg.device)

    # posterior over the real frames; causal, so its first o+1 steps have only seen frames <= o
    post = model.observe(batch)
    gt = batch["image"]
    rows = [("ground truth", "", gt.cpu().numpy(), None)]
    for k, onset in enumerate(action_onsets(batch["action"]), start=1):
        ctx = onset + 1
        imagined = (decoder.render(model.imagine(post[:, :ctx], batch["prev_action"][:, ctx:])) * 255).byte()
        rows.append((f"real until lift {k}", f"imagined from step {ctx}", torch.cat([gt[:, :ctx], imagined], 1)
                     .cpu().numpy(), ctx))

    action = batch["action"].cpu().numpy()
    ball = batch["meta/ball_slot"].cpu().numpy()
    fps = arrays.fps
    for i, episode in enumerate(episodes):
        per_episode = [(title, sub, frames[i], switch) for title, sub, frames, switch in rows]
        video = episode_video(episode, per_episode, action[i], ball[i], arrays.frameskip, int(cfg.scale))
        imageio.mimwrite(outdir / f"ep{episode:04d}.mp4", video, fps=fps, macro_block_size=1)
    imageio.mimwrite(outdir / "all_episodes.mp4", grid_video(episodes, rows, arrays.frameskip), fps=fps,
                     macro_block_size=1)
    print(f"world model step {wm_state['step']}, decoder step {dec_state['step']} | "
          f"{len(episodes)} episodes -> {outdir}")


if __name__ == "__main__":
    main()
