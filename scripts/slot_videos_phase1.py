"""Rollout videos annotated with the slot head's belief about where the ball is, frame by frame.

For one eval episode: the posterior chain runs over real frames up to the lift onset, then the
prior imagines the lift. At EVERY step the reward decoder's memory-only read-out r_t (read at the
last memory token, so it sees the state and nothing else) gives p(slot 0/1/2). Each frame shows:

    left   the real frame
    right  real until the onset, then the model's imagined frame (readout -> pixel decoder)
    below  p(slot) bars, the true slot (dense label: switches at a swap's halfway frame),
           and whether the model is OBSERVING (posterior) or IMAGINING (prior)

Meaningful for a model trained with loss.slot_supervision: dense, where r_t is supervised on
every frame. With lift-only supervision r_t is only trained during the lift.

    pixi run python scripts/slot_videos_phase1.py <decoder_run> <outdir> <episode>[:<tag>] ...
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from info_gathering.dataset_memtest1 import memtest1_datasets  # noqa: E402
from info_gathering.dataset_smt_prelim import dense_ball_slot  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402
from world_model.training.rollout import action_onsets  # noqa: E402
from world_model.utils.checkpoint import load_frozen_models  # noqa: E402

ZARR = "data/memtest4_eval.zarr"
PHASE = {0: "reveal", 1: "hold", 2: "lift"}


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


@torch.no_grad()
def slot_probs(model, z):
    """z (T, K, d) -> (T, 3) p(slot) from r_t, the read-out at the last memory token."""
    dummy_e = torch.zeros(z.shape[0], 1, model.embed_dim, device=z.device)
    dummy_a = torch.zeros(z.shape[0], 1, model.action_dim, device=z.device)
    logits = model.reward_decoder(z, dummy_e, dummy_a)        # (T, 2, 3); index 0 = r_t, memory only
    return torch.softmax(logits[:, 0].float(), -1).cpu().numpy()


@torch.no_grad()
def rollout(model, decoder, batch):
    onset = action_onsets(batch["action"])[1]
    post = model.observe(batch)                                # warm-up at t=0, posterior to the end
    imag = model.imagine(post[:, :onset + 1], batch["prev_action"][:, onset + 1:])
    z = torch.cat([post["z"][0, :onset + 1], imag["z"][0]], 0).float()
    imagined = decoder.render(imag)[0].clamp(0, 1)
    return onset, z, (imagined * 255).byte().cpu().numpy()


def draw(frame_real, frame_right, probs, truth, t, T, onset, phase, title, verdict):
    S, W = 256, 256 * 2 + 30
    img = Image.new("RGB", (W, 520), (18, 18, 22))
    d = ImageDraw.Draw(img)
    big, mid, small = font(18), font(15), font(13)
    imagining = t > onset
    d.text((10, 8), title, font=big, fill=(235, 235, 235))
    d.text((10, 32), f"step {t}/{T-1}   phase: {phase}", font=mid, fill=(190, 190, 190))
    mode, col = (("IMAGINING (prior, no frames)", (230, 70, 70)) if imagining
                 else ("OBSERVING (posterior)", (120, 200, 120)))
    d.text((W - 10 - d.textlength(mode, font=mid), 32), mode, font=mid, fill=col)
    y0 = 58
    img.paste(Image.fromarray(frame_real).resize((S, S), Image.Resampling.NEAREST), (10, y0))
    img.paste(Image.fromarray(frame_right).resize((S, S), Image.Resampling.NEAREST), (20 + S, y0))
    d.text((10, y0 + S + 4), "real", font=small, fill=(160, 160, 160))
    d.text((20 + S, y0 + S + 4), "model (imagined)" if imagining else "real (model observing)",
           font=small, fill=(230, 70, 70) if imagining else (160, 160, 160))
    if imagining:
        d.rectangle([18 + S, y0 - 2, 21 + 2 * S, y0 + S + 1], outline=(230, 70, 70), width=3)

    # slot belief bars
    by, bh, bw, gap = y0 + S + 24, 64, 160, 16
    pred = int(np.argmax(probs))
    d.text((10, by), "model's belief p(slot)   (yellow triangle = where the ball really is)",
           font=mid, fill=(220, 220, 220))
    for k in range(3):
        x = 10 + k * (bw + gap)
        d.rectangle([x, by + 20, x + bw, by + 20 + bh], outline=(90, 90, 90))
        h = int(bh * probs[k])
        fill = (70, 150, 230) if k == pred else (80, 80, 95)
        if h > 0:
            d.rectangle([x + 1, by + 20 + bh - h, x + bw - 1, by + 19 + bh], fill=fill)
        d.text((x + 4, by + 22), f"slot {k}{' (left)' if k == 0 else ''}", font=small, fill=(230, 230, 230))
        d.text((x + 4, by + 40), f"{probs[k]:.2f}", font=mid, fill=(255, 255, 255))
        if k == truth:
            d.polygon([(x + bw // 2 - 9, by + bh + 34), (x + bw // 2 + 9, by + bh + 34),
                       (x + bw // 2, by + bh + 22)], fill=(250, 200, 40))
    ok = pred == truth
    d.text((10, by + bh + 52), f"predicted slot {pred}   true slot {truth}   "
                               f"{'correct' if ok else 'WRONG'}", font=mid,
           fill=(120, 200, 120) if ok else (230, 70, 70))
    d.text((10, by + bh + 76), verdict, font=small, fill=(170, 170, 170))
    return np.asarray(img)


def main(decoder_run, outdir, specs):
    model, decoder, *_ = load_frozen_models(f"{decoder_run}/checkpoints/decoder_latest.pt", "cuda:0")
    eval_set, _ = memtest1_datasets(ZARR, image_size=64, holdout_episodes=0,
                                    meta_keys=("ball_under_target", "ball_slot", "phase"))
    arrays = eval_set.arrays
    real_len = arrays.steps - arrays.meta["pad"].sum(1)
    dense = dense_ball_slot(arrays.meta["ball_slot"], real_len)
    import zarr
    swaps = zarr.open_group(str(ROOT / ZARR), mode="r")["episode_swapped"][:]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        e, _, tag = spec.partition(":")
        e = int(e)
        n = int(real_len[e])
        batch = to_device(default_collate([eval_set[e]]), "cuda:0")
        batch = {k: v[:, :n] if v.dim() > 1 else v for k, v in batch.items()}
        onset, z, imagined = rollout(model, decoder, batch)
        probs = slot_probs(model, z)
        real = arrays.image[e, :n]
        final_pred, final_true = int(np.argmax(probs[onset + 1])), int(dense[e, onset + 1])
        title = f"ep {e}   {int(swaps[e])} cup swaps   {tag.replace('_', ' ')}"
        verdict = (f"at the lift the model says slot {final_pred}; ball is in slot {final_true} "
                   f"-> {'CORRECT' if final_pred == final_true else 'WRONG'}")
        frames = []
        for t in range(n):
            right = real[t] if t <= onset else imagined[t - onset - 1]
            frames.append(draw(real[t], right, probs[t], int(dense[e, t]), t, n, onset,
                               PHASE[int(arrays.meta["phase"][e, t])], title, verdict))
        path = out / f"ep{e:03d}_{tag}_{'correct' if final_pred == final_true else 'wrong'}.mp4"
        imageio.mimwrite(path, frames, fps=4, macro_block_size=1)
        hit = probs.argmax(1) == dense[e, :n]
        wrong = np.flatnonzero(~hit)
        first = (f"first wrong at step {wrong[0]} ({PHASE[int(arrays.meta['phase'][e, wrong[0]])]})"
                 if len(wrong) else "never wrong")
        print(f"{path.name}  | {n} frames, onset {onset}, per-frame slot acc {hit.mean():.2f}, "
              f"{first}, lift prediction {final_pred} vs {final_true}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
