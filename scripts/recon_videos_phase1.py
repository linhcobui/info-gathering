"""Videos of what the POSTERIOR belief looks like, decoded to pixels at every step.

Three strips per frame:
    real                the recorded frame
    from belief         readout(posterior state at t) -> pixel decoder. The model has seen
                        frames up to t and nothing after; this is its belief rendered.
    imagined            from the lift onset on, the PRIOR chain rendered the same way. Before
                        the onset there is nothing to imagine yet.

Below them, the head's read-out at that step (r_t, memory only): p(slot) bars for a slot head,
or the ball/empty call for a binary head, with the true slot marked.

    pixi run python scripts/recon_videos_phase1.py <decoder_run> <outdir> <episode>[:tag] ...
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

PHASE = {0: "reveal", 1: "hold", 2: "lift"}
SURF, INK, MUTED = (18, 18, 22), (235, 235, 235), (150, 150, 150)
GOOD, BAD, ACC = (120, 200, 120), (230, 70, 70), (70, 150, 230)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


@torch.no_grad()
def render(model, decoder, z):
    """readout(z) -> embedding -> pixels, for a (T, K, d) stack of states."""
    emb = model.readout(z.unsqueeze(0)).float()
    frames = decoder.render(TensorDict({"embed": emb}, batch_size=tuple(emb.shape[:2])))[0]
    return (frames.clamp(0, 1) * 255).byte().cpu().numpy()


@torch.no_grad()
def head_readout(model, z, act):
    """r_t, read at the last memory token: memory only, no observation."""
    e = torch.zeros(z.shape[0], 1, model.embed_dim, device=z.device)
    out = model.reward_decoder(z, e, act.unsqueeze(1))[:, 0].float()
    if model.reward_decoder.head == "slot":
        return torch.softmax(out, -1).cpu().numpy(), "slot"
    return out.cpu().numpy(), "binary"


def draw_two(real, dec, read, kind, truth, t, T, onset, phase, title):
    """Two strips: ground truth, and the model's decode -- posterior up to the lift, prior after."""
    S, gap = 288, 14
    img = Image.new("RGB", (2 * S + 3 * gap, S + 104), SURF)
    d = ImageDraw.Draw(img)
    big, mid, small = font(19), font(15), font(13)
    imagining = t > onset
    d.text((gap, 8), title, font=big, fill=INK)
    d.text((gap, 32), f"step {t}/{T-1}   phase: {phase}", font=mid, fill=(190, 190, 190))
    mode = "IMAGINING (prior, no frames)" if imagining else "OBSERVING (posterior)"
    d.text((2 * S + 2 * gap - d.textlength(mode, font=mid), 32), mode, font=mid,
           fill=BAD if imagining else GOOD)
    y0 = 56
    for i, (lab, arr) in enumerate((("ground truth", real), ("decoded from the state", dec))):
        x = gap + i * (S + gap)
        img.paste(Image.fromarray(arr).resize((S, S), Image.Resampling.NEAREST), (x, y0))
        d.text((x, y0 + S + 4), lab, font=small, fill=BAD if (i == 1 and imagining) else MUTED)
        if i == 1 and imagining:
            d.rectangle([x - 2, y0 - 2, x + S + 1, y0 + S + 1], outline=BAD, width=3)
    if kind == "slot":
        pred = int(np.argmax(read)); ok = pred == truth
        txt = f"belief: ball in slot {pred} (p={read[pred]:.2f})   truth: slot {truth}"
    else:
        says = read > 0; ok = says == (truth == 0)
        txt = f"belief: ball under the left cup? {'YES' if says else 'no'}   truth: slot {truth}"
    d.text((gap, y0 + S + 26), txt + f"   {'correct' if ok else 'WRONG'}", font=mid,
           fill=GOOD if ok else BAD)
    return np.asarray(img)


def draw(real, belief, imag, read, kind, truth, t, T, onset, phase, title):
    S, gap = 224, 12
    W = 3 * S + 4 * gap
    img = Image.new("RGB", (W, S + 190), SURF)
    d = ImageDraw.Draw(img)
    big, mid, small = font(17), font(14), font(12)
    d.text((gap, 8), title, font=big, fill=INK)
    d.text((gap, 30), f"step {t}/{T-1}   phase: {phase}", font=mid, fill=(190, 190, 190))
    y0 = 54
    strips = [("real frame", real), ("decoded from the posterior belief", belief),
              ("imagined (prior)" if t > onset else "imagined: not started", imag)]
    for i, (lab, arr) in enumerate(strips):
        x = gap + i * (S + gap)
        if arr is None:
            d.rectangle([x, y0, x + S, y0 + S], outline=(60, 60, 66))
            d.text((x + 8, y0 + S // 2), "—", font=big, fill=(70, 70, 76))
        else:
            img.paste(Image.fromarray(arr).resize((S, S), Image.Resampling.NEAREST), (x, y0))
        col = BAD if (i == 2 and t > onset) else MUTED
        d.text((x, y0 + S + 4), lab, font=small, fill=col)
        if i == 2 and t > onset:
            d.rectangle([x - 2, y0 - 2, x + S + 1, y0 + S + 1], outline=BAD, width=2)
    by = y0 + S + 26
    if kind == "slot":
        pred = int(np.argmax(read))
        d.text((gap, by), "belief about the ball's slot (read from memory alone):", font=mid, fill=INK)
        for k in range(3):
            x = gap + k * 150
            d.rectangle([x, by + 22, x + 130, by + 60], outline=(90, 90, 90))
            w = int(130 * read[k])
            if w > 0:
                d.rectangle([x + 1, by + 23, x + w, by + 59], fill=ACC if k == pred else (80, 80, 95))
            d.text((x + 6, by + 30), f"slot {k}   {read[k]:.2f}", font=small, fill=(235, 235, 235))
            if k == truth:
                d.text((x + 6, by + 64), "^ ball is here", font=small, fill=(250, 200, 40))
        ok = pred == truth
        d.text((gap, by + 84), f"predicted slot {pred}   true slot {truth}   {'correct' if ok else 'WRONG'}",
               font=mid, fill=GOOD if ok else BAD)
    else:
        says = read > 0
        ok = says == (truth == 0)
        d.text((gap, by), f"belief: ball under the LEFT cup?  {'YES' if says else 'no'}  (logit {read:+.2f})",
               font=mid, fill=INK)
        d.text((gap, by + 26), f"truth: ball is in slot {truth}  ->  {'YES' if truth == 0 else 'no'}   "
                               f"{'correct' if ok else 'WRONG'}", font=mid, fill=GOOD if ok else BAD)
    return np.asarray(img)


def main(decoder_run, outdir, specs):
    model, decoder, *_ = load_frozen_models(f"{decoder_run}/checkpoints/decoder_latest.pt", "cuda:0")
    zp = "data/memtest4_eval5.zarr" if any(":d5" in s for s in specs) else "data/memtest4_eval.zarr"
    ds, _ = memtest1_datasets(zp, image_size=64, holdout_episodes=0,
                              meta_keys=("ball_under_target", "ball_slot", "phase"))
    A = ds.arrays
    real_len = A.steps - A.meta["pad"].sum(1)
    dense = dense_ball_slot(A.meta["ball_slot"], real_len)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        e, _, tag = spec.partition(":")
        e = int(e)
        n = int(real_len[e])
        batch = to_device(default_collate([ds[e]]), "cuda:0")
        batch = {k: v[:, :n] if v.dim() > 1 else v for k, v in batch.items()}
        onset = action_onsets(batch["action"])[1]
        post = model.observe(batch)
        z = post["z"][0].float()
        belief = render(model, decoder, z)
        imag = model.imagine(post[:, :onset + 1], batch["prev_action"][:, onset + 1:])
        imagined = render(model, decoder, imag["z"][0].float())
        # the caption must describe the state actually being DRAWN: posterior while observing,
        # prior once imagining. Reading the posterior throughout would report a belief that has
        # seen the lift frames next to a picture generated without them.
        read, kind = head_readout(model, z, batch["prev_action"][0].float())
        read_im, _ = head_readout(model, imag["z"][0].float(), batch["prev_action"][0, onset + 1:].float())
        real = A.image[e, :n]
        title = f"ep {e}  {tag.replace('_', ' ')}"
        import os
        two = os.environ.get("TWO_PANEL", "") == "1"
        if two:
            # the decoded strip switches source at the lift: posterior while observing, prior after
            dec = [belief[t] if t <= onset else imagined[t - onset - 1] for t in range(n)]
            rd = [read[t] if t <= onset else read_im[t - onset - 1] for t in range(n)]
            frames = [draw_two(real[t], dec[t], rd[t], kind, int(dense[e, t]), t, n, onset,
                               PHASE[int(A.meta["phase"][e, t])], title) for t in range(n)]
        else:
            frames = [draw(real[t], belief[t], None if t <= onset else imagined[t - onset - 1],
                           read[t], kind, int(dense[e, t]), t, n, onset,
                           PHASE[int(A.meta["phase"][e, t])], title) for t in range(n)]
        path = out / f"ep{e:03d}_{tag or 'recon'}{'_2panel' if two else ''}.mp4"
        imageio.mimwrite(path, frames, fps=4, macro_block_size=1)
        print(f"{path}  ({n} frames, onset {onset})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
