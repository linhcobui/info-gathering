"""Eval: does the world model imagine the ball under the lifted cup?

For every evaluated episode the frozen world model watches the real frames up to each action
onset -- the step whose action switches from [0,0,0] to a lift -- then imagines the next
`horizon` steps from the actions alone, and the decoder renders them. A colour detector looks
for the yellow ball in those frames; the label is whether the ball really is under the lifted
cup. For the second lift the model has also watched the first one.

Scored with the same detector:
    gt        real frames                          (detector sanity check, should be perfect)
    recon     decoded posterior of the real frames (what the decoder shows when the model sees them)
    imagined  decoded open-loop imagination        (the test)

    pixi run eval-lift decoder_run=logdir/world_model/r2dreamer/<wm run>/decoder/<timestamp>
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from info_gathering.ball_detector import ball_pixels  # noqa: E402
from info_gathering.dataset import EpisodeWindows  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402
from world_model.training.rollout import action_onsets  # noqa: E402
from world_model.utils.checkpoint import load_frozen_models  # noqa: E402
from world_model.utils.classification import binary_metrics  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402

SOURCES = ("gt", "recon", "imagined")
GREEN, RED = (90, 200, 110), (230, 70, 70)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_sheet(path, title, rows, first_step):
    """One row per episode: label | real frames | imagined frames (outlined green if the call was right)."""
    scale, gap, label_w, header = 2, 6, 230, 44
    cell = 64 * scale
    strip = max(len(r["frames"]["gt"]) for r in rows) * cell
    sheet = Image.new("RGB", (label_w + 2 * strip + 3 * gap, header + len(rows) * (cell + gap)), (18, 18, 22))
    d = ImageDraw.Draw(sheet)
    big, small = _font(16), _font(13)
    x_gt, x_im = label_w + gap, label_w + 2 * gap + strip
    d.text((gap, 6), title, font=big, fill=(235, 235, 235))
    d.text((x_gt, 26), f"real frames from step {first_step}", font=small, fill=(170, 170, 170))
    d.text((x_im, 26), f"imagined from step {first_step} (decoded)", font=small, fill=(170, 170, 170))
    for i, r in enumerate(rows):
        y = header + i * (cell + gap)
        ok = r["imagined_ball"] == r["label"]
        d.text((gap, y + 6), f"ep {r['episode']}  ball under slot {r['ball_slot']}", font=small, fill=(220, 220, 220))
        d.text((gap, y + 28), f"truth:    {'BALL' if r['label'] else 'empty'}", font=small, fill=(220, 220, 220))
        d.text((gap, y + 48), f"imagined: {'BALL' if r['imagined_ball'] else 'empty'}", font=small,
               fill=GREEN if ok else RED)
        d.text((gap, y + 68), f"yellow px (max): gt {r['gt_max_pixels']}, im {r['imagined_max_pixels']}", font=small,
               fill=(150, 150, 150))
        for x0, frames in ((x_gt, r["frames"]["gt"]), (x_im, r["frames"]["imagined"])):
            for j, frame in enumerate(frames):
                sheet.paste(Image.fromarray(frame).resize((cell, cell), Image.Resampling.NEAREST), (x0 + j * cell, y))
        d.rectangle([x_im - 3, y - 3, x_im + len(r["frames"]["imagined"]) * cell + 2, y + cell + 2],
                    outline=GREEN if ok else RED, width=2)
    sheet.save(path)


@hydra.main(version_base=None, config_path="../configs/dreamer", config_name="eval_lift")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    outdir = Path(cfg.outdir)

    # frozen world model + decoder, data exactly as in training
    model, decoder, train_set, _, dec_state, wm_state = load_frozen_models(cfg.decoder_checkpoint, cfg.device)

    arrays = train_set.arrays
    episodes = range(int(cfg.episodes.start), int(cfg.episodes.stop))
    seen = sorted(set(episodes) & set(train_set.episodes.tolist()))
    if seen:
        print(f"WARNING: {len(seen)} evaluated episodes were used for training: {seen[:10]}...")
    eval_set = EpisodeWindows(arrays, episodes, window=None)
    loader = DataLoader(eval_set, batch_size=int(cfg.batch_size), shuffle=False)
    threshold, min_pixels = float(cfg.detector.threshold), int(cfg.detector.min_pixels)

    rows = []
    for batch in loader:
        batch = to_device(batch, cfg.device)
        B, T = batch["is_first"].shape
        # the posterior is a causal filter: its latent at step t has only seen frames <= t,
        # so its first o+1 steps are exactly "watched the real frames until the onset"
        post = model.observe(batch)
        for lift, onset in enumerate(action_onsets(batch["action"]), start=1):
            ctx, end = onset + 1, min(T, onset + 1 + int(cfg.horizon))
            imag = model.imagine(post[:, :ctx], batch["prev_action"][:, ctx:end])
            frames = {
                "gt": batch["image"][:, ctx:end].float() / 255.0,
                "recon": decoder.render(post[:, ctx:end]),
                "imagined": decoder.render(imag),
            }
            pixels = {k: ball_pixels(v, threshold) for k, v in frames.items()}  # (B, H)
            # the label: is the ball under a cup lifted by the onset action?
            lifted = batch["action"][:, onset].reshape(B, arrays.frameskip, -1).amax(1) > 0  # (B, slots)
            ball_slot = batch["meta/ball_slot"][:, onset].long()
            label = lifted.gather(1, ball_slot[:, None]).squeeze(1)
            mse = (frames["imagined"] - frames["gt"]).pow(2).mean(dim=(1, 2, 3, 4))
            for i in range(B):
                row = {
                    "episode": int(batch["episode"][i]),
                    "lift": lift,
                    "onset_step": onset,
                    "onset_frame": onset * arrays.frameskip,
                    "lifted_slots": " ".join(str(s) for s in torch.nonzero(lifted[i]).flatten().tolist()),
                    "ball_slot": int(ball_slot[i]),
                    "label": bool(label[i]),
                    "imagined_mse": float(mse[i]),
                }
                for k in SOURCES:
                    row[f"{k}_ball"] = bool((pixels[k][i] >= min_pixels).any())
                    row[f"{k}_max_pixels"] = int(pixels[k][i].max())
                row["frames"] = {k: (frames[k][i].clamp(0, 1) * 255).byte().cpu().numpy() for k in ("gt", "imagined")}
                rows.append(row)

    # metrics per lift and overall
    metrics = {}
    lifts = sorted({r["lift"] for r in rows})
    for group in [*lifts, "all"]:
        sel = [r for r in rows if group == "all" or r["lift"] == group]
        metrics[f"lift{group}" if group != "all" else "all"] = {
            k: binary_metrics([r[f"{k}_ball"] for r in sel], [r["label"] for r in sel]) for k in SOURCES
        }
    (outdir / "metrics.json").write_text(json.dumps({"decoder_checkpoint": str(Path(cfg.decoder_checkpoint).resolve()),
                                                     "decoder_step": int(dec_state["step"]),
                                                     "wm_checkpoint": dec_state["wm_checkpoint"],
                                                     "wm_step": int(wm_state["step"]),
                                                     "config": OmegaConf.to_container(cfg, resolve=True),
                                                     "metrics": metrics}, indent=2))
    fields = [k for k in rows[0] if k != "frames"]
    with (outdir / "per_lift.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    for lift in lifts:
        sel = sorted((r for r in rows if r["lift"] == lift), key=lambda r: r["episode"])
        onset = sel[0]["onset_step"]
        draw_sheet(outdir / f"sheet_lift{lift}.png",
                   f"lift {lift}: watched real frames through step {onset}, imagined steps {onset + 1}+  |  "
                   f"imagined F1 {metrics[f'lift{lift}']['imagined']['f1']:.3f}", sel, onset + 1)

    print(f"\n{len(eval_set)} episodes, lifts at steps {sorted({r['onset_step'] for r in rows})} | "
          f"world model step {wm_state['step']}, decoder step {dec_state['step']} | {outdir}")
    print(f"{'group':6s} {'source':9s} {'n':>4s} {'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s} "
          f"{'prec':>6s} {'recall':>6s} {'F1':>6s} {'acc':>6s}")
    for group, by_source in metrics.items():
        for k, m in by_source.items():
            print(f"{group:6s} {k:9s} {m['n']:4d} {m['tp']:4d} {m['fp']:4d} {m['tn']:4d} {m['fn']:4d} "
                  f"{m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['accuracy']:6.3f}")


if __name__ == "__main__":
    main()
