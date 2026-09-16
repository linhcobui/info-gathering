"""memtest1 eval: after watching the hold, does the world model imagine the ball under the lifted cup?

For every episode of the eval set the frozen world model watches the real frames up to the step
the lift action turns on -- the end of the hold -- and then imagines the rest of the lift from the
actions alone. The decoder renders those frames and a colour detector looks for the ball; the label
is whether the ball is really there. Episodes are grouped by hold length so each batch shares its
action onsets.

Two segments are scored:
    lift    the second onset, after the hold: the memory test
    reveal  the first onset, before the ball has ever been shown: a control, where the model
            cannot know and should say "empty" (or guess)

and three sources, as in the cups eval:
    gt        real frames                          (detector sanity check)
    recon     decoded posterior of the real frames (what the decoder shows when the model sees them)
    imagined  decoded open-loop imagination        (the test)

    pixi run eval-memtest1 decoder_run=logdir/world_model/r2dreamer_memtest1/<wm run>/decoder/<ts>
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from info_gathering.ball_detector import ball_pixels  # noqa: E402
from info_gathering.dataset_memtest1 import memtest1_datasets  # noqa: E402
from info_gathering.memtest1 import PHASES  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402
from world_model.training.rollout import action_onsets  # noqa: E402
from world_model.utils.checkpoint import load_frozen_models  # noqa: E402
from world_model.utils.classification import binary_metrics  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402

SOURCES = ("gt", "recon", "imagined")
SEGMENTS = {1: "reveal", 2: "lift"}  # by onset index
# reference palette (dataviz skill), light mode
SURFACE, INK, SECOND, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GOOD, BAD = "#0ca30c", "#d03b3b"  # status palette: correct / wrong


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_sheet(path, title, rows):
    """One row per episode: label | real frames | imagined frames (outlined green when the call is right)."""
    scale, gap, label_w, header = 2, 6, 250, 44
    cell = 64 * scale
    strip = max(len(r["gt"]) for r in rows) * cell
    sheet = Image.new("RGB", (label_w + 2 * strip + 3 * gap, header + len(rows) * (cell + gap)), (18, 18, 22))
    d = ImageDraw.Draw(sheet)
    big, small = _font(16), _font(13)
    x_gt, x_im = label_w + gap, label_w + 2 * gap + strip
    d.text((gap, 6), title, font=big, fill=(235, 235, 235))
    d.text((x_gt, 26), "real frames after the action turns on", font=small, fill=(170, 170, 170))
    d.text((x_im, 26), "imagined from the same step (decoded)", font=small, fill=(170, 170, 170))
    for i, r in enumerate(rows):
        y = header + i * (cell + gap)
        ok = r["imagined_ball"] == r["label"]
        colour = (90, 200, 110) if ok else (230, 70, 70)
        d.text((gap, y + 6), f"hold {r['hold']}  (steps {r['context']}+)", font=small, fill=(220, 220, 220))
        d.text((gap, y + 28), f"truth:    {'BALL' if r['label'] else 'empty'}", font=small, fill=(220, 220, 220))
        d.text((gap, y + 48), f"imagined: {'BALL' if r['imagined_ball'] else 'empty'}", font=small, fill=colour)
        d.text((gap, y + 68), f"yellow px: gt {r['gt_max_pixels']}, im {r['imagined_max_pixels']}", font=small,
               fill=(150, 150, 150))
        for x0, frames in ((x_gt, r["gt"]), (x_im, r["imagined"])):
            for j, frame in enumerate(frames):
                sheet.paste(Image.fromarray(frame).resize((cell, cell), Image.Resampling.NEAREST), (x0 + j * cell, y))
        d.rectangle([x_im - 3, y - 3, x_im + len(r["imagined"]) * cell + 2, y + cell + 2], outline=colour, width=2)
    sheet.save(path)


def draw_hold_chart(path, rows, f1):
    """Per-hold outcome: one marker per episode, correct or wrong. Two episodes per hold, so no curve."""
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    size = max(10, min(90, 3600 / max(1, len({r["hold"] for r in rows}))))  # shrink as the hold range grows
    for lane, label in ((1, "ball"), (0, "empty")):
        sel = [r for r in rows if bool(r["label"]) == bool(lane)]
        for r in sel:
            ok = r["imagined_ball"] == r["label"]
            ax.scatter(r["hold"], lane, s=size, marker="o" if ok else "X", color=GOOD if ok else BAD, zorder=3)
    trained = [r["hold"] for r in rows if r["hold"] <= 15]
    ax.axvspan(min(trained) - 0.5, 15.5, color=GRID, alpha=0.5, zorder=0)
    ax.text(15.4, 1.42, "holds seen in training", ha="right", color=MUTED, fontsize=10)
    ax.scatter([], [], s=90, marker="o", color=GOOD, label="correct")
    ax.scatter([], [], s=90, marker="X", color=BAD, label="wrong")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["empty", "ball"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("hold length (steps with the cup closed)", color=SECOND, fontsize=11, labelpad=8)
    ax.set_title(f"memtest1: imagined lift after the hold  ·  F1 {f1:.3f}", color=INK, fontsize=14, pad=14, loc="left")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=11, labelcolor=SECOND, ncol=2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)


def write_video(path, clip, fps, scale=3):
    """Two rows: the real episode, and the real episode switching to the imagined lift at `context`."""
    names = {v: k for k, v in PHASES.items()}
    real, imagined, ctx = clip["real"], clip["imagined"], clip["context"]
    mixed = np.concatenate([real[:ctx], imagined])
    T = len(real)
    cell, gap, label_w, header = 64 * scale, 10, 250, 54
    width, height = label_w + cell + gap, header + 2 * (cell + gap)
    width += width % 2
    height += height % 2
    big, small = _font(16), _font(13)
    frames = []
    for t in range(T):
        canvas = Image.new("RGB", (width, height), (18, 18, 22))
        d = ImageDraw.Draw(canvas)
        header_bits = f"hold {clip['hold']}   step {t}/{T - 1}"
        if clip.get("tag"):
            header_bits += f"   {clip['tag'].replace('_', '  ')}"
        d.text((gap, 6), header_bits, font=big, fill=(235, 235, 235))
        d.text((gap, 28), f"action {int(clip['action'][t])}  {names[int(clip['phase'][t])]}   |   truth: "
                          f"{'BALL' if clip['label'] else 'empty'}", font=small, fill=(190, 190, 190))
        for r, (title, strip) in enumerate((("ground truth", real), ("real, then imagined", mixed))):
            y = header + r * (cell + gap)
            imagining = r == 1 and t >= ctx
            d.text((gap, y + 6), title, font=small, fill=(220, 220, 220))
            if r == 1:
                d.text((gap, y + 26), f"switches at step {ctx}", font=small, fill=(150, 150, 150))
                d.text((gap, y + 52), "IMAGINED" if imagining else "real", font=big,
                       fill=(230, 60, 60) if imagining else (150, 150, 150))
                if imagining:
                    d.text((gap, y + 76), f"predicts: {'BALL' if clip['predicted'] else 'empty'}", font=small,
                           fill=(230, 60, 60))
            canvas.paste(Image.fromarray(strip[t]).resize((cell, cell), Image.Resampling.NEAREST), (label_w, y))
            if imagining:
                d.rectangle([label_w - 4, y - 4, label_w + cell + 3, y + cell + 3], outline=(230, 60, 60), width=3)
        frames.append(np.asarray(canvas))
    imageio.mimwrite(path, frames, fps=max(1, int(fps)), macro_block_size=1)


@hydra.main(version_base=None, config_path="../configs/dreamer", config_name="eval_memtest1")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    outdir = Path(cfg.outdir)
    model, decoder, _, _, dec_state, wm_state = load_frozen_models(cfg.decoder_checkpoint, cfg.device)
    wm_cfg = OmegaConf.create(wm_state["config"])

    label_key = str(cfg.get("label_key", "ball_present"))
    eval_set, _ = memtest1_datasets(cfg.zarr_path, image_size=wm_cfg.model.image_size, holdout_episodes=0,
                                    meta_keys=tuple(cfg.get("meta_keys", ["ball_present", "phase"])))
    arrays = eval_set.arrays
    pad = arrays.meta["pad"]
    real_len = arrays.steps - pad.sum(1)          # padding repeats the final frame
    # frames outside the hold: memtest1/2/3 have closed 10 + reveal 10 + show 10, memtest4 has no
    # closed phase, so its offset is 20
    hold = real_len - int(cfg.get("hold_offset", 30))
    threshold, min_pixels = float(cfg.detector.threshold), int(cfg.detector.min_pixels)

    rows, video_clips = [], []
    video_holds = {int(v) for v in cfg.videos.holds}
    for h in np.unique(hold):                     # group by hold so a batch shares its onsets
        episodes = np.flatnonzero(hold == h)
        batch = to_device(default_collate([eval_set[int(e)] for e in episodes]), cfg.device)
        post = model.observe(batch)               # causal filter: step t has only seen frames <= t
        for index, onset in enumerate(action_onsets(batch["action"]), start=1):
            segment = SEGMENTS.get(index, f"onset{index}")
            ctx = onset + 1
            end = min(int(real_len[episodes[0]]), ctx + int(cfg.horizon))
            imagined = model.imagine(post[:, :ctx], batch["prev_action"][:, ctx:end])
            frames = {
                "gt": batch["image"][:, ctx:end].float() / 255.0,
                "recon": decoder.render(post[:, ctx:end]),
                "imagined": decoder.render(imagined),
            }
            pixels = {k: ball_pixels(v, threshold) for k, v in frames.items()}
            # read the label at the onset step: it can change during the hold (memtest2's swap)
            label = batch[f"meta/{label_key}"][:, onset].bool()
            mse = (frames["imagined"] - frames["gt"]).pow(2).mean(dim=(1, 2, 3, 4))
            for i, e in enumerate(episodes):
                row = {"episode": int(e), "segment": segment, "hold": int(h), "onset": onset, "context": ctx,
                       "imagined_steps": end - ctx, "label": bool(label[i]), "imagined_mse": float(mse[i])}
                for k in SOURCES:
                    row[f"{k}_ball"] = bool((pixels[k][i] >= min_pixels).any())
                    row[f"{k}_max_pixels"] = int(pixels[k][i].max())
                if "meta/ball_slot" in batch:
                    # the ball changes slot exactly once per swap, so count its transitions
                    slots = batch["meta/ball_slot"][i, :int(real_len[e])]
                    row["swaps"] = int((slots[1:] != slots[:-1]).sum())
                row["frames"] = {k: (frames[k][i].clamp(0, 1) * 255).byte().cpu().numpy() for k in ("gt", "imagined")}
                rows.append(row)
                if segment == "lift" and int(h) in video_holds:
                    n = int(real_len[e])
                    tag = ""
                    if "meta/ball_slot" in batch:  # the ball changes slot once per swap
                        slots = batch["meta/ball_slot"][i, :n]
                        tag = f"start{int(slots[0])}_{row['swaps']}swap"
                    video_clips.append({"hold": int(h), "label": bool(label[i]), "context": ctx,
                                        "episode": int(e), "tag": tag,
                                        "predicted": row["imagined_ball"],
                                        "real": arrays.image[int(e), :n],
                                        "imagined": row["frames"]["imagined"],
                                        "action": arrays.action[int(e), :n, 0],
                                        "phase": arrays.meta["phase"][int(e), :n]})

    metrics = {seg: {k: binary_metrics([r[f"{k}_ball"] for r in rows if r["segment"] == seg],
                                       [r["label"] for r in rows if r["segment"] == seg]) for k in SOURCES}
               for seg in sorted({r["segment"] for r in rows})}
    (outdir / "metrics.json").write_text(json.dumps({
        "decoder_checkpoint": str(Path(cfg.decoder_checkpoint).resolve()), "decoder_step": int(dec_state["step"]),
        "wm_checkpoint": dec_state["wm_checkpoint"], "wm_step": int(wm_state["step"]),
        "eval_zarr": str(Path(cfg.zarr_path).resolve()), "episodes": int(len(np.unique(hold)) * 2),
        "config": OmegaConf.to_container(cfg, resolve=True), "metrics": metrics}, indent=2))

    fields = [k for k in rows[0] if k != "frames"]
    with (outdir / "per_episode.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)

    lift = sorted([r for r in rows if r["segment"] == "lift"], key=lambda r: (r["hold"], not r["label"]))
    sheet_rows = [{**r, **r["frames"]} for r in lift]
    title = f"memtest1: watched through the hold, imagined the lift  |  F1 {metrics['lift']['imagined']['f1']:.3f}"
    max_rows = int(cfg.sheet_max_rows)
    if len(sheet_rows) > max_rows:  # keep the sheet a readable size over long hold ranges
        keep = np.linspace(0, len(sheet_rows) - 1, max_rows).round().astype(int)
        sheet_rows = [sheet_rows[i] for i in keep]
        title += f"  ({max_rows} of {len(lift)} episodes shown)"
    draw_sheet(outdir / "sheet_lift.png", title, sheet_rows)
    draw_hold_chart(outdir / "outcome_by_hold.png", lift, metrics["lift"]["imagined"]["f1"])

    if video_clips:
        video_dir = outdir / "videos"
        video_dir.mkdir(exist_ok=True)
        for clip in sorted(video_clips, key=lambda c: (c["hold"], c.get("tag", ""), not c["label"])):
            tag = f"_{clip['tag']}" if clip.get("tag") else ""
            name = (f"hold{clip['hold']:02d}{tag}_{'ball' if clip['label'] else 'empty'}"
                    f"_ep{clip.get('episode', 0):03d}.mp4")
            write_video(video_dir / name, clip, cfg.videos.fps)
        print(f"{len(video_clips)} rollout videos -> {video_dir}")

    print(f"\n{len(lift)} episodes, holds {int(hold.min())}-{int(hold.max())} | world model step {wm_state['step']}, "
          f"decoder step {dec_state['step']} | {outdir}")
    print(f"{'segment':8s} {'source':9s} {'n':>3s} {'TP':>3s} {'FP':>3s} {'TN':>3s} {'FN':>3s} "
          f"{'prec':>6s} {'recall':>6s} {'F1':>6s} {'acc':>6s}")
    for seg, by_source in metrics.items():
        for k, m in by_source.items():
            print(f"{seg:8s} {k:9s} {m['n']:3d} {m['tp']:3d} {m['fp']:3d} {m['tn']:3d} {m['fn']:3d} "
                  f"{m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['accuracy']:6.3f}")
    wrong = [r for r in lift if r["imagined_ball"] != r["label"]]
    print(f"lift mistakes: {len(wrong)}/{len(lift)} at holds {[r['hold'] for r in wrong]}")


if __name__ == "__main__":
    main()
