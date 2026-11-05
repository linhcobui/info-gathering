"""PHASE 1 gate 5: the ceiling the imagined rollout is measured against.

eval_memtest1.py scores two sources that bracket the model from the outside -- `recon` (real
embeddings straight through the pixel head, the decoder's own ceiling) and `imagined` (the
readout of PRIOR states, the test). Neither isolates the readout, so a low `imagined` F1 could
mean either "the state lost the ball" or "the readout cannot render a state that still has it".

This script closes that gap by scoring the readout on states the model holds while OBSERVING,
where the answer is known to be present:

    recon           real embedding                      -> pixels     the decoder ceiling
    readout_post    readout(posterior-chained state)    -> pixels     the eval's observing half
    readout_teacher readout(teacher window encoding)    -> pixels     docs/SMT_PHASE1.md gate 5

READ THESE AS A RENDERING CEILING AND NOTHING MORE. Every row except `imagined` is scored over
show offsets 1-9, i.e. the frames where the cup is up and the ball is visible -- and both the
posterior and the teacher have ALREADY OBSERVED those frames at the steps their states are read
from. readout_post and readout_teacher are therefore reconstructions routed through the readout.
They answer "can the readout render a state that has seen the ball", which bounds the rendering
path. They say NOTHING about whether the state carried the answer through the hold, and a high
score here is not evidence about memory.

To ask whether the carried state holds the answer, read it BEFORE the cup moves:
scripts/onset_probe_phase1.py reads the reward head at the lift onset through read-outs that see
no observation at all. That is the measurement that separates the encoder, the posterior chain
and the prior.

    pixi run readout-ceiling-phase1 decoder_run=<pixel_decoder_phase1 run>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from info_gathering.ball_detector import ball_pixels  # noqa: E402
from info_gathering.dataset_memtest1 import memtest1_datasets  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402
from world_model.training.rollout import action_onsets  # noqa: E402
from world_model.utils.checkpoint import load_frozen_models  # noqa: E402
from world_model.utils.classification import binary_metrics  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402

SOURCES = ("gt", "recon", "readout_post", "readout_teacher", "imagined")


@hydra.main(version_base=None, config_path="../configs/smt_phase1", config_name="readout_ceiling_phase1")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model, decoder, _, _, dec_state, wm_state = load_frozen_models(cfg.decoder_checkpoint, cfg.device)
    wm_cfg = OmegaConf.create(wm_state["config"])

    eval_set, _ = memtest1_datasets(cfg.zarr_path, image_size=wm_cfg.model.image_size, holdout_episodes=0,
                                    meta_keys=tuple(cfg.meta_keys))
    arrays = eval_set.arrays
    real_len = arrays.steps - arrays.meta["pad"].sum(1)
    hold = real_len - int(cfg.hold_offset)
    threshold, min_pixels = float(cfg.detector.threshold), int(cfg.detector.min_pixels)

    rows = []
    for h in np.unique(hold):                      # group by hold so a batch shares its onsets
        episodes = np.flatnonzero(hold == h)
        batch = to_device(default_collate([eval_set[int(e)] for e in episodes]), cfg.device)

        model.observe_mode = "recurrent"
        post = model.observe(batch)                # warm-up + posterior chain: the eval's own states
        model.observe_mode = "window"
        teacher = model.observe(batch)             # a fresh teacher window at every step
        model.observe_mode = "recurrent"

        onset = action_onsets(batch["action"])[1]  # the lift; onset 1 is the cover, no ball question
        ctx = onset + 1
        end = min(int(real_len[episodes[0]]), ctx + int(cfg.horizon))
        imagined = model.imagine(post[:, :ctx], batch["prev_action"][:, ctx:end])

        def render_readout(latents):
            """The SAME pixel head, fed readout(z) instead of the real embedding."""
            z = latents["z"][:, ctx:end].float()
            return decoder.render(TensorDict({"embed": model.readout(z).float()},
                                             batch_size=tuple(z.shape[:2])))

        frames = {
            "gt": batch["image"][:, ctx:end].float() / 255.0,
            "recon": decoder.render(post[:, ctx:end]),
            "readout_post": render_readout(post),
            "readout_teacher": render_readout(teacher),
            "imagined": decoder.render(imagined),
        }
        pixels = {k: ball_pixels(v, threshold) for k, v in frames.items()}
        label = batch[f"meta/{cfg.label_key}"][:, onset].bool()
        for i, e in enumerate(episodes):
            row = {"episode": int(e), "hold": int(h), "label": bool(label[i])}
            for k in SOURCES:
                row[f"{k}_ball"] = bool((pixels[k][i] >= min_pixels).any())
            if "meta/ball_slot" in batch:
                slots = batch["meta/ball_slot"][i, :int(real_len[e])]
                row["swaps"] = int((slots[1:] != slots[:-1]).sum())
            rows.append(row)

    labels = [r["label"] for r in rows]
    metrics = {k: binary_metrics([r[f"{k}_ball"] for r in rows], labels) for k in SOURCES}

    # per-episode rows, so the by-swap split can be joined on episode_swapped in the zarr.
    # NOTE the `swaps` column here counts BALL-slot transitions, not cup swaps -- with 3 cups a
    # random pair leaves the ball alone ~1/3 of the time (docs/SMT_PRELIM.md 8). Join on the zarr for
    # the cup-swap count, exactly as scripts/slack_split_prelim.py does.
    import csv as _csv
    fields = list(rows[0])
    with (outdir / "per_episode.csv").open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (outdir / "ceiling.json").write_text(json.dumps({
        "decoder_checkpoint": str(Path(cfg.decoder_checkpoint).resolve()),
        "wm_step": int(wm_state["step"]), "eval_zarr": str(Path(cfg.zarr_path).resolve()),
        "n": len(rows), "metrics": metrics,
        "config": OmegaConf.to_container(cfg, resolve=True)}, indent=2))

    print(f"\n{len(rows)} lift episodes | wm step {wm_state['step']} | decoder step {dec_state['step']}")
    print(f"{'source':17s} {'n':>4s} {'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s} {'F1':>7s} "
          f"{'acc':>7s} {'bal_acc':>8s} {'MCC':>7s}")
    for k in SOURCES:
        m = metrics[k]
        print(f"{k:17s} {m['n']:4d} {m['tp']:4d} {m['fp']:4d} {m['tn']:4d} {m['fn']:4d} "
              f"{m['f1']:7.3f} {m['accuracy']:7.3f} {m['balanced_accuracy']:8.3f} {m['mcc']:7.3f}")
    print(f"\n-> {outdir / 'ceiling.json'}")


if __name__ == "__main__":
    main()
