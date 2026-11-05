"""PHASE 1: does the ONSET state hold the ball answer? The measurement that localizes the failure.

The pixel eval reports `imagined` (prior-chained from the onset) against `recon` / `readout_*`
ceilings -- but every ceiling row renders states taken AT the lift frames, which the posterior
and the teacher have already observed. Those rows bound the rendering path and say nothing
about the hold. This script asks the question they cannot:

    at the lift onset, BEFORE the cup has moved, does the carried state contain the answer?

It reads the reward head at the onset state. Per docs/SMT_PHASE1.md 2.2 the read-out r_{t+1} sits at
the a_t token and sees `memory + a_t` in totality -- no observation at all -- so it is a clean
probe of the state. (r_t at z_{K-1} is memory-only and is reported alongside.) The label is
ball_under_target, constant through the lift.

Three state sources, so the chain and the encoder are separable:

    teacher    the teacher's window encoding at the onset      -- is it in the ENCODER?
    posterior  warm up once, chain the posterior to the onset  -- did the CHAIN keep it?
    prior      chain the posterior to the onset, then prior 9x -- did IMAGINATION lose it?

Read against the pixel eval's `imagined` row by swap count. If posterior is high at 3-4 swaps
and prior is not, the prior updater is the constraint. If posterior is also low, the encoder or
the chain is, and docs/SMT_PHASE1.md 7's expectation stands.

    pixi run onset-probe-phase1 wm_run=<smt_phase1 run dir>
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from info_gathering.dataset_memtest1 import memtest1_datasets  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402
from world_model.training.rollout import action_onsets  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402
from world_model.utils.classification import binary_metrics  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402

SOURCES = ("teacher", "posterior", "prior")


@torch.no_grad()
def read_reward(model, z, embed, action, onset, fut_len, real_len):
    """Reward logits from the state at `onset`. Returns (r_t, r_t+1) -- both observation-free.

    e_fut / a_fut are supplied exactly as the anchor dataset builds them (steps onset..onset+F-1,
    right-clamped to the last real step). They only affect read-outs from r_{t+2} onward, which
    this function does not return -- they are passed so the decoder runs on its training shape.
    """
    index = np.clip(np.arange(onset, onset + fut_len), 0, int(real_len) - 1)
    e_fut = embed[:, index]                 # e_t.. : the same frames in both decoder layouts
    a_fut = action[:, index].float()
    raw = model.reward_decoder(z, e_fut, a_fut)
    logits = model.reward_decoder.ball_logit(raw)
    slot = raw[:, 1].argmax(-1) if raw.dim() == 3 else None
    return logits[:, 0], logits[:, 1], slot


@hydra.main(version_base=None, config_path="../configs/smt_phase1", config_name="onset_probe_phase1")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    state = CheckpointManager.load(cfg.wm_checkpoint)
    wm_cfg = OmegaConf.create(state["config"])
    train_set, _ = instantiate(wm_cfg.data)
    model = instantiate(wm_cfg.model, action_dim=train_set.action_dim, device=cfg.device, _recursive_=False)
    model.load_train_state_dict(state["model"])
    model.requires_grad_(False).eval()
    fut_len = int(wm_cfg.model.fut_len)

    eval_set, _ = memtest1_datasets(cfg.zarr_path, image_size=wm_cfg.model.image_size, holdout_episodes=0,
                                    meta_keys=tuple(cfg.meta_keys))
    arrays = eval_set.arrays
    real_len = arrays.steps - arrays.meta["pad"].sum(1)
    hold = real_len - int(cfg.hold_offset)

    rows = []
    for h in np.unique(hold):
        episodes = np.flatnonzero(hold == h)
        batch = to_device(default_collate([eval_set[int(e)] for e in episodes]), cfg.device)
        onset = action_onsets(batch["action"])[1]          # the lift; the cup has NOT yet moved
        n = int(real_len[episodes[0]])

        model.observe_mode = "recurrent"
        post = model.observe(batch)                        # warm-up + posterior chain to the onset
        model.observe_mode = "window"
        teacher = model.observe(batch)                     # a fresh teacher window at the onset
        model.observe_mode = "recurrent"
        embed = post["embed"]

        z = {"teacher": teacher["z"][:, onset].float(), "posterior": post["z"][:, onset].float()}
        # the imagined state: the same 9 prior steps the pixel eval runs, then read at the end
        z_prior = z["posterior"]
        for i in range(onset + 1, min(n, onset + 1 + int(cfg.horizon))):
            z_prior = model.prior.step(z_prior, batch["prev_action"][:, i].float()).float()
        z["prior"] = z_prior

        label = batch[f"meta/{cfg.label_key}"][:, onset].bool()
        logits = {k: read_reward(model, v, embed, batch["action"], onset, fut_len, n)
                  for k, v in z.items()}
        for i, e in enumerate(episodes):
            row = {"episode": int(e), "hold": int(h), "onset": int(onset), "label": bool(label[i]),
                   "slot": int(batch["meta/ball_slot"][i, onset])}
            for k in SOURCES:
                row[f"{k}_r0"] = bool(logits[k][0][i] > 0)      # memory only
                row[f"{k}_r1"] = bool(logits[k][1][i] > 0)      # memory + a_t, no observation
                if logits[k][2] is not None:                    # slot head: which slot at r_t+1
                    row[f"{k}_slot"] = int(logits[k][2][i])
            rows.append(row)

    z_eval = zarr.open_group(str(ROOT / cfg.zarr_path), mode="r")
    swaps = z_eval["episode_swapped"][:][np.array([r["episode"] for r in rows])]
    for r, s in zip(rows, swaps):
        r["cup_swaps"] = int(s)

    with (outdir / "per_episode.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = np.array([r["label"] for r in rows])
    print(f"\n{len(rows)} episodes | wm step {state['step']} | eval_warmup {model.eval_warmup} | "
          f"positive rate {labels.mean():.3f}")
    print("reward head read at the LIFT ONSET, before the cup moves. r1 sees memory + one action,\n"
          "no observation; r0 sees memory alone. Neither is a reconstruction.\n")
    summary = {}
    for head in ("r0", "r1"):
        print(f"--- {head} ({'memory only' if head == 'r0' else 'memory + a_t'}), F1 by cup swaps")
        print(f"{'state':<12}" + "".join(f"{str(k) + ' sw':>9}" for k in range(5)) + f"{'all':>9}")
        for source in SOURCES:
            pred = np.array([r[f"{source}_{head}"] for r in rows])
            cells = "".join(f"{binary_metrics(pred[swaps == k], labels[swaps == k])['f1']:>9.3f}"
                            for k in range(5))
            overall = binary_metrics(pred, labels)
            summary[f"{source}_{head}"] = overall
            print(f"{source:<12}{cells}{overall['f1']:>9.3f}")
        print()
    (outdir / "onset_probe.json").write_text(json.dumps({
        "wm_checkpoint": str(Path(cfg.wm_checkpoint).resolve()), "wm_step": int(state["step"]),
        "n": len(rows), "metrics": summary,
        "config": OmegaConf.to_container(cfg, resolve=True)}, indent=2))
    print(f"-> {outdir / 'onset_probe.json'}")


if __name__ == "__main__":
    main()
