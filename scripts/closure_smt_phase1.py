"""PHASE 1 gate 4: does the posterior chain stay on the teacher's manifold? Run this FIRST.

The eval maintains the belief recurrently across ~40 real steps before it imagines anything, so
if the chained state drifts off the teacher's encoding over that span nothing downstream is
interpretable -- the pixels would be rendering a state that no longer means anything.

Two chains from every step t, compared against the teacher's own encoding of the window at t+h:

    posterior  fed the REAL embedding at each step -> the observing half of the eval
    prior      fed the action only                 -> the imagining half

The posterior curve is the one gate 4 is about. The prior curve is reported next to it because
its knee is the imagination horizon, and the eval only ever asks for 10 imagined steps.

    pixi run closure-smt-phase1 wm_run=<smt_phase1 run dir>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from info_gathering.memtest4 import hold_base  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

GEOMETRY = OmegaConf.create({"move_frames": 4, "swap_frames": 6, "swap_gap": 0})


@torch.no_grad()
def closure(model, batch, teacher_z, embed, horizon, real_len, mode):
    """Mean squared distance between the chained state at t+h and the teacher's encoding there."""
    steps = teacher_z.shape[1]
    errors = {h: [] for h in range(1, horizon + 1)}
    for t in range(steps - 1):
        z = teacher_z[:, t]
        for h in range(1, horizon + 1):
            if t + h >= steps:
                break
            action = batch["prev_action"][:, t + h].float()     # the action leading into t+h
            z = (model.posterior.step(z, embed[:, t + h], action) if mode == "posterior"
                 else model.prior.step(z, action)).float()
            valid = (torch.as_tensor(t + h) < real_len).float()
            if float(valid.sum()) == 0:
                break
            err = (z - teacher_z[:, t + h]).pow(2).mean(dim=(-1, -2))
            errors[h].append(float((err * valid).sum() / valid.sum()))
    return {h: float(np.mean(v)) for h, v in errors.items() if v}


@hydra.main(version_base=None, config_path="../configs/smt_phase1", config_name="closure_phase1")
def main(cfg: DictConfig):
    state = CheckpointManager.load(cfg.wm_checkpoint)
    wm_cfg = OmegaConf.create(state["config"])
    train_set, _ = instantiate(wm_cfg.data)
    model = instantiate(wm_cfg.model, action_dim=train_set.action_dim, device=cfg.device, _recursive_=False)
    model.load_train_state_dict(state["model"])
    model.requires_grad_(False).eval()

    eval_set, _ = instantiate(wm_cfg.data, zarr_path=cfg.zarr_path, holdout_episodes=0)
    arrays = eval_set.arrays
    real_len = torch.from_numpy((arrays.steps - arrays.meta["pad"].sum(1)).astype("int64"))
    picked = np.linspace(0, len(eval_set) - 1, int(cfg.episodes)).astype(int)
    batch = to_device(default_collate([eval_set[int(i)] for i in picked]), cfg.device)
    lengths = real_len[picked].to(cfg.device)

    # the teacher's encoding of the window ending at every step: the manifold the chain must stay on
    embed = model._embed(batch["image"])
    teacher_z = model.encode_windows(embed, batch["prev_action"].float(), batch["is_first"])
    variance = float(teacher_z.var())

    print(f"wm step {state['step']} | {len(picked)} episodes | eval_warmup {model.eval_warmup} "
          f"| teacher z variance {variance:.5f}\n")
    curves = {mode: closure(model, batch, teacher_z, embed, int(cfg.horizon), lengths, mode)
              for mode in ("posterior", "prior")}
    print(f"{'h':>3} {'posterior mse':>15} {'posterior r2':>14} {'prior mse':>12} {'prior r2':>10}")
    for h in sorted(curves["posterior"]):
        p, o = curves["posterior"][h], curves["prior"].get(h, float("nan"))
        print(f"{h:>3} {p:>15.5f} {1 - p / variance:>14.4f} {o:>12.5f} {1 - o / variance:>10.4f}")

    # the span the eval actually demands: every real step from the warm-up to the lift onset
    import zarr as zarr_mod
    z = zarr_mod.open_group(str(ROOT / cfg.zarr_path), mode="r")
    swaps, holds = z["episode_swapped"][:], z["episode_hold"][:]
    slack = np.array([h - hold_base(GEOMETRY, n) for h, n in zip(holds, swaps)])
    onset = np.array([int(e) for e in holds]) + 10          # reveal is 10 steps, then the hold
    print(f"\neval slack (static frames between the last swap and the lift): "
          f"min {slack.min()} median {int(np.median(slack))} max {slack.max()}")
    print(f"posterior steps the eval chains before the lift (onset - eval_warmup): "
          f"min {onset.min() - model.eval_warmup} median {int(np.median(onset)) - model.eval_warmup} "
          f"max {onset.max() - model.eval_warmup}")
    print(f"fraction of episodes needing more than {cfg.horizon} chained steps: "
          f"{float(((onset - model.eval_warmup) > int(cfg.horizon)).mean()):.3f}")

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "closure.json").write_text(json.dumps({
        "wm_checkpoint": str(Path(cfg.wm_checkpoint).resolve()), "wm_step": int(state["step"]),
        "teacher_variance": variance, "eval_warmup": int(model.eval_warmup),
        "curves": {m: {str(h): v for h, v in c.items()} for m, c in curves.items()},
        "config": OmegaConf.to_container(cfg, resolve=True)}, indent=2))
    print(f"\n-> {outdir / 'closure.json'}")


if __name__ == "__main__":
    main()
