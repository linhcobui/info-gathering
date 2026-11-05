"""PRELIMINARY gate D: how far can the one-step cell chain before it leaves the teacher's state?

Rolls the predictor forward h = 1..H steps from every anchor and compares its state against the
teacher's encoding at t+h. Two modes separate the two ways a rollout can go wrong:

    posterior  the cell is fed the REAL embedding at each step -> pure state-chaining error
    open       the decoder predicts the next embedding and the cell consumes that -> what
               imagine() actually does at eval

The knee of the open-loop curve is the memory horizon. Compared against the eval set's slack
distribution (the static frames between the last swap and the lift), it says in advance whether
the end-to-end eval can succeed.

    pixi run closure-smt-prelim wm_run=<run dir>
"""
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


@torch.no_grad()
def closure(model, batch, horizon, real_len, open_loop):
    post = model.observe(batch)
    deter, logit = post["deter"].float(), post["logit"].float()
    stoch, embed = post["stoch"].float(), post["embed"].float()
    steps = deter.shape[1]
    errors = {h: [] for h in range(1, horizon + 1)}

    for t in range(steps - 1):
        d, l, s = deter[:, t], logit[:, t], stoch[:, t]
        e = embed[:, t]
        for h in range(1, horizon + 1):
            if t + h >= steps:
                break
            action = batch["prev_action"][:, t + h].float()      # the action leading into t+h
            e = (model.teacher_dec.decode_step(d, l, e, action).float() if open_loop
                 else embed[:, t + h])
            _, d, l = model.predictor.step(s, d, action, e)
            d, l = d.float(), l.float()
            s = model._stoch(l)
            valid = (torch.as_tensor(t + h) < real_len).float()
            if float(valid.sum()) == 0:
                break
            err = (d - deter[:, t + h]).pow(2).mean(-1)
            errors[h].append(float((err * valid).sum() / valid.sum()))
    return {h: float(np.mean(v)) for h, v in errors.items() if v}


@hydra.main(version_base=None, config_path="../configs/smt_prelim", config_name="closure")
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

    variance = float(model.observe(batch)["deter"].float().var())
    print(f"wm step {state['step']} | {len(picked)} episodes | deter variance {variance:.4f}\n")
    print(f"{'h':>3} {'posterior mse':>15} {'posterior r2':>14} {'open mse':>12} {'open r2':>10}")
    curves = {}
    for mode in ("posterior", "open"):
        curves[mode] = closure(model, batch, int(cfg.horizon), lengths, mode == "open")
    for h in sorted(curves["posterior"]):
        p, o = curves["posterior"][h], curves["open"].get(h, float("nan"))
        print(f"{h:>3} {p:>15.5f} {1 - p / variance:>14.4f} {o:>12.5f} {1 - o / variance:>10.4f}")

    # the slack the eval will actually demand of this curve
    zarr_path = ROOT / cfg.zarr_path
    import zarr as zarr_mod
    z = zarr_mod.open_group(str(zarr_path), mode="r")
    swaps = z["episode_swapped"][:]
    holds = z["episode_hold"][:]
    data_cfg = OmegaConf.create({"move_frames": 4, "swap_frames": 6, "swap_gap": 0})
    slack = np.array([h - hold_base(data_cfg, n) for h, n in zip(holds, swaps)])
    print(f"\neval slack (static frames between the last swap and the lift): "
          f"min {slack.min()} median {int(np.median(slack))} max {slack.max()}")
    print(f"fraction needing more than {cfg.horizon} chained steps: "
          f"{float((slack > int(cfg.horizon)).mean()):.3f}")


if __name__ == "__main__":
    main()
