"""Is the answer in the RSSM latent, independent of the decoder?

Fits a ridge probe on frozen posterior latents and reports accuracy, optionally training on one
group of episodes and testing on another (e.g. fit on 0/1-swap episodes, test on 2-swap ones).
The decoder is never used, so this separates "the world model lost the information" from
"the decoder renders it wrongly" -- a distinction the detector-based eval cannot make.

    pixi run probe-latent --decoder-run <run> --zarr data/memtest3_eval.zarr
    pixi run probe-latent --decoder-run <run> --zarr data/memtest2_eval.zarr --fit-groups 0 1
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from info_gathering.dataset_memtest2 import memtest2_datasets  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402
from world_model.training.rollout import action_onsets  # noqa: E402
from world_model.utils.checkpoint import load_frozen_models  # noqa: E402


def ridge_probe(X, y, fit, alpha):
    """Fit sign(X w) on the `fit` rows; return predictions for every row."""
    X = (X - X[fit].mean(0)) / (X[fit].std(0) + 1e-6)
    X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = X[fit].T @ X[fit] + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, X[fit].T @ y[fit])
    return np.sign(X @ w)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decoder-run", required=True, help="a train_decoder.py run directory")
    p.add_argument("--checkpoint", default=None, help="defaults to <decoder-run>/checkpoints/decoder_latest.pt")
    p.add_argument("--zarr", required=True, help="eval zarr to probe")
    p.add_argument("--label-key", default="ball_under_target", help="meta key, read at the lift onset")
    p.add_argument("--fit-groups", type=int, nargs="*", default=None,
                   help="swap counts to fit on (default: all, reported as leave-one-group-out)")
    p.add_argument("--alpha", type=float, default=5.0, help="ridge regularisation")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    checkpoint = args.checkpoint or f"{args.decoder_run}/checkpoints/decoder_latest.pt"
    model, _, _, _, dec, wm = load_frozen_models(checkpoint, args.device)
    eval_set, _ = memtest2_datasets(args.zarr, image_size=64, holdout_episodes=0,
                                    meta_keys=(args.label_key, "ball_slot", "phase"))
    arrays = eval_set.arrays
    real_len = arrays.steps - arrays.meta["pad"].sum(1)
    swaps_recorded = zarr.open_group(args.zarr, mode="r")["episode_swapped"][:]

    onset_feats, lift_feats, labels = [], [], []
    for e in range(len(eval_set)):
        batch = to_device(default_collate([eval_set[e]]), args.device)
        post = model.observe(batch)
        onset = action_onsets(batch["action"])[-1]          # the lift, after the hold
        feat = model.agent.rssm.get_feat(post["stoch"], post["deter"])[0].float().cpu().numpy()
        onset_feats.append(feat[onset])
        lift_feats.append(feat[int(real_len[e]) - 1])
        labels.append(int(batch[f"meta/{args.label_key}"][0, onset]))

    y = np.where(np.array(labels) == 1, 1.0, -1.0)
    groups = np.asarray(swaps_recorded)
    print(f"world model step {wm['step']}, decoder step {dec['step']} | {len(y)} episodes | "
          f"groups (swaps): {dict(zip(*[x.tolist() for x in np.unique(groups, return_counts=True)]))}")

    for where, feats in (("onset", onset_feats), ("lift", lift_feats)):
        X = np.stack(feats)
        splits = ([(np.isin(groups, args.fit_groups), f"fit on swaps {args.fit_groups}")] if args.fit_groups
                  else [(groups != g, f"leave out swaps {g}") for g in np.unique(groups)])
        for fit, name in splits:
            pred = ridge_probe(X, y, fit, args.alpha)
            per_group = " ".join(f"{g}:{int((pred[groups == g] == y[groups == g]).sum())}/{int((groups == g).sum())}"
                                 for g in np.unique(groups))
            print(f"[{where:5s}] {name:22s} fit {(pred[fit] == y[fit]).mean():.3f} | "
                  f"held out {(pred[~fit] == y[~fit]).mean():.3f} | per swap count {per_group}")


if __name__ == "__main__":
    main()
