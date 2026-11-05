"""PRELIMINARY: the F1 split the smt_prelim experiment is actually about.

slack = hold - hold_base(n_swaps) is the number of static frames between the last swap and the
lift -- i.e. how far the decisive information must travel through the chained one-step predictor
rather than sitting inside the teacher's ctx_len window.

    slack <= ctx_len-1 : the last swap is still in the teacher's window; the teacher alone can answer
    slack >  ctx_len-1 : only the predictor chain carries it

An aggregate F1 cannot distinguish 0.95/0.45 from 0.72/0.72, so this split is the result.

    pixi run python scripts/slack_split_prelim.py <per_episode.csv> <eval.zarr> [ctx_len]
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import zarr  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from info_gathering.memtest4 import hold_base  # noqa: E402
from world_model.utils.classification import binary_metrics  # noqa: E402

GEOMETRY = OmegaConf.create({"move_frames": 4, "swap_frames": 6, "swap_gap": 0})


def counts(label, pred):
    m = binary_metrics(pred, label)
    return m["tp"], m["fp"], m["tn"], m["fn"], m["f1"], m["accuracy"], m["balanced_accuracy"], m["mcc"]


def main(csv_path, zarr_path, ctx_len=10):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["segment"] == "lift"]
    z = zarr.open_group(str(zarr_path), mode="r")
    swaps_all, hold_all = z["episode_swapped"][:], z["episode_hold"][:]

    episode = np.array([int(r["episode"]) for r in rows])
    label = np.array([r["label"] == "True" for r in rows])
    swaps = swaps_all[episode]
    slack = np.array([h - hold_base(GEOMETRY, n) for h, n in zip(hold_all[episode], swaps)])
    sources = [s for s in ("gt", "recon", "imagined") if f"{s}_ball" in rows[0]]
    preds = {s: np.array([r[f"{s}_ball"] == "True" for r in rows]) for s in sources}

    print(f"{len(rows)} lift episodes | ctx_len {ctx_len} | positives {label.mean():.3f}")
    print(f"slack: min {slack.min()} median {int(np.median(slack))} max {slack.max()}\n")

    for source in sources:
        print(f"--- {source}")
        print(f"{'group':>22} {'n':>5} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} {'F1':>8} {'acc':>8}"
              f" {'bal_acc':>8} {'MCC':>8}")
        groups = [(f"slack <= {ctx_len - 1} (in window)", slack <= ctx_len - 1),
                  (f"slack >  {ctx_len - 1} (chained)", slack > ctx_len - 1)]
        groups += [(f"{n} swaps", swaps == n) for n in sorted(set(swaps.tolist()))]
        groups += [("all", np.ones(len(rows), bool))]
        for name, sel in groups:
            if not sel.any():
                continue
            tp, fp, tn, fn, f1, acc, bal, mcc = counts(label[sel], preds[source][sel])
            print(f"{name:>22} {int(sel.sum()):>5} {tp:>5} {fp:>5} {tn:>5} {fn:>5} {f1:>8.3f} {acc:>8.3f}"
                  f" {bal:>8.3f} {mcc:>8.3f}")
        print()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
