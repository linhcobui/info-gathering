"""Does an error metric track events or just how close things are to the camera? (diagnostic, privileged scoring)

For each metric, on held-out episodes, with lambda = its mean over all held-out steps (lambda_stat: mean):
    rho_*       Spearman correlation with: gripper y (+ = towards the side camera), bar y, the copy-last scene change
                (copy_l2_mean), the bar's tilt change over the step, grasped
    near/far    fraction of steps above lambda in the nearest / farthest fifth of gripper y
    lift/hold/lower   fraction above lambda in those probe stages (hold should be low: the tilt is in the context)
Privileged keys are read only here, to choose a metric.

    pixi run python scripts/seg_metric_compare.py --metrics err_l2_mean err_l2_changed err_cls err_cls_cos
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import zarr  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from seg.config import load_config, out_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", default=["err_l2_mean", "err_l2_changed", "err_cls", "err_cls_cos"])
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(overrides=args.overrides)
    err = pd.read_parquet(out_dir(cfg) / "wm_errors.parquet")
    err = err[~err.wm_train_episode].reset_index(drop=True)
    store = next(s for s in cfg.stores if not s.wm_train)
    z = zarr.open_group(str(ROOT / store.zarr), mode="r")
    a = dict(z.attrs)
    f1 = err.target_raw_frame.to_numpy()
    f0 = f1 - cfg.frameskip
    tilt = z["bar_tilt"][:]
    priv = dict(eff_y=z["effector_pos"][:, 1][f1], bar_y=z["bar_pos"][:, 1][f1], scene_change=err.copy_l2_mean.to_numpy(),
                dtilt=np.abs(tilt[f1] - tilt[f0]), grasped=z["grasped"][:][f1].astype(float))
    seg = np.array(a["play_segments"])[z["segment_type"][:][f1]]
    stage = np.array(a["probe_stages"])[z["probe_stage"][:][f1]]
    q = pd.qcut(priv["eff_y"], 5, labels=False)
    rows = []
    for m in args.metrics:
        v = err[m].to_numpy(dtype=np.float64)
        lam = np.nanmean(v)
        y = v > lam
        r = dict(metric=m, lam=lam, frac_above=y.mean())
        r.update({f"rho_{k}": spearmanr(v, x, nan_policy="omit").statistic for k, x in priv.items()})
        r.update(far=y[q == 0].mean(), near=y[q == 4].mean())
        for st in ("lift", "hold", "lower"):
            r[st] = y[(seg == "probe") & (stage == st)].mean()
        rows.append(r)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"held-out: {err.episode_id.nunique()} eps, {len(err)} steps; lambda = mean of each metric")
    print(df.round(3).to_string(index=False))
    df.to_csv(out_dir(cfg) / "diagnostics" / "metric_compare.csv", index=False)


if __name__ == "__main__":
    main()
