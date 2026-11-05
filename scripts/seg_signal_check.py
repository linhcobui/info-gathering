"""Which error signal has peaks at bar events rather than a slow arm-motion level? (diagnostic, privileged scoring)

For every (error metric x normalisation) candidate, on the held-out episodes, with lambda = the lambda_quantile of the
candidate over all held-out steps and y = 1[value > lambda]:
    runs/ep         distinct runs of y = 1 per episode ("peaks"), and their mean length (steps)
    rho_bar         Spearman correlation of the value with |change in bar tilt| over the predicted step (privileged)
    rho_arm         ... with the gripper's speed over that step (privileged)
    precision       fraction of runs that meet a bar event (|tilt change| > event_deg in one step) within +-tol steps
    recall          fraction of bar events met by a run within +-tol steps
Privileged keys are read only here, to choose the signal; the pipeline never reads them.

    pixi run python scripts/seg_signal_check.py
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
from seg.threshold import normalised_metric  # noqa: E402


def runs(y):
    """(start, end) inclusive index runs of y == 1."""
    d = np.diff(np.concatenate([[0], y.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-deg", type=float, default=5.0)
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(overrides=args.overrides)
    out = out_dir(cfg)
    err = pd.read_parquet(out / "wm_errors.parquet")
    err = err[~err.wm_train_episode].sort_values(["episode_id", "t"]).reset_index(drop=True)
    store = next(s for s in cfg.stores if not s.wm_train)
    z = zarr.open_group(str(ROOT / store.zarr), mode="r")
    tilt, eff = z["bar_tilt"][:].astype(np.float64), z["effector_pos"][:].astype(np.float64)
    f1 = err.target_raw_frame.to_numpy()                      # frame t+1
    f0 = f1 - cfg.frameskip                                    # frame t
    dtilt = np.abs(tilt[f1] - tilt[f0])
    arm = np.linalg.norm(eff[f1] - eff[f0], axis=1)
    event = np.degrees(dtilt) > args.event_deg
    err["copy_ratio_topk"] = err.err_l2_topk / err.copy_l2_topk.clip(lower=1e-3)
    cands = [(m, n) for m in ("err_l2_mean", "err_l2_topk", "err_l2_max", "err_cos_mean", "copy_ratio_topk", "copy_l2_topk")
             for n in ("global", "episode", "detrend")]
    rows = []
    ep = err.episode_id.to_numpy()
    bounds = np.flatnonzero(np.concatenate([[True], ep[1:] != ep[:-1], [True]]))
    for m, n in cands:
        v = normalised_metric(err, m, n, cfg.detrend_window)
        lam = np.quantile(v, cfg.lambda_quantile)
        y = v > lam
        n_runs, lens, hit_runs, ev_tot, ev_hit = 0, [], 0, 0, 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            r = runs(y[a:b])
            evs = np.flatnonzero(event[a:b])
            n_runs += len(r)
            lens += [e - s + 1 for s, e in r]
            hit_runs += sum(any(s - args.tol <= x <= e + args.tol for x in evs) for s, e in r)
            ev_tot += len(evs)
            ev_hit += sum(any(s - args.tol <= x <= e + args.tol for s, e in r) for x in evs)
        rows.append(dict(metric=m, normalise=n, runs_per_ep=n_runs / (len(bounds) - 1), mean_run_len=np.mean(lens),
                         rho_bar=spearmanr(v, dtilt).statistic, rho_arm=spearmanr(v, arm).statistic,
                         precision=hit_runs / max(n_runs, 1), recall=ev_hit / max(ev_tot, 1)))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"held-out: {err.episode_id.nunique()} eps, {len(err)} steps; bar events (|tilt change| > {args.event_deg} deg "
          f"in one step): {event.sum()} ({event.mean():.3f} of steps); chance precision ~ "
          f"{np.mean([event[max(0, i - args.tol):i + args.tol + 1].any() for i in range(len(event))]):.3f}")
    print(df.round(3).to_string(index=False))
    df.to_csv(out / "diagnostics" / "signal_check.csv", index=False)


if __name__ == "__main__":
    main()
