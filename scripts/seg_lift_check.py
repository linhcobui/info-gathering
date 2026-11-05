"""Does the DINO-WM's error at probe lifts grow with the grasp's distance from the hidden COM? (privileged diagnostic)

A probe grasps at s and lifts. Within ~4.5 cm of the true COM c the bar stays level; further away it swings, and which
way depends on c. If the WM's error reflects that hidden state, it should grow with |s - c|. For every held-out probe,
the error of the WM's prediction of the frames 0 .. n_after steps after the lift starts (frame tau predicted from the
context ending at tau - h, open loop over h steps) is averaged, then binned by |s - c|.

    pixi run python scripts/seg_lift_check.py --errors outputs/wm_rollout_errors_heldout200.parquet
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import zarr  # noqa: E402

from seg.config import load_config  # noqa: E402

BINS = [0, 3, 6, 10, 30]
LABELS = ["<3 cm (level)", "3-6", "6-10", ">10 cm (swings)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", required=True)
    ap.add_argument("--n-after", type=int, default=5)
    ap.add_argument("--metric", default="err_l2_mean")
    args = ap.parse_args()
    cfg = load_config()
    err = pd.read_parquet(ROOT / args.errors)
    err = err[~err.wm_train_episode]
    store = next(s for s in cfg.stores if not s.wm_train)
    z = zarr.open_group(str(ROOT / store.zarr), mode="r")
    a = dict(z.attrs)
    seg, stage = z["segment_type"][:], z["probe_stage"][:]
    s_all, tilt = z["grasp_offset_s"][:], np.degrees(z["bar_tilt"][:])
    c_ep = z["episode"]["com_offset_c"][:]
    lift_code, probe_code = a["probe_stages"].index("lift"), a["play_segments"].index("probe")
    # model frames of each held-out episode, and the lift starts among them
    ends = z["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    probes = []
    for e, (s0, e0) in enumerate(zip(starts, ends)):
        fi = s0 + cfg.frameskip * np.arange((e0 - s0) // cfg.frameskip)
        is_lift = (seg[fi] == probe_code) & (stage[fi] == lift_code)
        for tau0 in np.flatnonzero(is_lift & ~np.concatenate([[False], is_lift[:-1]])):
            span = fi[tau0:tau0 + args.n_after + 1]
            probes.append(dict(episode_id=f"{store.name}/ep{e:03d}", tau0=int(tau0), off=abs(s_all[fi[tau0]] - c_ep[e]) * 100,
                               max_tilt=float(np.abs(tilt[span]).max())))
    P = pd.DataFrame(probes)
    P["bin"] = pd.cut(P.off, BINS, labels=LABELS)
    err = err.assign(tau=err.t + err.h)
    rows = []
    for h, eh in err.groupby("h"):
        m = P.merge(eh[["episode_id", "tau", args.metric, "copy_l2_mean"]], on="episode_id")
        m = m[(m.tau >= m.tau0) & (m.tau <= m.tau0 + args.n_after)]
        per = m.groupby(["episode_id", "tau0"]).agg(err=(args.metric, "mean"), copy=("copy_l2_mean", "mean")).reset_index()
        per = per.merge(P[["episode_id", "tau0", "bin", "max_tilt"]], on=["episode_id", "tau0"])
        g = per.groupby("bin", observed=True).agg(n=("err", "size"), max_tilt=("max_tilt", "median"), err=("err", "mean"),
                                                  copy=("copy", "mean"))
        g["h"], g["all_steps_mean"] = h, eh[args.metric].mean()
        rows.append(g.reset_index())
    out = pd.concat(rows)
    pd.set_option("display.width", 200)
    print(f"{len(P)} held-out probes; {args.metric} of frames 0..{args.n_after} after the lift starts, predicted h steps ahead "
          "(open loop from real context); copy = copy-last baseline over the same h steps")
    for h, g in out.groupby("h"):
        far, near = g.err.iloc[-1], g.err.iloc[0]
        print(f"\nh = {h}  (mean over all held-out steps {g.all_steps_mean.iloc[0]:.2f}; far/near ratio {far / near:.2f})")
        print(g[["bin", "n", "max_tilt", "err", "copy"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
