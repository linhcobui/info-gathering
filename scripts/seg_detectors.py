"""Try several surprise detectors on the h-step error and score them the same way (privileged scoring, diagnostics).

Every variant turns its signal into a percentile rank over held-out val steps (so 0.8 = higher than 80% of them), fires
in hysteresis windows (on above 1 - fire_frac, off below 1 - fire_frac - 0.15), and is scored on held-out val + test
episodes (not used to fit g's weights):
    win/ep, len         windows per episode, mean length (steps); fire = fraction of steps inside a window
    rho_cam             Spearman of the signal with gripper y (+ = towards the side camera): the slow level we don't want
    lift fire rates     fraction of probe-lift frames 0-5 inside a window, by case: end grasp |s| > 6 cm & bar swings
                        (predictable), end grasp & level (surprise), middle grasp |s| < 6 cm (outcome open)
    mid/end             middle-grasp fire rate / predictable fire rate: should be > 1 if the detector tracks hidden state
    lift/rest           lift-frame fire rate / fire rate elsewhere
    grasp prec / rand   fraction of windows within 2 steps of a grasp onset vs random windows of the same length

    pixi run python scripts/seg_detectors.py --src outputs/v5_h4_topk --h 4
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

from seg.config import load_config, write_meta  # noqa: E402
from seg.episodes import load_store  # noqa: E402
from seg.error_predictor import FeatureSource, Samples, build_samples, fit, predict, proba_np  # noqa: E402
from seg.pipeline import _attn_source  # noqa: E402
from seg.quantile import fit_quantiles, predict_quantiles  # noqa: E402
from seg.segment import segment_all  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402


def privileged(cfg, errors, h):
    """Held-out privileged tables: per-row gripper y; probe lifts (episode_id, tau0, case); grasp onsets."""
    store = next(s for s in cfg.stores if not s.wm_train)
    z = zarr.open_group(str(ROOT / store.zarr), mode="r")
    a = dict(z.attrs)
    seg, stage, s_all, tilt = z["segment_type"][:], z["probe_stage"][:], z["grasp_offset_s"][:], np.degrees(z["bar_tilt"][:])
    grasped, eff_y = z["grasped"][:], z["effector_pos"][:, 1]
    ends = z["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    L, PR = a["probe_stages"].index("lift"), a["play_segments"].index("probe")
    lifts, onsets = [], {}
    for e, (s0, e0) in enumerate(zip(starts, ends)):
        fi = s0 + cfg.frameskip * np.arange((e0 - s0) // cfg.frameskip)
        eid = f"{store.name}/ep{e:03d}"
        il = (seg[fi] == PR) & (stage[fi] == L)
        for t0 in np.flatnonzero(il & ~np.concatenate([[False], il[:-1]])):
            end, swings = abs(s_all[fi[t0]]) > 0.06, float(np.abs(tilt[fi[t0:t0 + 6]]).max()) > 10
            lifts.append(dict(episode_id=eid, tau0=int(t0),
                              case="middle (open)" if not end else ("end, swings (predictable)" if swings else "end, level (surprise)")))
        g = grasped[fi]
        onsets[eid] = np.flatnonzero(g[1:] & ~g[:-1]) + 1
    held = errors[~errors.wm_train_episode]
    cam = pd.Series(eff_y[held.target_raw_frame.to_numpy()], index=held.index)
    return pd.DataFrame(lifts), onsets, cam


def evaluate(name, sig, errors, split, cfg, h, lifts, onsets, cam, fire_frac):
    """sig: per error row. Returns (metrics dict, scores table for videos)."""
    held = ~errors.wm_train_episode.to_numpy()
    val = split == "val"
    ref = np.sort(sig[val & np.isfinite(sig)])
    rank = np.searchsorted(ref, sig, side="right") / len(ref)
    scores = pd.DataFrame(dict(episode_id=errors.episode_id, t=errors.t, split=split, wm_train_episode=errors.wm_train_episode,
                               g=rank.astype(np.float32), signal=sig.astype(np.float32)))
    on = 1 - fire_frac
    c = OmegaConf.merge(cfg, dict(on_thresh=on, off_thresh=on - 0.15, error_horizon=h))
    ev = np.isin(split, ["val", "test"])
    sc = scores[ev]
    segs = segment_all(sc, c)
    n_ep = sc.episode_id.nunique()
    T = sc.groupby("episode_id").t.max() + h + 1
    inwin = {}
    for s in segs:
        inwin.setdefault(s.episode_id, set()).update(range(s.t_start, s.t_end + 1))
    L = lifts[lifts.episode_id.isin(set(sc.episode_id))]
    rate = lambda df: np.mean([np.mean([tau in inwin.get(r.episode_id, ()) for tau in range(r.tau0, r.tau0 + 6)])
                               for r in df.itertuples()]) if len(df) else np.nan
    lift_frames = {(r.episode_id, tau) for r in L.itertuples() for tau in range(r.tau0, r.tau0 + 6)}
    rest = [(e, t + h) for e, t in zip(sc.episode_id, sc.t) if (e, t + h) not in lift_frames]
    rest_rate = np.mean([t in inwin.get(e, ()) for e, t in rest])
    hit = lambda e, a, b: any(a - 2 <= t <= b + 2 for t in onsets.get(e, ()))
    rng = np.random.default_rng(0)
    rand = [hit(s.episode_id, a, a + s.length - 1) for _ in range(10) for s in segs
            for a in [int(rng.integers(h + 2, max(h + 3, T[s.episode_id] - s.length)))]]
    cases = {k: rate(L[L.case == k]) for k in ("end, swings (predictable)", "end, level (surprise)", "middle (open)")}
    m = dict(variant=name, win_per_ep=len(segs) / n_ep, mean_len=np.mean([s.length for s in segs]) if segs else np.nan,
             fire=sum(s.length for s in segs) / len(sc),
             rho_cam=spearmanr(sig[held & np.isfinite(sig)], cam.loc[errors.index[held & np.isfinite(sig)]]).statistic,
             **{f"lift: {k}": v for k, v in cases.items()}, rest=rest_rate,
             mid_over_end=cases["middle (open)"] / cases["end, swings (predictable)"],
             lift_over_rest=np.mean(list(cases.values())) / rest_rate,
             grasp_prec=np.mean([hit(s.episode_id, s.t_start, s.t_end) for s in segs]) if segs else np.nan,
             grasp_rand=np.mean(rand) if rand else np.nan, n_probes=len(L))
    return m, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/v5_h4_topk")
    ap.add_argument("--h", type=int, default=4)
    ap.add_argument("--metric", default="err_l2_topk")
    ap.add_argument("--fire-frac", type=float, default=0.2)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--pooling", default="mean", help="mean | attn (attn: held-out episodes only, patches in RAM)")
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--tag", default="", help="suffix for the output folders and summary")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    src = ROOT / args.src
    cfg = load_config(overrides=[f"paths.out_dir={args.src}", f"error_metric={args.metric}", f"error_horizon={args.h}",
                                 f"g.n_actions={args.h}", f"g.pooling={args.pooling}", f"g.arch={args.arch}", *args.overrides])
    seed_everything(cfg.seed)
    dev = torch.device(cfg.device)
    errors = pd.read_parquet(src / "wm_errors.parquet")
    labels = pd.read_parquet(src / "labels.parquet")
    frames = pd.read_parquet(src / "frames.parquet")
    split = labels.split.to_numpy()
    act = np.load(src / "frame_action.npy")
    S = build_samples(errors, labels, frames, act, cfg.history_len, args.metric, args.h)
    held = ~errors.wm_train_episode.to_numpy()
    if args.pooling in ("attn", "grid"):
        store = next(st for st in cfg.stores if not st.wm_train)
        srcF = _attn_source(load_store(store, cfg.frameskip), frames, len(frames), dev, None, args.pooling)
    else:
        srcF = FeatureSource("mean", dev, frame_mean=np.load(src / "frame_mean.npy"))
    tr, va = S.subset(split == "train"), S.subset(split == "val")
    S_held = S.subset(held)

    def on_held(values):
        """Per-row array: values for held-out rows (the only ones scored), NaN elsewhere."""
        out = np.full(len(S), np.nan)
        out[held] = values
        return out
    lifts, onsets, cam = privileged(cfg, errors, args.h)
    out_root = ROOT / "outputs" / "detectors"
    out_root.mkdir(parents=True, exist_ok=True)
    logerr = S.logerr

    def jump_labels():
        """Derivative label: log err_t above the median of the previous 4 steps' log errors by more than kappa
        (kappa = the 90th percentile of that jump over train rows)."""
        d = pd.DataFrame(dict(e=errors.episode_id, x=logerr))
        prev = d.groupby("e").x.transform(lambda v: v.shift(1).rolling(4, min_periods=1).median())
        jump = (d.x - prev).to_numpy()
        kappa = np.nanquantile(jump[split == "train"], 0.9)
        return np.nan_to_num(jump > kappa).astype(np.float32), jump

    variants = {}
    # 1. the current classifier: P(err_t > mean error)
    sc5 = pd.read_parquet(src / "g_scores.parquet")
    variants["exceed_mean (current g)"] = lambda: sc5.g.to_numpy(dtype=np.float64)

    def quant(taus, which):
        def run():
            model, _ = fit_quantiles(srcF, tr, va, cfg.g, taus, cfg.history_len, cfg.seed, log=None)
            qh = predict_quantiles(model, srcF, S_held, cfg.g.batch_size * 4)
            q = np.full((len(S), len(taus)), np.nan)
            q[held] = qh
            cover = {f"cover_q{t}": float((logerr[split == "test"] <= q[split == "test", i]).mean()) for i, t in enumerate(taus)}
            print(f"  {taus}: test coverage {cover}", flush=True)
            np.save(out_root / f"quantiles_{'_'.join(map(str, taus))}{args.tag}.npy", q)
            return q[:, -1] - q[:, 0] if which == "spread" else q[:, -1]
        return run
    variants["q90 - q50 spread"] = quant([0.5, 0.9], "spread")
    variants["q95 - q50 spread"] = quant([0.5, 0.95], "spread")
    variants["q90 level (no median)"] = quant([0.5, 0.9], "level")

    def jump():
        y, _ = jump_labels()
        model, _ = fit(srcF, Samples(tr.ctx, tr.row, tr.action, y[split == "train"], tr.logerr),
                       Samples(va.ctx, va.row, va.action, y[split == "val"], va.logerr), cfg.g, cfg.history_len, cfg.seed, log=None)
        lo, _ = predict(model, srcF, S_held, cfg.g.batch_size * 4)
        return on_held(proba_np(model, lo))
    variants["jump classifier (derivative)"] = jump

    rows = []
    for name, run in variants.items():
        if args.variants and not any(name.startswith(v) for v in args.variants):
            continue
        name = name + args.tag
        print(f"variant: {name}", flush=True)
        sig = np.asarray(run(), dtype=np.float64)
        m, scores = evaluate(name, sig, errors, split, cfg, args.h, lifts, onsets, cam, args.fire_frac)
        rows.append(m)
        d = out_root / (name.split(" (")[0].replace(" ", "_").replace("-", "minus") + args.tag.replace(" ", "_"))
        d.mkdir(exist_ok=True)
        scores.to_parquet(d / "g_scores.parquet")
        write_meta(d / "g_scores.parquet", cfg, variant=name, fire_frac=args.fire_frac)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(df.round(3).T.to_string())
    df.to_csv(out_root / f"summary{args.tag.replace(' ', '_')}.csv", index=False)
    (out_root / f"summary{args.tag.replace(' ', '_')}.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
