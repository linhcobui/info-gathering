"""The pipeline's steps, shared by the scripts and the synthetic end-to-end test. Each step reads the previous
steps' files from the output directory and writes its own, each with a .meta.json (config, git state)."""
import copy
import json

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from . import diagnostics as diag
from .config import write_meta
from .error_predictor import FeatureSource, build_samples, classification_report, fit, predict, proba_np
from .export import candidate_tests
from .metrics import compute_errors
from .resolvability import resolvability
from .segment import Segment, segment_all
from .threshold import make_labels

# ---------------------------------------------------------------- step 1


def errors_step(episodes, wm, cfg, out, mask_fn=None, log=print):
    want = bool(cfg.errors.save_wm_features) and hasattr(wm, "model")
    cls = None
    if cfg.errors.cls and hasattr(wm, "model"):
        from .cls import ClsScorer
        cls = ClsScorer(wm)
    errors, frames, arrays, feats = compute_errors(episodes, wm, cfg.errors.batch_size, mask_fn, want, log,
                                                   cfg.errors.topk, cfg.errors.change_thresh, cls)
    errors.to_parquet(out / "wm_errors.parquet")
    write_meta(out / "wm_errors.parquet", cfg, n_episodes=len(episodes), wm_step=getattr(wm, "step", None))
    frames.to_parquet(out / "frames.parquet")
    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v)
    if feats is not None:
        np.save(out / "wm_features.npy", feats.astype(np.float16))
    summary = error_summary(errors)
    (out / "wm_errors_summary.json").write_text(json.dumps(summary, indent=1))
    if log:
        for name, s in summary.items():
            log(f"  {name}: {s['n_episodes']} eps, {s['n_steps']} steps, err_l2_mean median {s['err_l2_mean']['median']:.3f}"
                f" p90 {s['err_l2_mean']['p90']:.3f}")
    return errors


def error_summary(errors):
    out = {}
    for flag, name in ((False, "held-out"), (True, "WM-train")):
        e = errors[errors.wm_train_episode == flag]
        if not len(e):
            continue
        out[name] = dict(n_episodes=int(e.episode_id.nunique()), n_steps=int(len(e)), **{
            m: dict(mean=float(e[m].mean()), median=float(e[m].median()), p90=float(e[m].quantile(0.9)),
                    p99=float(e[m].quantile(0.99)))
            for m in ("err_l2_mean", "err_l2_max", "err_cos_mean", "err_l2_topk", "copy_l2_mean", "copy_l2_topk",
                      "err_l2_changed", "frac_changed", "err_cls", "err_cls_cos", "copy_cls") if m in e and e[m].notna().any()})
    return out

# ---------------------------------------------------------------- step 2


def labels_step(cfg, out, log=print):
    errors = pd.read_parquet(out / "wm_errors.parquet")
    labels, split, info = make_labels(errors, cfg)
    labels.to_parquet(out / "labels.parquet")
    write_meta(out / "labels.parquet", cfg, **info)
    (out / "split.json").write_text(json.dumps(split, indent=1))
    (out / "threshold.json").write_text(json.dumps(info, indent=1))
    if log:
        stat = "mean" if cfg.lambda_stat == "mean" else f"q {cfg.lambda_quantile}"
        log(f"  lambda = {info['lambda_']:.4g} ({cfg.error_metric}, {cfg.normalise}, {stat}); "
            f"positive rate by split: " + ", ".join(f"{k} {v:.3f}" for k, v in info["positive_rate_by_split"].items()))
    return labels, info

# ---------------------------------------------------------------- step 3


def _attn_source(episodes, frames, n_frames, device, frame_proprio, pooling="attn"):
    base = frames.groupby("episode_id", sort=False).row.min()
    local, chunks, m = np.full(n_frames, -1, dtype=np.int64), [], 0
    for ep in episodes:
        r0 = int(base[ep.episode_id])
        local[r0:r0 + ep.T] = m + np.arange(ep.T)
        chunks.append(ep.load_visual())
        m += ep.T
    return FeatureSource(pooling, device, patches=np.concatenate(chunks), local=local, frame_proprio=frame_proprio)


def g_step(cfg, out, device, episodes=None, log=print):
    errors = pd.read_parquet(out / "wm_errors.parquet")
    labels = pd.read_parquet(out / "labels.parquet")
    frames = pd.read_parquet(out / "frames.parquet")
    if not (errors.episode_id.equals(labels.episode_id) and errors.t.equals(labels.t)):
        raise ValueError("labels.parquet does not line up with wm_errors.parquet; rerun make_labels")
    act = np.load(out / "frame_action.npy")
    prop = np.load(out / "frame_proprio.npy") if cfg.g.use_proprio else None
    samples = build_samples(errors, labels, frames, act, cfg.history_len, cfg.error_metric, cfg.g.n_actions)
    split = labels.split.to_numpy()
    pooling = cfg.g.pooling
    by_id = {e.episode_id: e for e in (episodes or [])}

    def source_for(ids):
        if pooling == "mean":
            return FeatureSource("mean", device, frame_mean=np.load(out / "frame_mean.npy"), frame_proprio=prop)
        if pooling == "wm":
            if not (out / "wm_features.npy").exists():
                raise FileNotFoundError("g.pooling wm needs wm_features.npy (errors.save_wm_features)")
            return FeatureSource("wm", device, wm_feat=np.load(out / "wm_features.npy"), frame_proprio=prop)
        if pooling == "attn":
            return _attn_source([by_id[i] for i in ids], frames, len(frames), device, prop)
        raise ValueError(f"g.pooling must be mean, attn or wm, got {pooling}")

    fit_ids = sorted(set(labels.episode_id[np.isin(split, ["train", "val", "test"])]))
    src = source_for(fit_ids)
    tr, va, te = (samples.subset(split == s) for s in ("train", "val", "test"))
    if log:
        log(f"  g: {pooling} pooling, {cfg.g.arch}; samples train {len(tr)} val {len(va)} test {len(te)}; "
            f"positive rate train {tr.y.mean():.3f}")
    model, hist = fit(src, tr, va, cfg.g, cfg.history_len, cfg.seed, log)
    nb = cfg.g.ece_bins
    lt, _ = predict(model, src, te, cfg.g.batch_size * 4)
    reports = {"g (calibrated)": classification_report(te.y, proba_np(model, lt), nb),
               "g (raw, class-weighted)": classification_report(te.y, 1 / (1 + np.exp(-lt)), nb)}
    metrics = dict(history=hist, test=reports)

    ablation = None
    if cfg.g.ablate_action and cfg.g.use_action:
        gcfg = copy.deepcopy(cfg.g)
        OmegaConf.update(gcfg, "use_action", False)
        if log:
            log("  ablation: g without a_t")
        ablation, hist2 = fit(src, tr, va, gcfg, cfg.history_len, cfg.seed, log)
        la, _ = predict(ablation, src, te, cfg.g.batch_size * 4)
        reports["g without a_t (calibrated)"] = classification_report(te.y, proba_np(ablation, la), nb)
        d = reports["g (calibrated)"]["auroc"] - reports["g without a_t (calibrated)"]["auroc"]
        metrics["ablation"] = dict(history=hist2, auroc_drop_without_action=float(d),
                                   auprc_drop_without_action=float(reports["g (calibrated)"]["auprc"]
                                                                   - reports["g without a_t (calibrated)"]["auprc"]),
                                   flag_action_barely_matters=bool(d < cfg.g.action_flag_auroc_delta),
                                   note="wm pooling already contains a_t" if pooling == "wm" else None)
        if log:
            log(f"  test AUROC drop without a_t: {d:+.3f}" + ("  <- FLAG: the action barely matters" if d < cfg.g.action_flag_auroc_delta else ""))

    # score every step of every episode (attn: a chunk of episodes at a time)
    g = np.zeros(len(samples))
    logit = np.zeros(len(samples))
    g_na = np.full(len(samples), np.nan)
    reg = np.full(len(samples), np.nan)
    all_ids = list(dict.fromkeys(errors.episode_id))
    chunks = [all_ids] if pooling != "attn" else [all_ids[i:i + 50] for i in range(0, len(all_ids), 50)]
    for ids in chunks:
        s = src if pooling != "attn" else source_for(ids)
        m = errors.episode_id.isin(ids).to_numpy()
        sub = samples.subset(m)
        lo, r = predict(model, s, sub, cfg.g.batch_size * 4)
        logit[m], g[m] = lo, proba_np(model, lo)
        if r is not None:
            reg[m] = r
        if ablation is not None:
            la, _ = predict(ablation, s, sub, cfg.g.batch_size * 4)
            g_na[m] = proba_np(ablation, la)
    scores = pd.DataFrame(dict(episode_id=errors.episode_id, t=errors.t, split=labels.split,
                               wm_train_episode=errors.wm_train_episode, y=labels.y, g=g.astype(np.float32),
                               logit=logit.astype(np.float32), logerr_pred=reg.astype(np.float32),
                               g_noaction=g_na.astype(np.float32)))
    wm_rows = (split == "wm_train")
    if wm_rows.any():
        reports["g on WM-train episodes (in-sample WM errors)"] = classification_report(labels.y[wm_rows], g[wm_rows], nb)
    scores.to_parquet(out / "g_scores.parquet")
    write_meta(out / "g_scores.parquet", cfg)
    gdir = out / "g"
    gdir.mkdir(exist_ok=True)
    torch.save(dict(state=model.state_dict(), cfg=OmegaConf.to_container(cfg.g), feat_dim=src.feat_dim), gdir / "g.pt")
    (gdir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    ddir = out / "diagnostics"
    ddir.mkdir(exist_ok=True)
    diag.reliability_plot({k: v for k, v in reports.items() if "WM-train" not in k}, cfg, ddir / "g_reliability.png")
    if log:
        for k, rep in reports.items():
            log(f"  {k}: AUROC {rep['auroc']:.3f}  AUPRC {rep['auprc']:.3f} (positive rate {rep['positive_rate']:.3f})"
                f"  ECE {rep['ece']:.3f}  N {rep['n']}")
    return scores, metrics


# ---------------------------------------------------------------- steps 4-6


def windows_step(cfg, out, episodes, wm, log=print):
    scores = pd.read_parquet(out / "g_scores.parquet")
    labels = pd.read_parquet(out / "labels.parquet")
    errors = pd.read_parquet(out / "wm_errors.parquet")
    info = json.loads((out / "threshold.json").read_text())
    segs = segment_all(scores, cfg)
    pd.DataFrame([s.as_dict() for s in segs], columns=["episode_id", "t_start", "t_end", "onset_score"]).to_parquet(out / "segments.parquet")
    write_meta(out / "segments.parquet", cfg)
    if log:
        log(f"  {len(segs)} candidate windows in {len({s.episode_id for s in segs})} episodes")
    records, controls = resolvability(segs, episodes, wm, cfg, cfg.seed, log)
    pd.DataFrame(records).to_parquet(out / "resolvability.parquet")
    pd.DataFrame(controls).to_parquet(out / "controls.parquet")
    write_meta(out / "resolvability.parquet", cfg, n_controls=len(controls))
    tests = candidate_tests(records, episodes, cfg, out)
    (out / "candidate_tests.json").write_text(json.dumps(tests, indent=1))
    write_meta(out / "candidate_tests.json", cfg)

    ddir = out / "diagnostics"
    (ddir / "episodes").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    by_ep = {}
    for s in segs:
        by_ep.setdefault(s.episode_id, []).append(s)
    n = cfg.diagnostics.n_episode_plots
    pick = []
    for flag in (False, True):
        eps = [e for e in episodes if e.wm_train == flag]
        k = min(len(eps), n - n // 2 if not flag else n // 2)
        pick += [eps[i] for i in sorted(rng.choice(len(eps), size=k, replace=False))] if k else []
    for ep in pick:
        diag.episode_plot(ep, labels[labels.episode_id == ep.episode_id], scores[scores.episode_id == ep.episode_id],
                          by_ep.get(ep.episode_id, []), info["lambda_"], cfg,
                          ddir / "episodes" / f"{ep.episode_id.replace('/', '_')}.png")
    scored = sorted([r for r in records if r["resolvability"] is not None], key=lambda r: -r["resolvability"])
    m = cfg.diagnostics.n_montage
    diag.montage(scored[:m], episodes, cfg, ddir / "montage_top.png", f"top {min(m, len(scored))} windows by resolvability")
    diag.montage(scored[::-1][:m], episodes, cfg, ddir / "montage_bottom.png", f"bottom {min(m, len(scored))} windows by resolvability")
    diag.error_histogram(errors, labels, info["lambda_"], cfg, ddir / "error_hist.png")
    diag.resolvability_histogram(records, controls, ddir / "resolvability_vs_controls.png")
    table = diag.summary_table(records, episodes, cfg)
    table.to_csv(ddir / "summary.csv", index=False)
    write_report(cfg, out, records, table, info, log)
    return segs, records, controls, tests


def markdown_table(df):
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(rows + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def write_report(cfg, out, records, table, info, log=print):
    """outputs/REPORT.md: the numbers. Interpretation is added by hand after looking at the diagnostics."""
    es = json.loads((out / "wm_errors_summary.json").read_text())
    gm = json.loads((out / "g" / "metrics.json").read_text()) if (out / "g" / "metrics.json").exists() else {}
    L = ["# Error-spike segmentation: report", "",
         f"Config: `configs/segmentation.yaml` (full config and git state in each output's `.meta.json`).", "",
         "## 1. WM one-step errors", "",
         "Teacher-forced one-step error of the frozen DINO-WM: predicted vs real DINO patch tokens e (visual part, "
         "256 patches x 384), per step t (prediction of frame t+1 from the 3 real frames ending at t). "
         "err_l2_mean = mean over patches of the per-patch L2 distance; err_cos_mean = mean over patches of 1 - cosine.", "",
         "| episodes | N eps | N steps | err_l2_mean median | p90 | p99 | err_cos_mean median | p90 |", "|---|---|---|---|---|---|---|---|"]
    nan = dict(median=float("nan"), p90=float("nan"), p99=float("nan"))
    for name, s in es.items():
        a, c = s.get("err_l2_mean", nan), s.get("err_cos_mean", nan)
        L.append(f"| {name} | {s['n_episodes']} | {s['n_steps']} | {a['median']:.3f} | {a['p90']:.3f} | {a['p99']:.3f} | "
                 f"{c['median']:.4f} | {c['p90']:.4f} |")
    L += ["", "## 2. Threshold", "",
          f"λ = {info['lambda_']:.4g}: the {'mean' if info.get('stat') == 'mean' else str(info['quantile']) + ' quantile'} of `{info['metric']}` ({info['normalise']}) over "
          f"held-out split(s) {info['fit_splits']} ({info['n_fit_rows']} steps). Fraction of steps with err > λ by split: "
          + ", ".join(f"{k} {v:.3f}" for k, v in info["positive_rate_by_split"].items()) + ".", ""]
    if gm:
        L += ["## 3. Error predictor g", "",
              f"g = {cfg.g.arch} on {cfg.g.pooling}-pooled frames + a_t, BCE (pos_weight {gm['history']['pos_weight']:.2f}) "
              f"{'+ log-error head ' if cfg.g.reg_head else ''}, Platt-calibrated on val. Scored on the held-out test split "
              "(episodes never used for training g or λ). AUROC / AUPRC: ranking of steps by g against y = 1[err > λ]; "
              f"ECE: expected calibration error over {cfg.g.ece_bins} equal-width probability bins.", "",
              "| model | N steps | positive rate | AUROC | AUPRC | ECE |", "|---|---|---|---|---|---|"]
        for k, r in gm["test"].items():
            L.append(f"| {k} | {r['n']} | {r['positive_rate']:.3f} | {r['auroc']:.3f} | {r['auprc']:.3f} | {r['ece']:.3f} |")
        if "ablation" in gm:
            a = gm["ablation"]
            L += ["", f"Without a_t the test AUROC drops by {a['auroc_drop_without_action']:+.3f} "
                  f"(AUPRC {a['auprc_drop_without_action']:+.3f})."
                  + (" **Flag: the action barely matters.**" if a["flag_action_barely_matters"] else "")]
        L.append("")
    L += ["## 4-5. Windows and resolvability", "",
          f"Windows: {cfg.seg_method} on q (on {cfg.on_thresh}, off {cfg.off_thresh}, merge < {cfg.merge_gap}, "
          f"length {cfg.min_len}-{cfg.max_len}). Resolvability: gap = blind - informed mean `{cfg.resolve_metric or cfg.error_metric}` "
          f"over the K = {cfg.resolve_horizon} frames after the window, minus the median gap of {cfg.n_control_windows} "
          f"random same-length control windows ({cfg.control_episodes}); pct = the gap's percentile among them. "
          "`frac_windows_near_*` uses privileged events (diagnostics only).", "",
          markdown_table(table) if len(table) else "(no windows)", "",
          "## Top windows", "", "| episode | frames | onset | gap | resolvability | pct |", "|---|---|---|---|---|---|"]
    top = sorted([r for r in records if r["resolvability"] is not None], key=lambda r: -r["resolvability"])[:15]
    for r in top:
        L.append(f"| {r['episode_id']} | {r['t_start']}-{r['t_end']} | {r['onset_score']:.2f} | {r['gap']:.3f} | "
                 f"{r['resolvability']:+.3f} | {r['resolvability_pct']:.0f} |")
    skipped = pd.Series([r["skipped"] for r in records if r["skipped"]]).value_counts().to_dict()
    L += ["", f"Skipped windows: {skipped or 'none'}.", "", "## Notes", "", "(added by hand after inspection)", ""]
    (out / "REPORT.md").write_text("\n".join(L))
    if log:
        log(f"  wrote {out / 'REPORT.md'}")


__all__ = ["errors_step", "labels_step", "g_step", "windows_step", "Segment"]
