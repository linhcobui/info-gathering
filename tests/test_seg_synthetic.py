"""Acceptance test of the pipeline's logic on the toy world (src/seg/toy.py): a hidden swing direction revealed
only by a lift, with an average-outcome stub WM. The pipeline must (a) put windows at the lift, (b) give them high
resolvability, (c) give random windows and the distractor windows (shake: predictable but uninformative) ~0."""
import json

import numpy as np
import pytest

from seg.config import load_config
from seg.pipeline import errors_step, g_step, labels_step, windows_step
from seg.toy import AverageOutcomeWM, toy_episodes


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("seg_toy")
    cfg = load_config(overrides=[
        f"paths.out_dir={out}", "device=cpu", "history_len=3", "errors.batch_size=64",
        "g.hidden=64", "g.epochs=25", "g.patience=8", "g.batch_size=256", "g.lr=3e-3",
        "resolve_horizon=5", "resolve_min_horizon=3", "n_control_windows=150",
        "diagnostics.n_episode_plots=4", "diagnostics.n_montage=5"])
    eps, v = toy_episodes(n=300, n_wm_train=40, seed=0)
    wm = AverageOutcomeWM(v)
    errors_step(eps, wm, cfg, out, log=None)
    labels_step(cfg, out, log=None)
    scores, metrics = g_step(cfg, out, "cpu", eps, log=None)
    segs, recs, ctrls, tests = windows_step(cfg, out, eps, wm, log=None)
    return dict(cfg=cfg, out=out, eps=eps, segs=segs, recs=recs, ctrls=ctrls, tests=tests, metrics=metrics)


def covering(recs, eid, frame):
    return [r for r in recs if r["episode_id"] == eid and r["t_start"] <= frame <= r["t_end"]]


def test_windows_at_the_lift(run):
    hit = [bool(covering(run["recs"], e.episode_id, e.t_lift + 1)) for e in run["eps"]]
    assert np.mean(hit) >= 0.95
    # the unpredictable spikes get no window of their own: g only flags what the past and the action predict
    spike_only = [r for r in run["recs"]
                  if not any(r["t_start"] <= f <= r["t_end"] for e in run["eps"] if e.episode_id == r["episode_id"]
                             for f in list(e.events["lift"]) + list(e.events["shake"]))]
    assert len(spike_only) <= 0.05 * len(run["recs"])


def test_lift_windows_resolve(run):
    lift = [r for e in run["eps"] for r in covering(run["recs"], e.episode_id, e.t_lift + 1) if r["resolvability"] is not None]
    assert len(lift) >= 0.9 * len(run["eps"])
    res = np.array([r["resolvability"] for r in lift])
    assert np.median(res) > 0.3                                  # blind error ~ (amp + noise) / P = 0.6, informed ~ 0.14
    assert np.mean([r["resolvability_pct"] >= 95 for r in lift]) >= 0.9


def test_controls_and_distractors_do_not_resolve(run):
    lift_med = np.median([r["resolvability"] for e in run["eps"] for r in covering(run["recs"], e.episode_id, e.t_lift + 1)
                          if r["resolvability"] is not None])
    gaps = np.array([c["gap"] for c in run["ctrls"]])
    assert abs(np.median(gaps)) < 0.05 * lift_med
    shake = [r["resolvability"] for e in run["eps"] for t in e.events["shake"]
             for r in covering(run["recs"], e.episode_id, t)
             if r["resolvability"] is not None and not (r["t_start"] <= e.t_lift + 1 <= r["t_end"])]
    assert len(shake) > 100                                      # shakes are predictable errors: they do get windows
    assert abs(np.median(shake)) < 0.1 * lift_med


def test_g_needs_the_action_and_is_calibrated(run):
    m = run["metrics"]
    # ~2 of the ~5 error spikes per episode are causeless flashes, which no g can rank above chance: the AUROC
    # ceiling is about 0.6 * 1 + 0.4 * 0.5 = 0.8
    assert m["test"]["g (calibrated)"]["auroc"] > 0.75
    assert m["test"]["g (calibrated)"]["auprc"] > 0.5            # positive rate ~0.1
    assert m["test"]["g (calibrated)"]["ece"] < 0.05
    assert m["ablation"]["auroc_drop_without_action"] > 0.1     # lifts and shakes are only announced by a_t
    assert not m["ablation"]["flag_action_barely_matters"]


def test_outputs_written(run):
    out = run["out"]
    for f in ("wm_errors.parquet", "labels.parquet", "g_scores.parquet", "candidate_tests.json", "REPORT.md",
              "wm_errors.parquet.meta.json", "diagnostics/error_hist.png", "diagnostics/g_reliability.png",
              "diagnostics/montage_top.png", "diagnostics/resolvability_vs_controls.png", "diagnostics/summary.csv"):
        assert (out / f).exists(), f
    tests = json.loads((out / "candidate_tests.json").read_text())
    res = [t["resolvability"] for t in tests if t["resolvability"] is not None]
    assert res == sorted(res, reverse=True)
    t0 = tests[0]
    assert {"episode_id", "t_start", "t_end", "onset_score", "resolvability", "resolvability_pct", "frame_paths",
            "outcome_frame_paths", "episode_return", "episode_success"} <= set(t0)
    assert t0["frame_paths"] and t0["outcome_frame_paths"]
    meta = json.loads((out / "candidate_tests.json.meta.json").read_text())
    assert "config" in meta and "seg_source_sha256" in meta
