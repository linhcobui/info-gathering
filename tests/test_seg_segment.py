"""Segmentation on hand-written g sequences (src/seg/segment.py)."""
import numpy as np
import pandas as pd
import pytest

from seg.config import load_config
from seg.segment import cap, entropy_windows, hysteresis, merge, segment_all, segment_series, smooth


def cfg(**kw):
    c = load_config(overrides=[f"{k}={v}" for k, v in kw.items()])
    return c


def test_hysteresis_opens_above_on_and_closes_below_off():
    q = [0.1, 0.6, 0.4, 0.35, 0.2, 0.1, 0.7, 0.8]
    # opens at 0.6, stays through 0.4 / 0.35 (>= off 0.3), closes at 0.2; reopens at 0.7 and runs to the end
    assert hysteresis(q, on=0.5, off=0.3) == [(1, 3), (6, 7)]


def test_hysteresis_needs_strictly_above_on():
    assert hysteresis([0.5, 0.5, 0.2], on=0.5, off=0.3) == []


def test_hysteresis_rejects_off_above_on():
    with pytest.raises(ValueError):
        hysteresis([0.1], on=0.3, off=0.5)


def test_merge_gap():
    iv = [(0, 1), (3, 4), (8, 9)]
    assert merge(iv, gap=2) == [(0, 4), (8, 9)]      # 1 step between the first two (< 2): merged; 3 steps: not
    assert merge(iv, gap=1) == iv                     # fewer than 1 step apart = touching only
    assert merge([(0, 1), (2, 3)], gap=1) == [(0, 3)]
    assert merge(iv, gap=4) == [(0, 9)]


def test_min_and_max_len():
    q = np.array([0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.6, 0.7, 0.9, 0.95, 0.9, 0.6, 0.6, 0.1])
    c = cfg(min_len=2, max_len=4, merge_gap=0, smooth_window=1, on_thresh=0.5, off_thresh=0.3)
    iv, _ = segment_series(q, c)
    # (0,0) too short; (3,5) kept; (7,13) is 7 long -> its best 4-step stretch is 8..11 (0.7 0.9 0.95 0.9)
    assert iv == [(3, 5), (8, 11)]


def test_cap_keeps_highest_mean_stretch():
    q = np.array([0.6, 0.6, 0.9, 0.95, 0.6])
    assert cap([(0, 4)], q, 2) == [(2, 3)]


def test_smooth_edges_and_identity():
    q = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(smooth(q, 1), q)
    assert np.allclose(smooth(q, 3), [0.5, 1 / 3, 1 / 3, 0.0])


def test_windows_never_cross_episodes_and_use_frame_coordinates():
    # g scores err_t (prediction of frame t+1); q_tau = g_{tau-1}. ep a ends high, ep b starts high.
    a = pd.DataFrame(dict(episode_id="a", t=np.arange(2, 8), g=[0.1, 0.1, 0.1, 0.1, 0.9, 0.9]))
    b = pd.DataFrame(dict(episode_id="b", t=np.arange(2, 8), g=[0.9, 0.9, 0.1, 0.1, 0.1, 0.1]))
    segs = segment_all(pd.concat([a, b]), cfg(merge_gap=5, min_len=1, smooth_window=1))
    got = sorted((s.episode_id, s.t_start, s.t_end) for s in segs)
    assert got == [("a", 7, 8), ("b", 3, 4)]
    assert all(s.onset_score == pytest.approx(0.9) for s in segs)


def test_segment_all_rejects_gaps_in_t():
    df = pd.DataFrame(dict(episode_id="a", t=[2, 3, 5], g=[0.1, 0.9, 0.1]))
    with pytest.raises(ValueError):
        segment_all(df, cfg())


def test_onset_score_mean():
    df = pd.DataFrame(dict(episode_id="a", t=np.arange(2, 8), g=[0.1, 0.6, 1.0, 0.8, 0.1, 0.1]))
    (s,) = segment_all(df, cfg(onset_score="mean", merge_gap=0))
    assert (s.t_start, s.t_end) == (4, 6) and s.onset_score == pytest.approx(0.8)


def test_entropy_method_brackets_a_high_stretch():
    q = np.array([0.02] * 6 + [0.97] * 5 + [0.02] * 6)
    iv = entropy_windows(smooth(q, 3), on=0.5, window=3, min_prominence=0.05)
    assert len(iv) == 1
    s, e = iv[0]
    assert s <= 7 and e >= 9 and s >= 4 and e <= 12          # covers the core of the high stretch, not the episode
