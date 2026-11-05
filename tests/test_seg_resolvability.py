"""Resolvability bookkeeping with a stub WM on a ramp world where every error is known exactly."""
import numpy as np
import pytest
import torch

from seg.config import load_config
from seg.episodes import Episode
from seg.resolvability import horizon, resolvability, rollout_errors
from seg.segment import Segment
from seg.wm import WorldModel

P, D, H = 2, 3, 3


class RampWM(WorldModel):
    """Predicts e_{t+1} = e_t + 1 and records what each call saw (last context frame value, a_t)."""

    def __init__(self):
        self.history_len, self.device, self.calls = H, torch.device("cpu"), []

    def predict_next(self, vis, prop, act, return_features=False):
        self.calls.append((vis[:, -1, 0, 0].tolist(), act[:, -1, 0].tolist()))
        e = vis[:, -1].float() + 1
        return (e, None) if return_features else e


def ramp_episode(eid, T, jump_at=None, jump=5.0):
    t = np.arange(T, dtype=np.float32)
    val = t + (jump * (t >= jump_at) if jump_at is not None else 0)
    vis = np.broadcast_to(val[:, None, None], (T, P, D)).copy()
    act = np.stack([t, t], 1)                     # a_t = t, so the stub's records show which action it got
    return Episode(eid, act, np.zeros((T, 1)), visual=vis)


def cfg(**kw):
    base = dict(history_len=H, resolve_horizon=4, resolve_min_horizon=2, n_control_windows=30, control_margin=0,
                control_episodes="all", error_metric="err_l2_mean", resolve_metric="null", resolve_batch_size=8)
    base.update(kw)
    return load_config(overrides=[f"{k}={v}" for k, v in base.items()])


def test_rollout_uses_real_context_then_logged_actions():
    ep = ramp_episode("a", 20)
    wm = RampWM()
    (err,) = rollout_errors(ep, [(10, 3)], wm, "err_l2_mean", 8)
    assert np.allclose(err, 0)                                  # the ramp continues exactly
    assert [c[0][0] for c in wm.calls] == [10, 11, 12]          # last context frame: real 10, then its own predictions
    assert [c[1][0] for c in wm.calls] == [10, 11, 12]          # actions a_10, a_11, a_12


def test_horizon_rules():
    assert horizon(T=30, s=10, u=12, K=8, H=3) == (8, None)
    assert horizon(T=30, s=25, u=26, K=8, H=3) == (3, None)     # truncated at the episode end: frames 27..29
    assert horizon(T=30, s=2, u=4, K=8, H=3) == (0, "no_blind_context")
    assert horizon(T=30, s=20, u=29, K=8, H=3) == (0, "no_frames_after")


def test_gap_blind_minus_informed_and_control_baseline():
    T, J = 40, 15
    eps = [ramp_episode("jump", T, jump_at=J)] + [ramp_episode(f"flat{i}", T) for i in range(3)]
    segs = [Segment("jump", J, J, 0.9),
            Segment("jump", 36, 36, 0.8),     # K_used = 3 >= min 2: scored on frames 37..39
            Segment("jump", 38, 38, 0.8),     # K_used = 1 < 2: skipped
            Segment("jump", 1, 1, 0.8)]       # no blind context
    recs, ctrls = resolvability(segs, eps, RampWM(), cfg(), seed=0, log=None)
    r = {(x["t_start"]): x for x in recs}
    # blind (context ends at J-1) never sees the +5 jump: every frame J+1..J+4 is off by 5 in all D numbers
    assert r[J]["K_used"] == 4
    assert r[J]["blind_err"] == pytest.approx([5 * np.sqrt(D)] * 4, rel=1e-5)
    assert r[J]["informed_err"] == pytest.approx([0] * 4, abs=1e-5)
    assert r[J]["gap"] == pytest.approx(5 * np.sqrt(D), rel=1e-5)
    assert r[J]["control_gap_median"] == pytest.approx(0, abs=1e-6)   # controls avoid the jump -> no gap
    assert r[J]["resolvability"] == pytest.approx(5 * np.sqrt(D), rel=1e-5)
    assert r[J]["resolvability_pct"] == pytest.approx(100)
    assert r[36]["K_used"] == 3 and r[36]["skipped"] is None and r[36]["gap"] == pytest.approx(0, abs=1e-5)
    assert r[38]["skipped"] == "horizon_too_short"
    assert r[1]["skipped"] == "no_blind_context"
    for c in ctrls:                                                    # controls stay clear of candidate windows
        assert c["episode_id"] != "jump" or c["t_end"] + 4 < J or c["t_start"] - H > J
        assert c["t_start"] >= H and c["t_end"] + 4 <= T - 1
    assert len([c for c in ctrls if c["length"] == 1]) == 30
