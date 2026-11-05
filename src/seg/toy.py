"""Synthetic acceptance world for the pipeline (tests/test_seg_synthetic.py).

Per episode a hidden bit b in {-1, +1} ("swings left / right") is drawn. It is invisible until the single `lift`
action at step t_lift; from frame t_lift + 1 on, patch 0 shows b * v and keeps showing it. Two distractors:
    shake  an action that makes patch 1 flash a fresh random vector in the next frame only: predictable error
           (g should flag it) that reveals nothing about the future (resolvability ~ 0)
    spike  a random one-frame flash on patch 2 with no cause: unpredictable error (g should stay low)
Every frame also carries Gaussian noise on all patches.

The stub WM predicts the average outcome given its H context frames: b * v on patch 0 once some context frame shows
the swing, 0 otherwise (the mean of +v and -v), and 0 on the flashing patches (their mean). It is the MSE-optimal
deterministic model of this world, so it fails exactly where hidden state or noise decides the next frame.
"""
import numpy as np
import torch

from .episodes import Episode
from .wm import WorldModel

LIFT, SHAKE = 0, 1    # action channels


def toy_episodes(n, T=60, P=4, D=8, amp=2.0, noise=0.05, n_shake=2, n_spike=2, n_wm_train=0, seed=0, gap=5):
    """n episodes; the first n_wm_train are flagged wm_train (as if the WM had trained on them)."""
    rng = np.random.default_rng(seed)
    v = np.zeros(D, np.float32)
    v[0] = amp
    eps = []
    for i in range(n):
        b = rng.choice([-1.0, 1.0])
        taken = []

        def draw(lo, hi):
            for _ in range(1000):
                t = int(rng.integers(lo, hi))
                if all(abs(t - u) >= gap for u in taken):
                    taken.append(t)
                    return t
            raise RuntimeError("toy episode too short for its events")
        t_lift = draw(8, T - 12)
        shakes = [draw(4, T - 2) for _ in range(n_shake)]
        spikes = [draw(4, T - 1) for _ in range(n_spike)]
        vis = rng.normal(0, noise, size=(T, P, D)).astype(np.float32)
        vis[t_lift + 1:, 0] += b * v
        for t in shakes:
            vis[t + 1, 1] += rng.normal(0, amp / np.sqrt(D) * 1.5, size=D)
        for t in spikes:
            vis[t, 2] += rng.normal(0, amp / np.sqrt(D) * 1.5, size=D)
        act = rng.normal(0, 0.05, size=(T, 2)).astype(np.float32)
        act[t_lift, LIFT] += 1.0
        act[shakes, SHAKE] += 1.0
        events = dict(lift=np.array([t_lift + 1]), shake=np.array(shakes) + 1, spike=np.array(spikes))
        ep = Episode(f"toy/ep{i:03d}", act, np.zeros((T, 1), np.float32), visual=vis, wm_train=i < n_wm_train,
                     store="toy", index=i, events=events, render=render)
        ep.hidden_bit, ep.t_lift = b, t_lift
        eps.append(ep)
    return eps, v


def render(ep, t, size=32):
    """A small grey image of frame t: one column band per patch, brightness = the patch's first number."""
    x = np.asarray(ep.visual[t, :, 0], dtype=np.float32)
    col = np.clip(127 + 50 * x, 0, 255).astype(np.uint8)
    img = np.repeat(np.repeat(col[None, :], size, 0), size // len(col), 1)
    return np.repeat(img[..., None], 3, 2)


class AverageOutcomeWM(WorldModel):
    def __init__(self, v, history_len=3, device="cpu"):
        self.v = torch.as_tensor(v, dtype=torch.float32)
        self.history_len, self.device = history_len, torch.device(device)

    def predict_next(self, vis, prop, act, return_features=False):
        vis = vis.float()
        b, H, P, D = vis.shape
        v = self.v.to(vis.device)
        proj = vis[:, :, 0] @ v / (v @ v)                         # (b, H): +1 / -1 where the swing shows, ~0 before
        shown = proj.abs() > 0.5
        last = torch.where(shown, torch.arange(H, device=vis.device), torch.full_like(proj, -1, dtype=torch.long)).amax(1)
        sign = torch.sign(proj.gather(1, last.clamp_min(0)[:, None])[:, 0]) * (last >= 0)
        pred = torch.zeros(b, P, D, device=vis.device)
        pred[:, 0] = sign[:, None] * v
        return (pred, None) if return_features else pred
