"""Step 5: does observing a window's outcome make the WM better at predicting what follows it?

For a window [s, u] (model frames, segment.py's convention) and horizon K, frames u+1 .. u+K are predicted twice,
both open loop with the logged actions:
    blind     real context ending at s-1 (frames s-H .. s-1), rolled through the window to u+K: never sees the outcome
    informed  real context ending at u (frames u-H+1 .. u), rolled to u+K
gap = mean blind error - mean informed error over u+1 .. u+K.

The blind rollout is longer, so it compounds more error whatever the hidden state. Control: the same gap for random
windows of the same length away from every candidate window; resolvability = gap - median(control gaps of that
length), plus the gap's percentile among them. K is truncated at the episode end (K_used) and the window skipped if
K_used < resolve_min_horizon; a window starting before frame H has no blind context and is skipped too.
"""
from collections import defaultdict

import numpy as np
import torch

from .metrics import error_metrics, to_device


def resolve_metric(cfg):
    m = cfg.resolve_metric or cfg.error_metric
    if m in ("err_l2_masked", "err_l2_changed", "frac_changed") or m.startswith(("copy_", "err_cls")):
        raise ValueError(f"resolvability does not support {m}; set resolve_metric")
    return m


def horizon(T, s, u, K, H):
    """(K_used, skip reason or None) for a window [s, u] in an episode of T frames."""
    if s < H:
        return 0, "no_blind_context"
    k = min(K, T - 1 - u)
    return k, (None if k >= 1 else "no_frames_after")


def rollout_errors(ep, reqs, wm, metric, batch_size, topk=16):
    """reqs: list of (ctx_end, n_steps). -> list of (n_steps,) per-step errors of frames ctx_end+1 .. ctx_end+n."""
    H, dev = wm.history_len, wm.device
    vis, prop, act = to_device(ep, dev)
    out = [None] * len(reqs)
    by_n = defaultdict(list)
    for i, (c, n) in enumerate(reqs):
        by_n[n].append(i)
    for n, ids in by_n.items():
        for b in range(0, len(ids), batch_size):
            chunk = ids[b:b + batch_size]
            c = torch.as_tensor([reqs[i][0] for i in chunk], device=dev)
            ctx = c[:, None] + torch.arange(-H + 1, 1, device=dev)
            aidx = c[:, None] + torch.arange(-H + 1, n, device=dev)            # a_{c-H+1} .. a_{c+n-1}
            tgt = c[:, None] + torch.arange(1, n + 1, device=dev)
            pred = wm.rollout(vis[ctx], prop[ctx], act[aidx])                 # (b, n, P, D)
            real = vis[tgt]
            e = error_metrics(pred.flatten(0, 1), real.flatten(0, 1), topk=topk)[metric].reshape(len(chunk), n)
            for j, i in enumerate(chunk):
                out[i] = e[j]
    return out


def window_requests(s, u, k):
    """Blind and informed rollout requests for frames u+1 .. u+k."""
    return [(s - 1, u + k - s + 1), (u, k)]


def score_pairs(ep, windows, wm, metric, batch_size, topk=16):
    """windows: list of (s, u, k) -> list of (blind (k,), informed (k,)) errors over frames u+1 .. u+k."""
    reqs = [r for s, u, k in windows for r in window_requests(s, u, k)]
    errs = rollout_errors(ep, reqs, wm, metric, batch_size, topk) if reqs else []
    return [(errs[2 * i][-k:], errs[2 * i + 1]) for i, (s, u, k) in enumerate(windows)]


def sample_controls(episodes, segments, lengths_by_group, cfg, rng):
    """Random control windows: {(group, length): [(episode_id, s, u)]}.

    group = the window's wm_train flag (control_episodes 'same_split'), or 'all' / 'heldout' (one pool for every
    window). A control [s, s+L-1] needs its whole span s-H .. u+K inside the episode and clear of every candidate
    window (widened by control_margin).
    """
    H, K = cfg.history_len, cfg.resolve_horizon
    busy = defaultdict(list)
    for sg in segments:
        busy[sg.episode_id].append((sg.t_start - cfg.control_margin, sg.t_end + cfg.control_margin))

    def pool(group):
        if cfg.control_episodes == "same_split":
            return [e for e in episodes if e.wm_train == group]
        if cfg.control_episodes == "heldout":
            return [e for e in episodes if not e.wm_train]
        if cfg.control_episodes == "all":
            return list(episodes)
        raise ValueError(f"control_episodes must be same_split, heldout or all, got {cfg.control_episodes}")

    out = {}
    for group, L in sorted(lengths_by_group):
        eps = [e for e in pool(group) if e.T - 1 - K - (L - 1) >= H]   # some valid start exists
        picks = []
        for _ in range(cfg.n_control_windows if eps else 0):
            for _ in range(cfg.control_max_tries):
                ep = eps[rng.integers(len(eps))]
                s = int(rng.integers(H, ep.T - K - L + 1))               # s >= H and u + K <= T - 1
                u = s + L - 1
                lo, hi = s - H, u + K
                if all(hi < a or lo > b for a, b in busy[ep.episode_id]):
                    picks.append((ep.episode_id, s, u))
                    break
        out[(group, L)] = picks
    return out


def control_group(ep, cfg):
    return ep.wm_train if cfg.control_episodes == "same_split" else cfg.control_episodes


def resolvability(segments, episodes, wm, cfg, seed, log=print):
    """-> (one dict per segment, one dict per control window)."""
    metric, H, K = resolve_metric(cfg), wm.history_len, cfg.resolve_horizon
    by_id = {e.episode_id: e for e in episodes}
    plans = []
    for sg in segments:
        ep = by_id[sg.episode_id]
        k, why = horizon(ep.T, sg.t_start, sg.t_end, K, H)
        if why is None and k < cfg.resolve_min_horizon:
            why = "horizon_too_short"
        plans.append((sg, ep, k, why))
    lengths = {(control_group(ep, cfg), sg.length) for sg, ep, k, why in plans if why is None}
    controls = sample_controls(episodes, segments, lengths, cfg, np.random.default_rng(seed))

    # every rollout of an episode in one pass, so each episode's tokens are loaded once
    work = defaultdict(list)
    for i, (sg, ep, k, why) in enumerate(plans):
        if why is None:
            work[sg.episode_id].append(("win", i, sg.t_start, sg.t_end, k))
    ctrl_rows = []
    for (group, L), picks in controls.items():
        for eid, s, u in picks:
            work[eid].append(("ctl", len(ctrl_rows), s, u, K))
            ctrl_rows.append(dict(episode_id=eid, group=group, length=L, t_start=s, t_end=u))
    results = {}
    for n, (eid, items) in enumerate(work.items()):
        pairs = score_pairs(by_id[eid], [(s, u, k) for _, _, s, u, k in items], wm, metric, cfg.resolve_batch_size,
                            cfg.errors.topk)
        for (kind, i, *_), pr in zip(items, pairs):
            results[(kind, i)] = pr
        if log and (n + 1) % 100 == 0:
            log(f"  rollouts: {n + 1}/{len(work)} episodes")
    for i, row in enumerate(ctrl_rows):
        b, f = results[("ctl", i)]
        row.update(blind_err=b.tolist(), informed_err=f.tolist(), gap=float(b.mean() - f.mean()))

    ctl_by = defaultdict(list)
    for row in ctrl_rows:
        ctl_by[(row["group"], row["length"])].append(row)
    out = []
    for i, (sg, ep, k, why) in enumerate(plans):
        rec = dict(sg.as_dict(), length=sg.length, wm_train_episode=ep.wm_train, K_used=int(k), skipped=why,
                   gap=None, resolvability=None, resolvability_pct=None, n_controls=0, control_gap_median=None)
        if why is None:
            b, f = results[("win", i)]
            gap = float(b.mean() - f.mean())
            cg = np.array([np.mean(c["blind_err"][:k]) - np.mean(c["informed_err"][:k])
                           for c in ctl_by[(control_group(ep, cfg), sg.length)]])
            rec.update(gap=gap, blind_err=b.tolist(), informed_err=f.tolist(), n_controls=int(len(cg)))
            if len(cg):
                med = float(np.median(cg))
                pct = 100.0 * (np.mean(cg < gap) + 0.5 * np.mean(cg == gap))
                rec.update(resolvability=gap - med, resolvability_pct=float(pct), control_gap_median=med)
            else:
                rec["skipped"] = "no_controls"
        out.append(rec)
    return out, ctrl_rows
