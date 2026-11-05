"""Step 1: the WM's teacher-forced one-step error at every valid t of every episode.

err[t] scores the prediction of frame t+1 from the real context ending at t, so valid t = H-1 .. T-2. Metrics
compare predicted and real patch tokens (P patches x D numbers), visual part only:
    err_l2_mean    mean over patches of the per-patch L2 distance
    err_l2_max     max over patches (local events the mean dilutes)
    err_cos_mean   mean over patches of 1 - cosine similarity
    err_l2_masked  mean per-patch L2 over the patches mask_fn(episode, t) selects for frame t+1 (NaN without a mask)
    err_l2_topk    mean per-patch L2 over the k worst patches (local: a tilting bar covers few patches, while arm
                   motion shifts every patch's token a little and dominates the mean)
    copy_l2_mean / copy_l2_topk   the same two for the copy-last-frame baseline e_t vs e_{t+1}: how much the
                   scene itself changed, for reference (not a WM error)
    err_l2_changed mean per-patch L2 over the patches whose real token changed by more than change_thresh this step
                   (|e_{t+1} - e_t| per patch); 0 when none did. Removes how many patches an object covers (near the
                   camera it covers more, and every covered patch carries some error), keeping how badly the WM
                   predicted the patches that moved. frac_changed = the fraction of such patches.
    err_cls / err_cls_cos / copy_cls   CLS-token errors through the WM's decoder and DINOv2 (seg/cls.py), when on
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

METRICS = ("err_l2_mean", "err_l2_max", "err_cos_mean", "err_l2_masked", "err_l2_topk", "copy_l2_mean", "copy_l2_topk",
           "err_l2_changed", "frac_changed", "err_cls", "err_cls_cos", "copy_cls")


def error_metrics(e_hat, e, mask=None, topk=16, prev=None, change_thresh=10.0):
    """e_hat, e (b, P, D); mask (b, P) bool or None; prev (b, P, D) = e_t for the copy-last baseline (else NaN)
    -> {metric: (b,) float32 array}."""
    e_hat, e = e_hat.float(), e.float()
    d = (e_hat - e).norm(dim=-1)
    cos = F.cosine_similarity(e_hat, e, dim=-1, eps=1e-8)
    k = min(int(topk), d.shape[1])
    out = dict(err_l2_mean=d.mean(1), err_l2_max=d.amax(1), err_cos_mean=(1 - cos).mean(1),
               err_l2_topk=d.topk(k, dim=1).values.mean(1))
    nan = torch.full_like(d[:, 0], float("nan"))
    if prev is None:
        out.update(copy_l2_mean=nan, copy_l2_topk=nan, err_l2_changed=nan, frac_changed=nan)
    else:
        dc = (prev.float() - e).norm(dim=-1)
        moved = (dc > change_thresh).to(d.dtype)
        out.update(copy_l2_mean=dc.mean(1), copy_l2_topk=dc.topk(k, dim=1).values.mean(1),
                   err_l2_changed=(d * moved).sum(1) / moved.sum(1).clamp_min(1), frac_changed=moved.mean(1))
    for m in ("err_cls", "err_cls_cos", "copy_cls"):
        out[m] = nan
    if mask is None:
        out["err_l2_masked"] = torch.full_like(d[:, 0], float("nan"))
    else:
        m = mask.to(d.dtype)
        out["err_l2_masked"] = (d * m).sum(1) / m.sum(1)     # 0/0 = NaN when the mask is empty
    return {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}


def context_index(t, H, device):
    """(b,) context-end steps -> (b, H) frame indices t-H+1 .. t."""
    return torch.as_tensor(t, device=device)[:, None] + torch.arange(-H + 1, 1, device=device)


def to_device(ep, device):
    vis = torch.from_numpy(np.ascontiguousarray(ep.load_visual())).to(device)
    return vis, torch.from_numpy(ep.proprio).to(device), torch.from_numpy(ep.action).to(device)


def episode_errors(ep, wm, batch_size, mask_fn=None, want_features=False, topk=16, change_thresh=10.0, cls_scorer=None):
    """Metrics for t = H-1 .. T-2 of one episode, the per-frame patch-mean tokens (T, D), and WM features."""
    H, dev = wm.history_len, wm.device
    vis, prop, act = to_device(ep, dev)
    ts = np.arange(H - 1, ep.T - 1)
    rows, feats = {k: [] for k in METRICS}, []
    for b in range(0, len(ts), batch_size):
        t = ts[b:b + batch_size]
        idx = context_index(t, H, dev)
        pred = wm.predict_next(vis[idx], prop[idx], act[idx], return_features=want_features)
        if want_features:
            pred, f = pred
            feats.append(f.cpu().numpy().astype(np.float32))
        mask = None
        if mask_fn is not None:
            masks = [mask_fn(ep, int(tt)) for tt in t]
            if all(m is not None for m in masks):
                mask = torch.as_tensor(np.stack(masks), device=dev)
        tt = torch.as_tensor(t, device=dev)
        m = error_metrics(pred, vis[tt + 1], mask, topk, prev=vis[tt], change_thresh=change_thresh)
        if cls_scorer is not None:
            m.update(cls_scorer.metrics(pred, vis[tt + 1], vis[tt]))
        for k, v in m.items():
            rows[k].append(np.asarray(v, dtype=np.float32))
    metrics = {k: np.concatenate(v) if v else np.zeros(0, np.float32) for k, v in rows.items()}
    pooled = vis.float().mean(1).cpu().numpy().astype(np.float16)
    return ts, metrics, pooled, (np.concatenate(feats) if feats else None)


def compute_errors(episodes, wm, batch_size, mask_fn=None, want_features=False, log=print, topk=16, change_thresh=10.0,
                   cls_scorer=None):
    """All episodes -> (errors table, frames table, frame arrays, WM features aligned to the errors table).

    errors: one row per (episode_id, t). frames: one row per model frame, `row` indexing the frame arrays
    (mean-pooled tokens, action, proprio), contiguous per episode in t order.
    """
    err_parts, frame_parts, pooled, acts, props, feats = [], [], [], [], [], []
    n_rows = 0
    for i, ep in enumerate(episodes):
        ts, m, pool, f = episode_errors(ep, wm, batch_size, mask_fn, want_features, topk, change_thresh, cls_scorer)
        base = dict(episode_id=ep.episode_id, store=ep.store, episode=ep.index, wm_train_episode=ep.wm_train)
        err_parts.append(pd.DataFrame(dict(**base, t=ts, target_raw_frame=ep.frame_index[ts + 1], **m)))
        frame_parts.append(pd.DataFrame(dict(episode_id=ep.episode_id, t=np.arange(ep.T), raw_frame=ep.frame_index,
                                             row=n_rows + np.arange(ep.T))))
        n_rows += ep.T
        pooled.append(pool)
        acts.append(ep.action)
        props.append(ep.proprio)
        if f is not None:
            feats.append(f)
        if log and (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(episodes)} episodes")
    errors = pd.concat(err_parts, ignore_index=True)
    frames = pd.concat(frame_parts, ignore_index=True)
    arrays = dict(frame_mean=np.concatenate(pooled), frame_action=np.concatenate(acts),
                  frame_proprio=np.concatenate(props))
    return errors, frames, arrays, (np.concatenate(feats) if feats else None)
