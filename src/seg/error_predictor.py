"""Step 3: g(e_{t-H+1:t}, a_t) ~ P(err_t > lambda): the WM's *expected* error, from the recent past and the action.

Inputs per sample (one per error row t): the H context frames, each reduced to a vector by `pooling`
    mean   patch mean of the frame's DINO tokens (precomputed in step 1)
    attn   learned-query attention over the frame's 256 patches (reads the cached patches)
    wm     the WM predictor's own features for this context (step 1, cfg.errors.wm_feature_layer); one vector per
           sample that already includes a_t, so the no-action ablation is not clean with it
plus a_t (unless use_action is false) and optionally the context frames' proprio. Trunk: an MLP over the
concatenated inputs, or a small transformer over H frame tokens + an action token. Heads: a logit for y_t (BCE,
class-weighted) and optionally log(err_t) (standardised, MSE). Class weighting skews the probabilities, so they are
recalibrated on the val split (Platt scaling) before g is read as a probability.
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata

# ---------------------------------------------------------------- metrics


def auroc(y, p):
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, p):
    """Average precision: mean over positives of the precision at their rank."""
    y = np.asarray(y).astype(bool)
    if not y.any():
        return float("nan")
    ys = y[np.argsort(-np.asarray(p), kind="mergesort")]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(prec[ys].mean())


def reliability(y, p, n_bins):
    """Equal-width bins of predicted probability -> (bins table, expected calibration error)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    count = np.bincount(b, minlength=n_bins)
    mean_p = np.bincount(b, p, n_bins) / np.maximum(count, 1)
    frac = np.bincount(b, y, n_bins) / np.maximum(count, 1)
    ece = float((count / max(len(p), 1) * np.abs(mean_p - frac)).sum())
    bins = dict(lo=(np.arange(n_bins) / n_bins).tolist(), count=count.tolist(), mean_p=mean_p.tolist(), frac_pos=frac.tolist())
    return bins, ece


def classification_report(y, p, n_bins):
    bins, ece = reliability(y, p, n_bins)
    return dict(n=int(len(y)), positive_rate=float(np.mean(y)) if len(y) else float("nan"), mean_p=float(np.mean(p)),
                auroc=auroc(y, p), auprc=auprc(y, p), ece=ece, reliability=bins)


# ---------------------------------------------------------------- inputs


class Samples:
    """One sample per error row. ctx (N, H): rows of the context frames in the frame tables; row (N,): the error row
    (indexes `wm` features); action (N, A) = a_t; y (N,); logerr (N,) = log err_t."""

    def __init__(self, ctx, row, action, y, logerr):
        self.ctx, self.row, self.action = np.asarray(ctx), np.asarray(row), np.asarray(action, dtype=np.float32)
        self.y, self.logerr = np.asarray(y, dtype=np.float32), np.asarray(logerr, dtype=np.float32)

    def __len__(self):
        return len(self.row)

    def subset(self, mask):
        return Samples(self.ctx[mask], self.row[mask], self.action[mask], self.y[mask], self.logerr[mask])


def build_samples(errors, labels, frames, frame_action, history_len, metric, n_actions=1):
    """Samples for every error row (same order as `errors`). The action input is a_t .. a_{t+n_actions-1}
    concatenated: for an h-step error, the h logged actions the open-loop rollout is driven by."""
    base = frames.groupby("episode_id", sort=False).row.min()
    T = frames.groupby("episode_id", sort=False).t.max() + 1
    b = errors.episode_id.map(base).to_numpy()
    t = errors.t.to_numpy()
    if np.any(t + n_actions > errors.episode_id.map(T).to_numpy()):
        raise ValueError(f"some error rows lack {n_actions} actions after t")
    ctx = (b + t)[:, None] + np.arange(-history_len + 1, 1)
    act = frame_action[(b + t)[:, None] + np.arange(n_actions)].reshape(len(t), -1)
    logerr = np.log(np.maximum(errors[metric].to_numpy(dtype=np.float64), 1e-8))
    return Samples(ctx, np.arange(len(errors)), act, labels.y.to_numpy(), logerr)


class FeatureSource:
    """Gathers g's per-frame inputs for a batch. `patches` (M, P, D) + `local` (N_frames,) map frame rows to rows of
    the loaded patch array (attn pooling only)."""

    def __init__(self, pooling, device, frame_mean=None, wm_feat=None, patches=None, local=None, frame_proprio=None):
        self.pooling, self.device = pooling, device   # attn and grid both read patches
        self.frame_mean = None if frame_mean is None else torch.as_tensor(frame_mean, dtype=torch.float32, device=device)
        self.wm_feat = None if wm_feat is None else torch.as_tensor(wm_feat, dtype=torch.float32, device=device)
        self.patches, self.local = patches, local
        self.frame_proprio = (None if frame_proprio is None
                              else torch.as_tensor(frame_proprio, dtype=torch.float32, device=device))

    @property
    def feat_dim(self):
        if self.pooling == "mean":
            return self.frame_mean.shape[1]
        if self.pooling == "wm":
            return self.wm_feat.shape[1]
        return self.patches.shape[-1]

    def n_frames(self, history_len):
        return 1 if self.pooling == "wm" else history_len

    def frames(self, ctx, row):
        if self.pooling == "mean":
            return self.frame_mean[torch.as_tensor(ctx, device=self.device)]
        if self.pooling == "wm":
            return self.wm_feat[torch.as_tensor(row, device=self.device)][:, None]
        loc = self.local[ctx]
        if (loc < 0).any():
            raise IndexError("attn pooling: a context frame's patches are not loaded")
        return torch.from_numpy(np.ascontiguousarray(self.patches[loc])).to(self.device).float()

    def proprio(self, ctx):
        return self.frame_proprio[torch.as_tensor(ctx, device=self.device)]


def norm_stats(source, train, cfg, rng):
    """Input mean/std from the train samples only."""
    rows = np.unique(train.ctx)
    if source.pooling == "mean":
        x = source.frame_mean[torch.as_tensor(rows, device=source.device)]
    elif source.pooling == "wm":
        x = source.wm_feat[torch.as_tensor(train.row, device=source.device)]
    else:
        pick = rng.choice(rows, size=min(len(rows), max(1, cfg.attn_norm_sample // source.patches.shape[1])),
                          replace=False)
        x = torch.from_numpy(np.ascontiguousarray(source.patches[source.local[np.sort(pick)]])).float()
        x = x.reshape(-1, x.shape[-1])
    st = dict(feat_mean=x.mean(0).cpu(), feat_std=x.std(0).clamp_min(1e-6).cpu(),
              act_mean=torch.as_tensor(train.action.mean(0)), act_std=torch.as_tensor(train.action.std(0)).clamp_min(1e-6),
              logerr_mean=torch.tensor(float(train.logerr.mean())), logerr_std=torch.tensor(float(train.logerr.std()) or 1.0))
    if source.frame_proprio is not None:
        p = source.frame_proprio[torch.as_tensor(rows, device=source.device)]
        st.update(prop_mean=p.mean(0).cpu(), prop_std=p.std(0).clamp_min(1e-6).cpu())
    return st


# ---------------------------------------------------------------- model


class GridPool(nn.Module):
    """Keeps the patch layout: each patch -> `proj` numbers, then two stride-2 convolutions over the 16 x 16 grid
    -> (..., out_dim) with out_dim = 4 x 4 x `channels`. Says *where* things are (e.g. where along the bar the gripper
    grasps), which patch means and a single attention query lose."""

    def __init__(self, dim, proj=32, channels=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, proj)
        self.conv = nn.Sequential(nn.Conv2d(proj, channels, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.GELU())
        self.out_dim = 16 * channels

    def forward(self, x):
        lead, (P, D) = x.shape[:-2], x.shape[-2:]
        side = int(P ** 0.5)
        x = self.proj(self.norm(x.reshape(-1, P, D)))                         # (n, P, proj)
        x = x.transpose(1, 2).reshape(len(x), -1, side, side)
        return self.conv(x).flatten(1).reshape(*lead, -1)


class AttnPool(nn.Module):
    """One learned query attending over a frame's patches: (..., P, D) -> (..., D)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        lead = x.shape[:-2]
        x = self.norm(x.reshape(-1, *x.shape[-2:]))
        out, _ = self.attn(self.query.expand(len(x), -1, -1), x, x, need_weights=False)
        return out[:, 0].reshape(*lead, -1)


def mlp(d_in, hidden, depth, dropout):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
        d = hidden
    return nn.Sequential(*layers)


class ErrorPredictor(nn.Module):
    def __init__(self, feat_dim, action_dim, n_frames, stats, cfg, proprio_dim=0):
        super().__init__()
        self.pooling, self.arch = cfg.pooling, cfg.arch
        self.use_action, self.use_proprio = bool(cfg.use_action), bool(cfg.use_proprio) and proprio_dim > 0
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        self.register_buffer("calib_a", torch.tensor(1.0))
        self.register_buffer("calib_b", torch.tensor(0.0))
        self.pool = (AttnPool(feat_dim, cfg.heads) if cfg.pooling == "attn"
                     else GridPool(feat_dim) if cfg.pooling == "grid" else None)
        d_frame = (self.pool.out_dim if isinstance(self.pool, GridPool) else feat_dim) + (proprio_dim if self.use_proprio else 0)
        h = cfg.hidden
        if cfg.arch == "mlp":
            self.trunk = mlp(n_frames * d_frame + (action_dim if self.use_action else 0), h, cfg.depth, cfg.dropout)
        elif cfg.arch == "transformer":
            self.frame_in = nn.Linear(d_frame, h)
            self.act_in = nn.Linear(action_dim, h) if self.use_action else None
            self.cls = nn.Parameter(torch.zeros(1, 1, h))
            self.pos = nn.Parameter(torch.randn(1, n_frames + 2, h) * 0.02)
            layer = nn.TransformerEncoderLayer(h, cfg.heads, 4 * h, cfg.dropout, batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, cfg.depth)
        else:
            raise ValueError(f"g.arch must be mlp or transformer, got {cfg.arch}")
        self.head = nn.Linear(h, 1)
        self.reg = nn.Linear(h, 1) if cfg.reg_head else None

    def forward(self, frames, action, proprio=None):
        """frames (b, n, F) or (b, n, P, D) for attn; action (b, A); proprio (b, n, p) -> (logit (b,), log-err pred or None)."""
        x = (frames - self.feat_mean) / self.feat_std
        if self.pool is not None:
            x = self.pool(x)
        if self.use_proprio:
            x = torch.cat([x, (proprio - self.prop_mean) / self.prop_std], -1)
        a = (action - self.act_mean) / self.act_std
        if self.arch == "mlp":
            h = self.trunk(torch.cat([x.flatten(1), a], 1) if self.use_action else x.flatten(1))
        else:
            tok = [self.cls.expand(len(x), -1, -1), self.frame_in(x)]
            if self.use_action:
                tok.append(self.act_in(a)[:, None])
            tok = torch.cat(tok, 1)
            h = self.trunk(tok + self.pos[:, :tok.shape[1]])[:, 0]
        return self.head(h).squeeze(-1), (self.reg(h).squeeze(-1) if self.reg is not None else None)

    def proba(self, logit):
        return torch.sigmoid(self.calib_a * logit + self.calib_b)


def proba_np(model, logits):
    """Calibrated probabilities from logits (numpy)."""
    a, b = float(model.calib_a), float(model.calib_b)
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(logits, dtype=np.float64) + b)))


# ---------------------------------------------------------------- training


def _batch(source, s, idx, use_proprio):
    ctx, row = s.ctx[idx], s.row[idx]
    dev = source.device
    return (source.frames(ctx, row), torch.as_tensor(s.action[idx], device=dev),
            source.proprio(ctx) if use_proprio else None)


@torch.no_grad()
def predict(model, source, s, batch_size):
    """-> (logits, log-err predictions in log units or None), numpy, in sample order."""
    model.eval()
    logits, regs = [], []
    for b in range(0, len(s), batch_size):
        idx = np.arange(b, min(b + batch_size, len(s)))
        lo, r = model(*_batch(source, s, idx, model.use_proprio))
        logits.append(lo.cpu().numpy())
        if r is not None:
            regs.append((r * model.logerr_std + model.logerr_mean).cpu().numpy())
    return np.concatenate(logits) if logits else np.zeros(0), (np.concatenate(regs) if regs else None)


def fit(source, train, val, cfg, history_len, seed, log=print):
    """Train g on `train`, early-stopping on val AUPRC; Platt-calibrate on val. -> (model, history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    stats = norm_stats(source, train, cfg, rng)
    p_dim = source.frame_proprio.shape[1] if (cfg.use_proprio and source.frame_proprio is not None) else 0
    model = ErrorPredictor(source.feat_dim, train.action.shape[1], source.n_frames(history_len), stats, cfg,
                           p_dim).to(source.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pi = float(train.y.mean())
    if cfg.class_weight == "auto":
        pw = (1 - pi) / max(pi, 1e-6)
    elif cfg.class_weight in (None, "none"):
        pw = 1.0
    else:
        pw = float(cfg.class_weight)
    pos_weight = torch.tensor(pw, device=source.device)
    best, best_state, bad, history = -math.inf, None, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        perm, tot = rng.permutation(len(train)), 0.0
        for b in range(0, len(perm), cfg.batch_size):
            idx = perm[b:b + cfg.batch_size]
            logit, reg = model(*_batch(source, train, idx, model.use_proprio))
            y = torch.as_tensor(train.y[idx], device=source.device)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
            if reg is not None and cfg.reg_weight > 0:
                tgt = (torch.as_tensor(train.logerr[idx], device=source.device) - model.logerr_mean) / model.logerr_std
                loss = loss + cfg.reg_weight * F.mse_loss(reg, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        vl, _ = predict(model, source, val, cfg.batch_size * 4)
        score = auprc(val.y, vl)
        history.append(dict(epoch=epoch, train_loss=tot / len(train), val_auprc=score, val_auroc=auroc(val.y, vl)))
        if log:
            log(f"    epoch {epoch:2d}  loss {tot / len(train):.4f}  val AUPRC {score:.3f}  AUROC {history[-1]['val_auroc']:.3f}")
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    if cfg.calibrate == "platt":
        vl, _ = predict(model, source, val, cfg.batch_size * 4)
        a, b = platt(vl, val.y)
        model.calib_a.fill_(a)
        model.calib_b.fill_(b)
    return model, dict(history=history, pos_weight=pw, train_positive_rate=pi, best_val_auprc=best)


def platt(logits, y, steps=200):
    """a, b minimising unweighted BCE of sigmoid(a * logit + b) against y."""
    x = torch.as_tensor(logits, dtype=torch.float64)
    t = torch.as_tensor(y, dtype=torch.float64)
    ab = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(ab[0] * x + ab[1], t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(ab[0].detach()), float(ab[1].detach())
