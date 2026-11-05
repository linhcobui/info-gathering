"""A quantile error predictor: how bad *could* the WM's error get here, not just whether it will exceed lambda.

From the same inputs as g (H context frames + actions), predict quantiles q_1 < q_2 < ... of log err_t with the
pinball (quantile) loss. For quantile level tau the loss of a prediction q against the realised x is
    tau * (x - q)        if x > q   (the error came out worse than predicted: a miss, weight tau)
    (1 - tau) * (q - x)  if x <= q  (it came out better: a false alarm, weight 1 - tau)
so at tau = 0.9 a miss costs 9x a false alarm: the upper quantile stays high wherever a big error is possible, and
is not dragged down by the times it doesn't happen.

Detector signal: spread_t = q_hi - q_lo (log units, i.e. log of the ratio "could get this bad" / "typically is").
A slow level that raises the error everywhere (e.g. objects close to the camera) moves both quantiles and cancels;
a moment whose outcome is open (several outcomes, a deterministic WM picks one) widens the spread.
Monotone quantiles: q_1 is predicted, the rest add positive (softplus) increments.
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .error_predictor import ErrorPredictor, _batch, norm_stats


class QuantilePredictor(ErrorPredictor):
    def __init__(self, feat_dim, action_dim, n_frames, stats, cfg, taus, proprio_dim=0):
        super().__init__(feat_dim, action_dim, n_frames, stats, cfg, proprio_dim)
        self.register_buffer("taus", torch.as_tensor(taus, dtype=torch.float32))
        self.head = nn.Linear(cfg.hidden, len(taus))
        self.reg = None

    def forward(self, frames, action, proprio=None):
        raw, _ = super().forward(frames, action, proprio)         # head is (h -> n_tau); super squeezes nothing
        return raw

    def quantiles(self, raw):
        """Standardised log-error quantiles, monotone in tau."""
        return torch.cat([raw[:, :1], raw[:, :1] + torch.cumsum(F.softplus(raw[:, 1:]), 1)], 1)


def pinball(q, x, taus):
    d = x[:, None] - q
    return torch.maximum(taus * d, (taus - 1) * d).mean()


@torch.no_grad()
def predict_quantiles(model, source, s, batch_size):
    """-> (N, n_tau) quantiles of log err (natural log units)."""
    model.eval()
    out = []
    for b in range(0, len(s), batch_size):
        idx = np.arange(b, min(b + batch_size, len(s)))
        q = model.quantiles(model(*_batch(source, s, idx, model.use_proprio)))
        out.append((q * model.logerr_std + model.logerr_mean).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(model.taus)))


def fit_quantiles(source, train, val, cfg, taus, history_len, seed, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    stats = norm_stats(source, train, cfg, rng)
    p_dim = source.frame_proprio.shape[1] if (cfg.use_proprio and source.frame_proprio is not None) else 0
    model = QuantilePredictor(source.feat_dim, train.action.shape[1], source.n_frames(history_len), stats, cfg, taus,
                              p_dim).to(source.device)
    taus_t = model.taus
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, bad, history = math.inf, None, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        perm, tot = rng.permutation(len(train)), 0.0
        for b in range(0, len(perm), cfg.batch_size):
            idx = perm[b:b + cfg.batch_size]
            q = model.quantiles(model(*_batch(source, train, idx, model.use_proprio)))
            x = (torch.as_tensor(train.logerr[idx], device=source.device) - model.logerr_mean) / model.logerr_std
            loss = pinball(q, x, taus_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        qv = predict_quantiles(model, source, val, cfg.batch_size * 4)
        xv = val.logerr[:, None]
        vl = float(np.mean(np.maximum(np.asarray(taus) * (xv - qv), (np.asarray(taus) - 1) * (xv - qv))))
        cover = (xv[:, 0] <= qv[:, -1]).mean()
        history.append(dict(epoch=epoch, train_loss=tot / len(train), val_pinball=vl, val_cover_top=float(cover)))
        if log:
            log(f"    epoch {epoch:2d}  pinball {tot / len(train):.4f}  val {vl:.4f}  val coverage of q{taus[-1]} {cover:.3f}")
        if vl < best:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, dict(history=history, best_val_pinball=best)
