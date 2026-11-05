"""SMT's loss and windowing helpers.

Semantics follow submodules/smt/src/model.py; nothing is added. The attention stacks
themselves come from x_transformers, the library SMT is written against.
"""
import torch
import torch.nn.functional as F


def sliding_windows(x, ctx_len):
    """(B, T, C) -> (B, T, ctx_len, C); window t is x[t-ctx_len+1 .. t], left-padded with zeros."""
    padded = F.pad(x, (0, 0, ctx_len - 1, 0))
    return padded.unfold(dimension=1, size=ctx_len, step=1).transpose(-1, -2)


def window_mask(steps, ctx_len, is_first):
    """(B, T, ctx_len) bool, True = a real step (x_transformers' polarity: True = attend).

    False on the left pad and on anything before the most recent episode start, so a window
    never reaches across an is_first boundary.
    """
    device = is_first.device
    index = torch.arange(steps, device=device)[:, None] - (ctx_len - 1) + torch.arange(ctx_len, device=device)
    starts = torch.where(is_first, torch.arange(steps, device=device),
                         torch.zeros((), dtype=torch.long, device=device))
    starts = starts.cummax(dim=1).values                    # (B, T) first step of the current episode
    return (index >= 0) & (index[None] >= starts[..., None])


def uniformity_loss(z, t=2.0, sample=512):
    """Gaussian-potential uniformity over normalised vectors; port of smt/src/model.py:41-53."""
    z = z / (z.norm(dim=-1, keepdim=True) + torch.finfo(z.dtype).eps)
    z = z.reshape(-1, z.shape[-1]).float()
    index = torch.randperm(len(z), device=z.device)[:sample]
    return torch.pdist(z[index], p=2).pow(2).mul(-t).exp().mean().log()


def normalize_mem(z):
    """SMT's normalize_mem with rms_norm=True, fsq off (model.py:79-87)."""
    return F.rms_norm(z, z.shape[-1:])


def masked_mean(loss, mask):
    """smt/src/model.py:55-59."""
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def state_feat(deter, logit):
    """(..., D), (..., S, K) -> (..., D + S*K).

    Probabilities rather than raw logits, so the teacher and the predictor only have to agree
    on distributions -- which is what the KL term enforces -- not on logit scale.
    """
    probs = torch.softmax(logit.float(), dim=-1)
    return torch.cat([deter.float(), probs.flatten(-2)], dim=-1)
