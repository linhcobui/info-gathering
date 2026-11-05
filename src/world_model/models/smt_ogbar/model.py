"""OGBar-SMT (docs/OGBAR_SMT.md): a stochastic SMT world model on frozen DINOv2 patch tokens.

    z_t         belief, K=10 tokens x d=256, each token rms-normalised (SMT's normalize_mem)
    x_t         DINOv2 ViT-S/14 patch tokens of step t, 256 x 384 (cached, frozen)
    a_t         the 5 raw 5-D actions taken AT step t, concatenated (25)

Parts (stage 1; the readout R is stage 2 and not here):
  FramePool    x_s -> one d-vector per step: multi-head attention pool over the 256 patches with
               learned queries. Written factored: head h's score for patch p is u_h . x_p + b_hp
               (= q_h . W_k x_p, bias-free keys, plus a learned per-patch bias), and the value
               projection is applied AFTER the weighted sum (W_v is linear, so sum_p w_p W_v x_p =
               W_v sum_p w_p x_p). Same function as a standard attention pool, but it never
               materialises a projected copy of every patch, so batch 128 x 200 steps fits.
  teacher      200 step tokens ending at t + K registers -> z*_t. Bidirectional (smt_prelim's
               TeacherEncoder, reused). Step token = Linear([pool(x_s) ; a_{s-1}]): each frame with
               the action that PRODUCED it, as SMT phase 1, so the window ending at t+1 holds only
               what Q is given (x_{t+1}, a_t).
  decoder      causal over [z*_t ; (x_{t+j}, a_{t+j}) j = 0..F-1] (paired layout, docs/WINDOW60.md); the
               output at token j predicts ALL 256 patches of x_{t+j+1}. Frame t+j+1 is never visible
               at the position that predicts it. A random subset of future steps is scored.
  PatchHead    one step's hidden state -> 256 x 384: the state becomes M memory tokens, 256 learned
               patch queries cross-attend to them, then an MLP and a Linear to 384.
  Q            posterior: [z_t ; x_t patches (+a_t) ; x_{t+1} patches] -> mu_Q, log sigma_Q
  P            prior:     [z_t ; x_t patches (+a_t)]                   -> mu_P, log sigma_P
               Both: SMT's Encoder trunk, the K register outputs -> two Linear heads. A sample is
               normalize_mem(mu + sigma * noise), mu itself rms-normalised. Q and P have their own
               patch projections (they do not share FramePool), so the prior loss reaches P only.

Losses (no others; no KL divergence anywhere, user 2026-10-18; no free bits):
  align    (coef_align > 0; PRIVILEGED, batch["cview"]) every pair of anchors in the batch is pulled together with
           weight w_ij = exp(-(c_i - c_j)^2 / (2 align_alpha^2)), c = c_view at the anchor step (cm): 1 cm apart
           pulls harder than 1.5 cm. pull = sum_ij w_ij 0.25 mean (z_i - z_j)^2 / sum_ij w_ij -> teacher, pool
  align_push  (coef_align_push > 0; PRIVILEGED) the uniformity loss on unit-norm tokens (token k of i against token k
           of j), each pair weighted by 1 - w_ij: log sum (1 - w) exp(-t ||u_i - u_j||^2) / sum (1 - w) (Align-Soft's
           push). Plain uniformity also pushes the same-CoM pairs the pull merges: at coef_align = coef_unif = 0.1 it
           spread z by everything but c (far/near 1.03 at step 500), at coef_align 1.0 z collapsed (z_spread 1e-4)
  decoder  MSE of predicted vs real patch embeddings of the scored future frames -> teacher, pool, decoder
  post     MSE of Q's SAMPLE vs z*_{t+1}, target live -> Q, teacher (and the pool under it)
  prior    Gaussian NLL of z*_{t+1} (frozen) under P(z*_t frozen, x_t, a_t) -> P only
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from x_transformers import Decoder, Encoder

from ..smt_prelim.blocks import normalize_mem, uniformity_loss
from ..smt_prelim.teacher import TeacherEncoder

DEFAULTS = dict(
    n_patches=256, patch_dim=384, d_model=256, n_registers=10, n_heads=8,
    encoder_depth=8, decoder_depth=4, posterior_depth=8, prior_depth=8, rotary_pos_emb=True, use_rmsnorm=True,
    window=200, future=200, pool_heads=8, head_mem_tokens=8, head_mlp_mult=2,
    updater_obs_input="current", z_dist="gaussian", prior_loss="nll", prior_mixture=1,
    sigma_min=0.01, sigma_max=1.0, sigma_init=0.1, beta_nll=0.0, input_noise=0.0,
    decoder_loss="plain", decoder_steps_scored=32, checkpoint_activations=True, head_chunk=32,
    coef_decoder=1.0, coef_post=1.0, coef_prior=1.0, coef_unif=0.0, unif_sample=512, unif_mode="per_token", normalize_actions=False,
    coef_align=0.0, align_alpha=2.0, coef_align_push=0.0, align_push_t=2.0,
)


def com_align(z, c, alpha, t=2.0):
    """Privileged CoM-similarity pull and (1 - w)-weighted push (module doc, "align", "align_push").
    z (B, K, d), c (B,) cm. Returns (pull, push, w, D2)."""
    flat = z.flatten(1).float()
    with torch.autocast("cuda", enabled=False):                            # fp32: bf16 would swamp near-pair distances
        sq = flat.pow(2).sum(-1)                                           # no sqrt: cdist's backward is NaN at 0
        d2 = (sq[:, None] + sq[None] - 2 * flat @ flat.t()).clamp(min=0) / flat.shape[1]   # mean of (z_i - z_j)^2
    w = torch.exp(-(c.float()[:, None] - c.float()[None]).pow(2) / (2 * float(alpha) ** 2))
    off = 1 - torch.eye(len(c), device=z.device)
    w = w * off
    with torch.autocast("cuda", enabled=False):
        u = F.normalize(z.float(), dim=-1)
        du = (2 - 2 * torch.einsum("ikd,jkd->kij", u, u)).clamp(min=0)       # (K, B, B) per-token squared distance
        wp = (1 - w) * off
        push = (wp * du.mul(-float(t)).exp()).sum() / (wp.sum() * du.shape[0]).clamp(min=1e-8)
    return 0.25 * (w * d2).sum() / w.sum().clamp(min=1e-8), push.clamp(min=1e-30).log(), w, d2


def per_token_uniformity(z, t=2.0):
    """SMT's Gaussian-potential uniformity, applied to each memory token separately across the batch:
    token k's B vectors (one per anchor) are spread apart, then the K values are averaged. Only different
    anchors can satisfy it -- unlike the pooled form, which the K tokens of ONE anchor can satisfy by
    pointing in different directions while every anchor holds the same belief (run 2, 2026-10-15)."""
    z = F.normalize(z.float(), dim=-1).transpose(0, 1)                    # (K, B, d)
    d2 = torch.cdist(z, z).pow(2)                                          # (K, B, B)
    B = z.shape[1]
    off = ~torch.eye(B, dtype=torch.bool, device=z.device)
    return torch.stack([d2[k][off].mul(-t).exp().mean().log() for k in range(z.shape[0])]).mean()


class FramePool(nn.Module):
    """(..., P, C) patch tokens -> (..., d). Multi-head attention pool, factored (see module doc)."""

    def __init__(self, n_patches, patch_dim, d_model, heads):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.score = nn.Linear(patch_dim, heads, bias=False)               # u_h = W_k,h^T q_h
        self.pos_bias = nn.Parameter(torch.zeros(heads, n_patches))         # b_hp, learned per patch
        self.value = nn.Parameter(torch.randn(heads, patch_dim, d_model // heads) / math.sqrt(patch_dim))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        lead = x.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])                                    # (N, P, C)
        w = torch.softmax((self.score(x) + self.pos_bias.t()).float(), dim=1).to(x.dtype)   # (N, P, h)
        pooled = torch.einsum("nph,npc->nhc", w, x)                         # (N, h, C)
        v = torch.einsum("nhc,hcd->nhd", pooled, self.value.to(x.dtype))    # (N, h, d/h)
        return self.out(v.reshape(v.shape[0], -1)).reshape(*lead, -1)


class PatchHead(nn.Module):
    """(N, d) hidden states -> (N, 256, 384): learned patch queries cross-attend to M state tokens."""

    def __init__(self, d_model, n_patches, patch_dim, mem_tokens=8, heads=8, mlp_mult=2):
        super().__init__()
        self.m, self.d = mem_tokens, d_model
        self.to_mem = nn.Linear(d_model, mem_tokens * d_model)
        self.queries = nn.Parameter(torch.randn(n_patches, d_model) * 0.02)
        self.norm_q, self.norm_kv = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, mlp_mult * d_model), nn.GELU(),
                                 nn.Linear(mlp_mult * d_model, d_model))
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, patch_dim))

    def forward(self, h):
        mem = self.norm_kv(self.to_mem(h).view(h.shape[0], self.m, self.d))
        q = self.queries.unsqueeze(0).expand(h.shape[0], -1, -1)
        x = q + self.attn(self.norm_q(q), mem, mem, need_weights=False)[0]
        x = x + self.mlp(x)
        return self.out(x)


class Updater(nn.Module):
    """Q (sees_next=True) or P: SMT's Encoder trunk over [z ; frame tokens] -> (mu, log sigma), each (B, K, d)."""

    def __init__(self, cfg, act_dim, depth, sees_next):
        super().__init__()
        d, P, C = cfg["d_model"], cfg["n_patches"], cfg["patch_dim"]
        self.K, self.sees_next = cfg["n_registers"], bool(sees_next)
        self.obs_current = cfg["updater_obs_input"] == "current"
        if self.obs_current:
            self.cur_in = nn.Linear(C + act_dim, d)                          # each patch of x_t with a_t
            self.cur_pos = nn.Parameter(torch.randn(P, d) * 0.02)
        else:
            self.act_in = nn.Linear(act_dim, d)                              # a_t alone, one token
        if self.sees_next:
            self.next_in = nn.Linear(C, d)
            self.next_pos = nn.Parameter(torch.randn(P, d) * 0.02)
        self.trunk = Encoder(dim=d, depth=depth, heads=cfg["n_heads"], rotary_pos_emb=cfg["rotary_pos_emb"],
                             use_rmsnorm=cfg["use_rmsnorm"], attn_flash=True)
        self.to_mu = nn.Linear(d, d)
        self.to_log_sigma = nn.Linear(d, d)
        nn.init.zeros_(self.to_log_sigma.weight)
        nn.init.constant_(self.to_log_sigma.bias, math.log(cfg["sigma_init"]))
        self.lo, self.hi = math.log(cfg["sigma_min"]), math.log(cfg["sigma_max"])

    def forward(self, z, x_cur, a, x_next=None):
        parts = [z]
        if self.obs_current:
            act = a.unsqueeze(1).expand(-1, x_cur.shape[1], -1).to(x_cur.dtype)
            parts.append(self.cur_in(torch.cat([x_cur, act], -1)) + self.cur_pos)
        else:
            parts.append(self.act_in(a).unsqueeze(1))
        if self.sees_next:
            parts.append(self.next_in(x_next) + self.next_pos)
        h = self.trunk(torch.cat(parts, 1))[:, :self.K]
        return normalize_mem(self.to_mu(h)), self.to_log_sigma(h).float().clamp(self.lo, self.hi)


class SmtOgbar(nn.Module):
    def __init__(self, action_dim=25, action_stats=None, **cfg):
        super().__init__()
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown smt_ogbar options {sorted(unknown)}")
        self.cfg = c = {**DEFAULTS, **cfg}
        if c["prior_loss"] != "nll":
            raise ValueError("prior_loss must be 'nll': no KL divergence in any loss (user, 2026-10-18)")
        if int(c["prior_mixture"]) != 1:
            raise NotImplementedError("prior_mixture > 1 is listed in docs/OGBAR_SMT.md but not built yet")
        assert c["updater_obs_input"] in ("current", "none") and c["z_dist"] in ("gaussian", "deterministic")
        assert c["decoder_loss"] in ("plain", "delta")
        d, K, A = c["d_model"], c["n_registers"], int(action_dim)
        self.K, self.A = K, A
        shared = dict(d_model=d, n_heads=c["n_heads"], n_registers=K, rotary_pos_emb=c["rotary_pos_emb"],
                      use_rmsnorm=c["use_rmsnorm"])
        self.pool = FramePool(c["n_patches"], c["patch_dim"], d, c["pool_heads"])
        # the teacher's own token_in reads [pool(x_s) ; a_{s-1}] (embed_dim = d, act_dim = A)
        self.teacher = TeacherEncoder(embed_dim=d, act_dim=A, depth=c["encoder_depth"], heads=False, **shared)
        # same stack, built with x_transformers' flash path (SDPA): the attention maps are never stored
        self.teacher.encoder = Encoder(dim=d, depth=c["encoder_depth"], heads=c["n_heads"],
                                       rotary_pos_emb=c["rotary_pos_emb"], use_rmsnorm=c["use_rmsnorm"], attn_flash=True)
        self.dec_step_in = nn.Sequential(nn.Linear(d + A, d), nn.SiLU(), nn.Linear(d, d))
        self.decoder = Decoder(dim=d, depth=c["decoder_depth"], heads=c["n_heads"],
                               rotary_pos_emb=c["rotary_pos_emb"], use_rmsnorm=c["use_rmsnorm"], attn_flash=True)
        self.head = PatchHead(d, c["n_patches"], c["patch_dim"], c["head_mem_tokens"], c["n_heads"], c["head_mlp_mult"])
        self.posterior = Updater(c, A, c["posterior_depth"], sees_next=True)
        self.prior = Updater(c, A, c["prior_depth"], sees_next=False)
        if c["normalize_actions"]:      # per-dimension training mean / std (filled from action_stats or the checkpoint)
            st = action_stats or {"mean": [0.0] * A, "std": [1.0] * A}
            self.register_buffer("action_mean", torch.as_tensor(st["mean"], dtype=torch.float32))
            self.register_buffer("action_std", torch.as_tensor(st["std"], dtype=torch.float32))

    def norm_action(self, a):
        """Raw 25-D actions -> what the model reads. Every caller (training, eval scripts) goes through this once."""
        return (a - self.action_mean) / self.action_std if self.cfg["normalize_actions"] else a

    def modules_by_term(self):
        return dict(pool=self.pool, teacher=self.teacher, decoder=nn.ModuleList([self.dec_step_in, self.decoder, self.head]),
                    posterior=self.posterior, prior=self.prior)

    # ------------------------------------------------------------------ pieces
    def sample(self, mu, log_sigma):
        if self.cfg["z_dist"] == "deterministic":
            return mu
        return normalize_mem(mu + log_sigma.exp().to(mu.dtype) * torch.randn_like(mu))

    def teacher_windows(self, pooled, action, ends):
        """z* for windows ending at each step in `ends` (B, n). pooled (B, S, d) -> (B, n, K, d)."""
        B, S, _ = pooled.shape
        W = self.cfg["window"]
        prev = torch.cat([torch.zeros_like(action[:, :1]), action[:, :-1]], 1)            # a_{s-1}
        idx = ends.unsqueeze(-1) - (W - 1) + torch.arange(W, device=ends.device)          # (B, n, W)
        mask = idx >= 0
        idx = idx.clamp(min=0)
        n = ends.shape[1]
        e = torch.gather(pooled, 1, idx.reshape(B, -1, 1).expand(-1, -1, pooled.shape[-1])).view(B * n, W, -1)
        a = torch.gather(prev, 1, idx.reshape(B, -1, 1).expand(-1, -1, prev.shape[-1])).view(B * n, W, -1)
        z = self.teacher.tokens(e, a.to(e.dtype), mask.view(B * n, W))
        return z.view(B, n, self.K, -1)

    def decode(self, z, pooled, action, t, length, steps=None):
        """Hidden state predicting x_{t+j+1} for j in `steps` (B, m) (default: all j).

        Returns (hidden (B, m, d), valid (B, m)). Token j = (x_{t+j}, a_{t+j}); the output at token j
        has seen z, x_t..x_{t+j} and a_t..a_{t+j}, never x_{t+j+1}.
        """
        B, S, d = pooled.shape
        Fw = self.cfg["future"]
        idx = t.unsqueeze(1) + torch.arange(Fw, device=t.device)                        # (B, F) step t+j
        ok = idx + 1 < length.unsqueeze(1)                                               # x_{t+j+1} exists
        idx = idx.clamp(max=S - 1)
        e = torch.gather(pooled, 1, idx.unsqueeze(-1).expand(-1, -1, d))
        a = torch.gather(action, 1, idx.unsqueeze(-1).expand(-1, -1, action.shape[-1]))
        tok = self.dec_step_in(torch.cat([e, a.to(e.dtype)], -1))
        out = self.decoder(torch.cat([z, tok], 1))[:, self.K:]                            # (B, F, d)
        if steps is None:
            return out, ok
        return torch.gather(out, 1, steps.unsqueeze(-1).expand(-1, -1, d)), torch.gather(ok, 1, steps)

    def _head_error(self, hid, patch, tgt_i):
        """Per scored step: MSE over all patches of the predicted vs real x_{t+j+1} (or its change)."""
        B, m, d = hid.shape
        ar = torch.arange(B, device=hid.device).unsqueeze(1)
        pred = self.head(hid.reshape(-1, d)).view(B, m, *patch.shape[2:])
        target = patch[ar, tgt_i]
        if self.cfg["decoder_loss"] == "delta":
            target = target - patch[ar, tgt_i - 1]
        return (pred.float() - target.float()).pow(2).mean((-1, -2))

    # ------------------------------------------------------------------ training
    def forward(self, batch, diagnostics=False):
        c = self.cfg
        patch, length, t = batch["patch"], batch["length"], batch["t"]
        action = self.norm_action(batch["action"])
        B, S = patch.shape[:2]
        ar = torch.arange(B, device=patch.device)
        # activation checkpointing (training only): teacher, Q, P and the patch head recompute their layers in
        # the backward pass instead of storing them -- identical results; at batch 128 they hold ~28 GB otherwise
        ck = (lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if (self.training and c["checkpoint_activations"]) \
            else (lambda f, *a: f(*a))
        pooled = self.pool(patch)                                                          # (B, S, d)
        zz = ck(self.teacher_windows, pooled, action, torch.stack([t, t + 1], 1))
        z_t, z_next = zz[:, 0], zz[:, 1]

        # decoder: a random subset of the valid future steps, all 256 patches each
        Fw, m = c["future"], int(c["decoder_steps_scored"])
        valid = (t.unsqueeze(1) + torch.arange(Fw, device=t.device) + 1) < length.unsqueeze(1)
        steps = torch.where(valid, torch.rand(B, Fw, device=t.device), -1.0).topk(min(m, Fw), dim=1).indices
        hid, ok = self.decode(z_t, pooled, action, t, length, steps)
        tgt_i = (t.unsqueeze(1) + steps + 1).clamp(max=S - 1)
        hc = int(c["head_chunk"])                         # anchors per head call: bounds the head's recompute peak
        err = torch.cat([ck(self._head_error, hid[i:i + hc], patch[i:i + hc], tgt_i[i:i + hc])
                         for i in range(0, B, hc)])                                           # (B, m)
        with torch.no_grad():   # baseline: repeat x_{t+j} (plain) = predict no change (delta); same error either way
            copy_err = (patch[ar.unsqueeze(1), tgt_i].float() - patch[ar.unsqueeze(1), tgt_i - 1].float()).pow(2).mean((-1, -2))
        okf = ok.float()
        n_ok = okf.sum().clamp(min=1)
        dec_loss = (err * okf).sum() / n_ok

        # updaters: one step, never unrolled
        x_t, x_n, a_t = patch[ar, t], patch[ar, t + 1], action[ar, t]
        q_in = z_t
        p_in = z_t.detach()
        if c["input_noise"] > 0:
            q_in = normalize_mem(q_in + c["input_noise"] * torch.randn_like(q_in))
            p_in = normalize_mem(p_in + c["input_noise"] * torch.randn_like(p_in))
        mu_q, ls_q = ck(self.posterior, q_in, x_t, a_t, x_n)
        z_q = self.sample(mu_q, ls_q)
        post = (z_q.float() - z_next.float()).pow(2).mean()                               # live target

        mu_p, ls_p = ck(self.prior, p_in, x_t, a_t)
        tgt = z_next.detach().float()
        if c["z_dist"] == "deterministic":
            prior = (mu_p.float() - tgt).pow(2).mean()
        else:
            nll = 0.5 * ((tgt - mu_p.float()) / ls_p.exp()).pow(2) + ls_p
            if c["beta_nll"] > 0:
                nll = nll * (2 * ls_p).exp().detach().pow(c["beta_nll"])
            prior = nll.mean()

        if c["coef_unif"] <= 0:
            unif = z_t.sum() * 0.0
        elif c["unif_mode"] == "per_token":                 # each memory token spread across anchors (user 2026-10-15)
            unif = per_token_uniformity(z_t)
        else:                                               # "pooled": phase 1's form, all K*B token vectors together
            unif = uniformity_loss(z_t, sample=int(c["unif_sample"]))
        if c["coef_align"] > 0:
            if "cview" not in batch:
                raise KeyError("coef_align > 0 needs batch['cview']: OgbarSmtAnchors(..., privileged=True)")
            align, align_push, w_al, d2_al = com_align(z_t, batch["cview"], c["align_alpha"], c["align_push_t"])
        else:
            align, w_al = z_t.sum() * 0.0, None
            align_push = align
        total = (c["coef_decoder"] * dec_loss + c["coef_post"] * post + c["coef_prior"] * prior
                 + c["coef_unif"] * unif + c["coef_align"] * align + c["coef_align_push"] * align_push)
        var = z_next.float().var().clamp(min=1e-8)
        with torch.no_grad():
            metrics = {
                "decoder_copy_baseline": (copy_err * okf).sum() / n_ok,
                "post_mu_r2": 1 - (mu_q.float() - z_next.float()).pow(2).mean() / var,
                "prior_mu_r2": 1 - (mu_p.float() - tgt).pow(2).mean() / var,
                "sigma_q": ls_q.exp().mean(), "sigma_p": ls_p.exp().mean(),
                "teacher_step_change": (z_next.float() - z_t.float()).pow(2).mean() / var,
                "decoder_scored": n_ok / B,
            }
            # collapse monitors: does z* differ between anchors, and does the decoder use it?
            metrics["z_spread"] = z_t.float().var(0).mean()                  # 0 = the same belief for every anchor
            metrics["sigma_p_spread"] = ls_p.exp().mean((1, 2)).std()
            if w_al is not None:      # does z separate by CoM? mean z distance, pairs < 1 cm apart vs > 6 cm apart
                dc = (batch["cview"].float()[:, None] - batch["cview"].float()[None]).abs()
                off = ~torch.eye(B, dtype=torch.bool, device=z_t.device)
                near, far = (dc < 1) & off, dc > 6
                if near.any() and far.any():
                    metrics["align_far_over_near"] = d2_al[far].mean() / d2_al[near].mean().clamp(min=1e-8)
            if diagnostics:
                hid_o, _ = self.decode(z_t.roll(1, 0), pooled, action, t, length, steps)   # another anchor's z*
                err_o = torch.cat([self._head_error(hid_o[i:i + hc], patch[i:i + hc], tgt_i[i:i + hc])
                                   for i in range(0, B, hc)])
                metrics["memory_gain"] = ((err_o - err.detach()) * okf).sum() / n_ok   # > 0: own z* helps
        return total, dict(decoder=dec_loss, post=post, prior=prior, unif=unif, align=align,
                           align_push=align_push), metrics
