"""Latent actions: a discrete "what happened" read off the FRAMES of a block, and a prior that plays it forward.

    u        = inverse(e_t .. e_{t+L})          frames only (never the memory); one of K codes
    z_{t+i}  = prior(z_{t+i-1}, a, u)           u held for the block; the prior runs on its own predictions
    p(u | z, a)                                 the chooser: what tends to happen next from this memory

With the cups down the frames do not depend on where the ball is, so u can only describe the cups' motion;
what that motion does to the ball has to come from the memory. Lift blocks are left out of training: their
frames show the ball, and u could carry the answer.

u is discrete (straight-through Gumbel-softmax over K codes, no usage balancing), so how many codes end up
used is a result, not a setting.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from ..smt_prelim.blocks import normalize_mem


class LatentActionPrior(nn.Module):
    def __init__(self, base_prior, embed_dim, block=6, n_codes=16, hidden=512):
        super().__init__()
        assert base_prior.mask_net is None, "DELTA-gated priors are not supported"
        self.n_codes, self.block = int(n_codes), int(block)
        self.n_registers, self.d_model = base_prior.n_registers, base_prior.d_model
        act_dim = base_prior.act_in.in_features
        self.act_in = copy.deepcopy(base_prior.act_in)              # warm start from the deterministic prior
        self.rnn = copy.deepcopy(base_prior.rnn)
        self.code = nn.Linear(self.n_codes, self.d_model, bias=False)
        nn.init.normal_(self.code.weight, std=0.5)
        self.inverse = nn.Sequential(nn.LayerNorm((block + 1) * embed_dim),
                                     nn.Linear((block + 1) * embed_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, self.n_codes))
        self.choose = nn.Sequential(nn.Linear(self.n_registers * self.d_model + act_dim, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, self.n_codes))

    def infer(self, frames, tau=1.0, hard=True):
        """frames (B, L+1, E) -> (one-hot-ish code (B, K), logits). Straight-through when hard."""
        logits = self.inverse(frames.flatten(1))
        if self.training:
            return F.gumbel_softmax(logits.float(), tau=tau, hard=hard), logits
        return F.one_hot(logits.argmax(-1), self.n_codes).float(), logits

    def step(self, z, action, code):
        """code: (B, K) one-hot (or soft) or (B,) long."""
        if code.dim() == 1:
            code = F.one_hot(code, self.n_codes).float()
        token = (self.act_in(action) + self.code(code.to(self.code.weight.dtype))).unsqueeze(1)
        out = self.rnn(torch.cat([z, token], dim=1))
        return normalize_mem(out[:, :self.n_registers, :])

    def rollout(self, z, actions, code):
        """Hold `code` for actions.shape[1] steps on the prior's own predictions -> (B, L, K, d)."""
        out = []
        for i in range(actions.shape[1]):
            z = self.step(z, actions[:, i], code)
            out.append(z)
        return torch.stack(out, 1)

    def code_logits(self, z, action):
        return self.choose(torch.cat([z.flatten(1), action], dim=-1))
