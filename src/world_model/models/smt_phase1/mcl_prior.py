"""A prior that commits to ONE of K possible next steps, so imagination can branch.

The phase-1 prior is trained by squared error to the next teacher state. When the next step can be any
of several things (no swap, or one of three swaps) and the action -- only a hold/lift flag -- does not
say which, squared error is minimised by their average: imagined holds become a frozen smear.

Here the prior takes one extra input, a code c in {0..K-1}:

    z_next = prior(z, a, c)          the code is added to the action token
    p(c | z, a)                      a small classifier: how likely each kind of change is from here

Training (multiple-choice learning): run all K codes, train the one that lands closest to the real next
state (the losers get a small share, so none drifts away from the data), and train p(c | z, a) to predict
which code won. No labels: the codes are discovered from the real next states.
Imagining: c ~ p(c | z, a) every step, or c chosen by hand.

The memory z is still the only state; the code is a per-step choice and is not carried.
"""
import copy

import torch
from torch import nn

from ..smt_prelim.blocks import normalize_mem


class MCLPrior(nn.Module):
    def __init__(self, base_prior, n_codes=8, code_std=0.5):
        super().__init__()
        assert base_prior.mask_net is None, "DELTA-gated priors are not supported"
        self.n_codes = int(n_codes)
        self.n_registers, self.d_model = base_prior.n_registers, base_prior.d_model
        self.act_in = copy.deepcopy(base_prior.act_in)          # warm start from the deterministic prior
        self.rnn = copy.deepcopy(base_prior.rnn)
        self.code = nn.Embedding(self.n_codes, self.d_model)
        nn.init.normal_(self.code.weight, std=code_std)
        act_dim = base_prior.act_in.in_features
        self.choose = nn.Sequential(nn.Linear(self.n_registers * self.d_model + act_dim, 512), nn.SiLU(),
                                    nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, self.n_codes))

    def step(self, z, action, code):
        """z (B, K, d), action (B, A), code (B,) long -> (B, K, d)."""
        token = (self.act_in(action) + self.code(code)).unsqueeze(1)
        out = self.rnn(torch.cat([z, token], dim=1))
        return normalize_mem(out[:, :self.n_registers, :])

    def all_codes(self, z, action):
        """Every code at once: (B, C, K, d)."""
        b, c = z.shape[0], self.n_codes
        codes = torch.arange(c, device=z.device).repeat(b)
        out = self.step(z.repeat_interleave(c, 0), action.repeat_interleave(c, 0), codes)
        return out.view(b, c, *z.shape[1:])

    def code_logits(self, z, action):
        return self.choose(torch.cat([z.flatten(1), action], dim=-1))

    def sample_step(self, z, action, generator=None):
        """c ~ p(c | z, a), then step. Returns (z_next, code)."""
        probs = torch.softmax(self.code_logits(z, action).float(), -1)
        code = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        return self.step(z, action, code), code
