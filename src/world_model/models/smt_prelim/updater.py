"""SMT's belief updater: a transformer over [memory tokens ; new observation ; action].

This is `RNN.rnn_update` (smt/src/model.py:200-214) with an action token added, because a
frame's successor depends on the action taken and SMT models unconditioned sequences.

The point of using this rather than the RSSM cell: here the memory slots attend to the new
observation directly, so the observation has a wide path into the state. The RSSM routes every
observation through a 32x16 categorical sample instead -- and in these runs that categorical
collapsed to uniform, making the carried state provably blind to what it saw (docs/SMT_PRELIM.md 6.4).
"""
import torch
from torch import nn
from x_transformers import Encoder

from .blocks import normalize_mem


class SmtBeliefUpdater(nn.Module):
    """(z_t, e_{t+1}, a_t) -> z_{t+1}. Memory is K tokens of width d_model."""

    def __init__(self, embed_dim=1024, act_dim=2, d_model=256, depth=8, n_heads=8,
                 n_registers=10, rotary_pos_emb=True, use_rmsnorm=True):
        super().__init__()
        self.n_registers, self.d_model = int(n_registers), int(d_model)
        self.embed_in = nn.Linear(embed_dim, d_model)
        self.act_in = nn.Linear(act_dim, d_model, bias=False)
        self.rnn = Encoder(dim=d_model, depth=depth, heads=n_heads,
                           rotary_pos_emb=rotary_pos_emb, use_rmsnorm=use_rmsnorm)

    def step(self, z, embed, action):
        """z (B, K, d), embed (B, E), action (B, A) -> (B, K, d)."""
        token = (self.embed_in(embed) + self.act_in(action)).unsqueeze(1)   # (B, 1, d)
        out = self.rnn(torch.cat([z, token], dim=1))                        # (B, K+1, d)
        return normalize_mem(out[:, :self.n_registers, :])                  # SMT model.py:211
