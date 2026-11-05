"""The RSSM cell used as a supervised one-step map, with no BPTT.

SMT's `rnn_update` (smt/src/model.py:200-214) takes (memory, next token) -> next memory. The
native RSSM's `obs_step` is the same signature plus an action: it applies prev_action through
the block-GRU Deter cell, then conditions the posterior on the new embedding. Reusing it means
the dynamics being tested here are exactly the ones the R2-Dreamer baseline used -- only the
training signal changes.

`_img_net` (the prior head) is built by RSSM.__init__ but never called: this model never
imagines a latent without an embedding. It is excluded from the optimiser by `parameters()`.
"""
import torch
from torch import nn

from .bridge import RSSM


class OneStepPredictor(nn.Module):
    def __init__(self, rssm_config, embed_dim=1024, act_dim=2):
        super().__init__()
        self.rssm = RSSM(rssm_config, embed_size=int(embed_dim), act_dim=int(act_dim))

    def step(self, stoch, deter, prev_action, embed):
        """(B,S,K), (B,D), (B,A), (B,E) -> (B,S,K), (B,D), (B,S,K).

        reset is always all-False: anchors are independent one-step problems, there is no
        sequence here to reset.
        """
        reset = torch.zeros(deter.shape[0], dtype=torch.bool, device=deter.device)
        return self.rssm.obs_step(stoch, deter, prev_action, embed, reset)

    def parameters(self, recurse=True):
        """Everything except the unused prior head, so the optimiser carries no dead state."""
        dead = {id(p) for p in self.rssm._img_net.parameters()}
        return (p for p in super().parameters(recurse) if id(p) not in dead)

    def dead_parameters(self):
        return self.rssm._img_net.parameters()
