"""The two one-step belief updaters: one that sees the observation, one that does not.

Both are one-step functions, applied once per anchor during training and never unrolled.
They become recurrent only at eval -- SMT's whole thesis, "Pretraining Recurrent Networks
without Recurrence" (submodules/smt/src/model.py:200-214, `RNN.rnn_update`).

The split is what smt_prelim did not have. There, one updater ran on real embeddings while
observing and on its own *predicted* embeddings while imagining, so the imagining path was
never trained on the inputs it actually received. Here imagining has its own module, which
sees no embedding at all and therefore cannot be fed a wrong one.

DELTA (docs/DELTA.md 3): an optional event gate on top of either updater,

    z_hat = normalize_mem(Updater(z_t, ...))           the ungated proposal
    m     = sigmoid(MaskNet(event))                    one value per region, in [0, 1]
    z_new = (1 - m) * z_t + m * z_hat

The mask sees ONLY the event -- (e_{t+1}, a_t) for the posterior, a_t for the prior -- never z,
time or history, so the same event always opens the same regions and stationarity is structural.
The convex form (not z + m * delta, renormed) keeps untouched entries EXACTLY unchanged under
the per-token rms-norm. `step` returns the gated state, so every caller -- observe(), imagine(),
DMT, the probes -- runs the same equation training does. With delta_mask "off" no MaskNet is
built and `step` is the original function, bit for bit.
"""
import torch
from torch import nn
from x_transformers import Encoder

from ..smt_prelim.blocks import normalize_mem

DELTA_MODES = ("off", "token", "dim")


def delta_mode(value):
    """YAML 1.1 reads a bare `off` as False; treat False / None as "off"."""
    return "off" if value is False or value is None else str(value)


class MaskNet(nn.Module):
    """event -> per-region gate logits. A GRU update gate that is blind to the state.

    token: one gate per memory token, (B, K, 1).  dim: one per scalar, (B, K, d).
    The output layer starts at weight 0 and bias `init`, so every gate opens at sigmoid(init)
    (0.95 for init 3.0) -- the updater is free to learn tracking before the L1 pressure, warmed
    up from 0, can shut anything.
    """

    def __init__(self, in_dim, mode, n_registers, d_model, hidden=256, init=3.0):
        super().__init__()
        if mode not in ("token", "dim"):
            raise ValueError(f"MaskNet mode must be 'token' or 'dim', got {mode!r}")
        self.mode, self.n_registers, self.d_model = mode, int(n_registers), int(d_model)
        out = self.n_registers * (self.d_model if mode == "dim" else 1)
        self.net = nn.Sequential(nn.Linear(int(in_dim), hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, out)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, float(init))

    def forward(self, event):
        """event (B, I) -> gate in [0, 1], (B, K, 1) or (B, K, d)."""
        logits = self.out(self.net(event))
        return torch.sigmoid(logits).view(event.shape[0], self.n_registers, -1)


def gate(z, z_hat, m):
    """(1 - m) * z + m * z_hat, in z_hat's dtype. m None = ungated (returns z_hat itself)."""
    if m is None:
        return z_hat
    m = m.to(z_hat.dtype)
    return (1.0 - m) * z.to(z_hat.dtype) + m * z_hat


class PosteriorUpdater(nn.Module):
    """f_post(z_t, e_{t+1}, a_t) -> z_{t+1}. Runs while observing. Memory is K tokens of width d."""

    def __init__(self, embed_dim=1024, act_dim=2, d_model=256, depth=8, n_heads=8,
                 n_registers=10, rotary_pos_emb=True, use_rmsnorm=True,
                 delta_mask="off", delta_mask_init=3.0, delta_mask_hidden=256):
        super().__init__()
        self.n_registers, self.d_model = int(n_registers), int(d_model)
        self.delta_mask = delta_mode(delta_mask)
        assert self.delta_mask in DELTA_MODES, f"delta_mask must be one of {DELTA_MODES}"
        # the gate reads the EVENT only: the observation and the action, never z
        self.mask_net = (None if self.delta_mask == "off" else
                         MaskNet(int(embed_dim) + int(act_dim), self.delta_mask, n_registers,
                                 d_model, hidden=delta_mask_hidden, init=delta_mask_init))
        # Pairwise concatenation, NOT smt_prelim's embed_in(e) + act_in(a). A sum of two linear
        # maps can only ever represent an additive (frame, action) token, so "this action given
        # that frame" is unrepresentable at the input. One Linear over the concatenation costs
        # the same parameters and does not impose that.
        self.token_in = nn.Linear(int(embed_dim) + int(act_dim), d_model)
        self.rnn = Encoder(dim=d_model, depth=depth, heads=n_heads,
                           rotary_pos_emb=rotary_pos_emb, use_rmsnorm=use_rmsnorm)

    def step_with_mask(self, z, embed, action):
        """z (B, K, d), embed (B, E), action (B, A) -> (gated z (B, K, d), mask or None)."""
        event = torch.cat([embed, action], dim=-1)
        token = self.token_in(event).unsqueeze(1)                                # (B, 1, d)
        out = self.rnn(torch.cat([z, token], dim=1))                             # (B, K+1, d)
        z_hat = normalize_mem(out[:, :self.n_registers, :])                      # SMT model.py:211
        m = None if self.mask_net is None else self.mask_net(event)
        return gate(z, z_hat, m), m

    def step(self, z, embed, action):
        """z (B, K, d), embed (B, E), action (B, A) -> (B, K, d). Gated when DELTA is on."""
        return self.step_with_mask(z, embed, action)[0]


class PriorUpdater(nn.Module):
    """f_prior(z_t, a_t) -> z_{t+1}. Runs while imagining; no observation exists to be given.

    `step` takes no embedding argument at all, so insensitivity to observations is structural
    rather than something training has to discover -- which is what makes the imagined half of
    the eval rollout a genuine test of the carried state.
    """

    def __init__(self, act_dim=2, d_model=256, depth=8, n_heads=8,
                 n_registers=10, rotary_pos_emb=True, use_rmsnorm=True,
                 delta_mask="off", delta_mask_init=3.0, delta_mask_hidden=256):
        super().__init__()
        self.n_registers, self.d_model = int(n_registers), int(d_model)
        self.delta_mask = delta_mode(delta_mask)
        assert self.delta_mask in DELTA_MODES, f"delta_mask must be one of {DELTA_MODES}"
        # m_p(a_t): with no observation, the action is the only event there is. At the lift
        # onset the cup has not moved yet, so this is the only signal that can open its region.
        self.mask_net = (None if self.delta_mask == "off" else
                         MaskNet(int(act_dim), self.delta_mask, n_registers, d_model,
                                 hidden=delta_mask_hidden, init=delta_mask_init))
        self.act_in = nn.Linear(int(act_dim), d_model)
        self.rnn = Encoder(dim=d_model, depth=depth, heads=n_heads,
                           rotary_pos_emb=rotary_pos_emb, use_rmsnorm=use_rmsnorm)

    def step_with_mask(self, z, action):
        """z (B, K, d), action (B, A) -> (gated z (B, K, d), mask or None)."""
        token = self.act_in(action).unsqueeze(1)
        out = self.rnn(torch.cat([z, token], dim=1))
        z_hat = normalize_mem(out[:, :self.n_registers, :])
        m = None if self.mask_net is None else self.mask_net(action)
        return gate(z, z_hat, m), m

    def step(self, z, action):
        """z (B, K, d), action (B, A) -> (B, K, d). Gated when DELTA is on."""
        return self.step_with_mask(z, action)[0]
