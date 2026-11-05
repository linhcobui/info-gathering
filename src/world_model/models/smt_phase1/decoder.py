"""The reward decoder: TRAINING ONLY. It is not used anywhere at eval.

SMT's read-out rule, verified from source (submodules/smt/src/model.py:148-159): the sequence
is [z_0..z_{K-1}, x_fut[0..Tf-1]], the slice is [K-1:-1], and since y_fut[j] == x_fut[j] each
token predicts the NEXT token -- the last memory token predicting the first future one. The
target is never visible to the position predicting it.

Phase 1 applies that rule to an interleaved action/observation sequence:

    [ z_0 .. z_9 | a_t | e_t | a_{t+1} | e_{t+1} | ... | a_{t+F-1} | e_{t+F-1} ]
         |          |            |
         v          v            v
        r_t     r_{t+1}      r_{t+2}    ...   read at the memory, then at every ACTION token

    r_t      <- z only
    r_{t+1}  <- z, a_t
    r_{t+k}  <- z, a_t..a_{t+k-1}, e_t..e_{t+k-2}

Memory tokens sit at the front of a causal stack, so every read-out sees all K of them.
Sequence length K + 2F, and F + 1 read-outs r_t .. r_{t+F}.

The property this buys: e_{t+j} first becomes visible when predicting r_{t+j+2}, so the frame
that reveals the ball can never be used to predict its own reward. The summed (e, a) layout
smt_prelim used did leak -- the token predicting r_{t+j+1} held e_{t+j}, and at the lift that
frame already shows the ball.

Deliberate redundancy, matching SMT: e_t is inside z_t (the encoder window ends at t) AND is
re-supplied as a token. So r_t and r_{t+1} are the read-outs that genuinely test the memory,
and later ones progressively less so as they accumulate direct observation tokens.

PAIRED LAYOUT (`layout="paired"`, docs/WINDOW60.md). One token per time step, holding that step's frame
and the action taken AT it:

    pos:  0 .. K-1 | K           K+1               ...  K+F-1
        [ z_0..z_9 | (e_t, a_t) | (e_{t+1}, a_{t+1}) | ... | (e_{t+F-1}, a_{t+F-1}) ]
    r_t at pos K-1;  r_{t+k} at pos K-1+k  (contiguous)
    r_{t+k}  <- z, (e_t, a_t) .. (e_{t+k-1}, a_{t+k-1})      newest frame e_{t+k-1}: a 1-frame lag

The pairing must be (frame, action taken AT it), not the teacher's (frame, action that produced
it): the latter would put the lift action in the same token as the first frame that shows the
ball. At the lift, r_{onset+1} reads at (e_onset, lift action) -- cup still down -- so it is the
one memory-only scored read-out. Stack K + F tokens instead of K + 2F. The token is a small MLP
of the concatenation: Linear([e; a]) alone is just W_e e + W_a a, additive by construction.
"""
import torch
from torch import nn
from x_transformers import Decoder


class RewardDecoder(nn.Module):
    """Causal decoder over [memory ; a_t, e_t, a_{t+1}, e_{t+1}, ...] -> reward logits."""

    def __init__(self, embed_dim=1024, act_dim=2, d_model=256, depth=4, n_heads=8,
                 n_registers=10, rotary_pos_emb=True, use_rmsnorm=True, use_obs=True, head="binary",
                 layout="interleaved"):
        super().__init__()
        self.n_registers, self.d_model = int(n_registers), int(d_model)
        if layout not in ("interleaved", "paired"):
            raise ValueError(f"layout must be 'interleaved' or 'paired', got {layout!r}")
        if layout == "paired" and not use_obs:
            raise ValueError("layout 'paired' pairs each action with a frame; it needs use_obs=True")
        self.layout = str(layout)
        # use_obs=False: the decoder sees [memory ; a_t, a_t+1, ...] and never a future frame, so
        # EVERY read-out must come from the memory. With observation tokens, read-outs from show
        # offset 3 on can copy the ball off an earlier lift frame (offset 1 already shows it).
        self.use_obs = bool(use_obs)
        if self.layout == "paired":
            # one (frame, action) token per step; one kind of token, so no type embedding
            self.step_in = nn.Sequential(nn.Linear(int(embed_dim) + int(act_dim), d_model), nn.SiLU(),
                                         nn.Linear(d_model, d_model))
        else:
            self.act_in = nn.Linear(int(act_dim), d_model)
            self.embed_in = nn.Linear(int(embed_dim), d_model) if self.use_obs else None
            # rotary encodes position, not type, and the two kinds now share one sequence
            self.kind = nn.Embedding(2, d_model)          # 0 = action token, 1 = observation token
        self.decoder = Decoder(dim=d_model, depth=depth, heads=n_heads,
                               rotary_pos_emb=rotary_pos_emb, use_rmsnorm=use_rmsnorm)
        # binary: one logit, "ball under the lifted cup (slot 0)". slot: three logits, "which slot
        # holds the ball". Either way it is the only head -- no embedding prediction.
        if head not in ("binary", "slot", "embed"):
            raise ValueError(f"unknown reward head {head!r}")
        self.head = head
        self.to_reward = nn.Linear(d_model, {"binary": 1, "slot": 3}.get(head, int(embed_dim)))

    def forward(self, z, e_fut, a_fut):
        """z (B, K, d), e_fut (B, F, E), a_fut (B, F, A) -> (B, F+1) logits for r_t .. r_{t+F}.

        e_fut / a_fut are e_t..e_{t+F-1} and a_t..a_{t+F-1} in both layouts.
        """
        batch, fut_len = a_fut.shape[0], a_fut.shape[1]
        if self.layout == "paired":
            steps = self.step_in(torch.cat([e_fut, a_fut], dim=-1))            # (B, F, d)
            out = self.decoder(torch.cat([z, steps], dim=1))                   # (B, K + F, d)
            return self._head(out[:, self.n_registers - 1:])                   # (B, F+1[, 3])
        acts = self.act_in(a_fut) + self.kind.weight[0]                        # (B, F, d)
        if not self.use_obs:
            # [z_0..z_9, a_t, .., a_t+F-1]: r_t at z_{K-1}, r_t+k at a_t+k-1 -- SMT's own [K-1:]
            # slice. r_t+k sees memory and a_t..a_t+k-1, nothing else. e_fut is ignored.
            out = self.decoder(torch.cat([z, acts], dim=1))                    # (B, K + F, d)
            return self._head(out[:, self.n_registers - 1:])                   # (B, F+1[, 3])
        obs = self.embed_in(e_fut) + self.kind.weight[1]                       # (B, F, d)
        inter = torch.stack([acts, obs], dim=2).reshape(batch, 2 * fut_len, -1)  # a_t, e_t, a_t+1, ...
        out = self.decoder(torch.cat([z, inter], dim=1))                       # (B, K + 2F, d)
        heads = torch.cat([out[:, self.n_registers - 1: self.n_registers],     # r_t   at z_{K-1}
                           out[:, self.n_registers::2]], dim=1)                # r_t+1.. at a-tokens
        return self._head(heads)                                               # (B, F+1[, 3])

    def _head(self, hidden):
        out = self.to_reward(hidden)
        return out.squeeze(-1) if self.head == "binary" else out

    def ball_logit(self, logits):
        """Any head -> the logit of "ball under the lifted cup (slot 0)", so >0 means ball.

        slot head: log p(slot 0) - log p(slot 1 or 2). Lets every binary metric and the onset
        probe compare the two heads on the same question.
        """
        if self.head == "embed":
            raise RuntimeError("the embed head predicts frame embeddings, not a ball logit")
        if self.head == "binary":
            return logits
        lp = torch.log_softmax(logits.float(), dim=-1)
        return lp[..., 0] - torch.logsumexp(lp[..., 1:], dim=-1)
