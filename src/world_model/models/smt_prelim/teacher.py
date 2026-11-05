"""SMT's teacher: a bidirectional encoder producing the memory, and a causal decoder reading it.

Uses x_transformers' Encoder (bidirectional) and Decoder (causal) with rotary position
embeddings and RMSNorm -- the same stack and the same flags as submodules/smt/src/model.py:116-118.

Two changes are forced by this setting, and there are no others:
  * tokens are continuous frame embeddings, so the token embedding is a Linear rather than
    SMT's nn.Embedding over a 256-word vocabulary;
  * every token also carries the action taken at its step, since a frame's successor depends
    on it -- SMT models unconditioned sequences and has no actions.

The memory is RSSM-shaped (deter 2048 + 32x16 categorical logits) so the one-step predictor
can be the native RSSM cell. K * d_model is chosen to equal deter + stoch*discrete, which is
SMT's own `d_model * mem_tokens == mem_size` constraint (model.py:113).
"""
import torch
from torch import nn
from x_transformers import Decoder, Encoder

from .blocks import normalize_mem, state_feat


class TeacherEncoder(nn.Module):
    """A window of frame embeddings + actions -> one Markovian state."""

    def __init__(self, embed_dim=1024, act_dim=2, d_model=256, depth=8, n_heads=8,
                 n_registers=10, deter=2048, stoch=32, discrete=16,
                 rotary_pos_emb=True, use_rmsnorm=True, heads=True):
        super().__init__()
        self.n_registers, self.d_model = int(n_registers), int(d_model)
        self.stoch, self.discrete = int(stoch), int(discrete)
        self.heads = bool(heads)
        flat = self.n_registers * self.d_model

        self.embed_in = nn.Linear(embed_dim, d_model)
        self.act_in = nn.Linear(act_dim, d_model, bias=False)
        self.encoder_mem_register = nn.Embedding(self.n_registers, d_model)  # SMT model.py:121
        self.encoder = Encoder(dim=d_model, depth=depth, heads=n_heads,
                               rotary_pos_emb=rotary_pos_emb, use_rmsnorm=use_rmsnorm)
        # heads=False (smt_phase1): the memory IS the state, there is no RSSM to regress onto,
        # so the two projections would be 6.55M parameters that receive no gradient. The
        # K*d == deter + stoch*discrete constraint is likewise an RSSM-shape constraint only.
        if self.heads:
            if flat != int(deter) + self.stoch * self.discrete:
                raise ValueError(f"{n_registers} registers x {d_model} = {flat} != "
                                 f"deter {deter} + stoch {stoch}x{discrete}")
            self.to_deter = nn.Linear(flat, int(deter))
            self.to_logit = nn.Linear(flat, self.stoch * self.discrete)

    def tokens(self, e_ctx, a_ctx, ctx_mask):
        """The K memory tokens, before any RSSM head. (B, C, E) -> (B, K, d).

        This is SMT's memory itself: `z` in Teacher.encode (model.py:142-144). It never touches
        to_deter / to_logit, so it is also the whole encoder when heads=False.
        """
        batch = e_ctx.shape[0]
        tokens = self.embed_in(e_ctx) + self.act_in(a_ctx)
        registers = self.encoder_mem_register(torch.arange(self.n_registers, device=e_ctx.device))
        registers = registers.unsqueeze(0).expand(batch, -1, -1)

        # register keys are never masked, so no query row is ever fully masked (SMT model.py:135)
        mask = torch.cat([ctx_mask, torch.ones(batch, self.n_registers, dtype=torch.bool,
                                               device=e_ctx.device)], dim=1)
        out = self.encoder(torch.cat([tokens, registers], dim=1), mask=mask)
        return normalize_mem(out[:, -self.n_registers:, :])  # SMT model.py:142-144

    def forward(self, e_ctx, a_ctx, ctx_mask):
        """e_ctx (B, C, E), a_ctx (B, C, A), ctx_mask (B, C) bool True=real -> deter, logit, z."""
        if not self.heads:
            raise RuntimeError("TeacherEncoder was built with heads=False; call .tokens(...)")
        z = self.tokens(e_ctx, a_ctx, ctx_mask)
        flat = z.reshape(z.shape[0], -1)
        # tanh: the RSSM's Deter cell is a convex mix of tanh candidates, so its deter lives in
        # (-1, 1) (rssm.py:71-75). An unbounded target here would be structurally unreachable.
        deter = torch.tanh(self.to_deter(flat))
        logit = self.to_logit(flat).view(z.shape[0], self.stoch, self.discrete)
        return deter, logit, z


class TeacherDecoder(nn.Module):
    """The predictive-state objective: from the state and the coming actions, predict the future.

    Token order, K memory tokens then F future tokens, run through a causal Decoder:

        [ z_0 .. z_{K-1} , u_0 , .. , u_{F-1} ]      u_j -> predicts e_{t+j+1}
        u_j = embed(e_{t+j}) + act(a_{t+j})

    Shifted one token from SMT, whose last memory token predicts the first future token
    (model.py:157, slice [K-1:-1]). The shift is where the action goes: predicting e_{t+1}
    requires knowing a_t, and a memory token has nowhere to put it.

    u_j deliberately carries e_{t+j}, which the decoder may copy: 61% of transitions in this
    data are bit-identical, and letting the decoder take that path is what keeps the memory
    from spending capacity on frame appearance. At a lift, copying is wrong and only the state
    can supply the answer -- that is where the memory pressure comes from.

    Padded future steps are handled in the loss, as SMT handles them (y == -1, model.py:61-67),
    not by an attention mask: attention here is causal, so a padded tail cannot reach a real
    prediction anyway.
    """

    def __init__(self, embed_dim=1024, act_dim=2, d_model=256, depth=4, n_heads=8,
                 n_registers=10, feat_dim=2560, rotary_pos_emb=True, use_rmsnorm=True,
                 twin_ball_decoder=False):
        super().__init__()
        self.n_registers, self.d_model = int(n_registers), int(d_model)
        self.from_state = nn.Linear(int(feat_dim), self.n_registers * d_model)
        self.embed_in = nn.Linear(embed_dim, d_model)
        self.act_in = nn.Linear(act_dim, d_model, bias=False)
        self.decoder = Decoder(dim=d_model, depth=depth, heads=n_heads,
                               rotary_pos_emb=rotary_pos_emb, use_rmsnorm=use_rmsnorm)
        self.to_pred = nn.Linear(d_model, embed_dim)
        # Auxiliary head: "will the ball be visible at this future step". One logit per position,
        # supervised only during the final lift once the cup has risen (show offsets 1-9).
        self.to_ball = nn.Linear(d_model, 1)
        # If one shared trunk degrades latent prediction, give the ball head its own decoder of
        # identical architecture rather than trading the two off inside one set of weights.
        self.ball_decoder = (Decoder(dim=d_model, depth=depth, heads=n_heads,
                                     rotary_pos_emb=rotary_pos_emb, use_rmsnorm=use_rmsnorm)
                             if twin_ball_decoder else None)

    def _run_tokens(self, z, e_fut, a_fut, decoder=None):
        """Shared trunk: [memory ; future tokens] -> per-position hidden states (B, F, d)."""
        tokens = self.embed_in(e_fut) + self.act_in(a_fut)
        out = (decoder or self.decoder)(torch.cat([z, tokens], dim=1))
        return out[:, self.n_registers:, :]

    def forward_tokens(self, z, e_fut, a_fut):
        """Decode from the memory tokens directly: z (B, K, d). The SMT-native path."""
        return self.to_pred(self._run_tokens(z, e_fut, a_fut))

    def predict_tokens(self, z, e_fut, a_fut):
        """Both heads from one decoder pass: (latents (B,F,E), ball logits (B,F))."""
        hidden = self._run_tokens(z, e_fut, a_fut)
        ball_hidden = (self._run_tokens(z, e_fut, a_fut, self.ball_decoder)
                       if self.ball_decoder is not None else hidden)
        return self.to_pred(hidden), self.to_ball(ball_hidden).squeeze(-1)

    def decode_step_tokens(self, z, embed, action):
        return self.forward_tokens(z, embed.unsqueeze(1), action.unsqueeze(1)).squeeze(1)

    @torch.no_grad()
    def decode_tokens_without_memory(self, z, e_fut, a_fut):
        """Diagnostic only: the same forward with the memory zeroed."""
        return self.forward_tokens(torch.zeros_like(z), e_fut, a_fut)

    def forward(self, deter, logit, e_fut, a_fut):
        """deter (B, D), logit (B, S, K), e_fut (B, F, E), a_fut (B, F, A) -> (B, F, E)."""
        batch, fut_len = e_fut.shape[0], e_fut.shape[1]
        memory = self.from_state(state_feat(deter, logit)).view(batch, self.n_registers, self.d_model)
        tokens = self.embed_in(e_fut) + self.act_in(a_fut)
        out = self.decoder(torch.cat([memory, tokens], dim=1))
        return self.to_pred(out[:, self.n_registers:, :])

    def decode_step(self, deter, logit, embed, action):
        """One step from the current state: (B, E), (B, A) -> (B, E). The open-loop path."""
        return self(deter, logit, embed.unsqueeze(1), action.unsqueeze(1)).squeeze(1)

    @torch.no_grad()
    def decode_without_memory(self, deter, logit, e_fut, a_fut):
        """Diagnostic only, no gradient: the same forward with the memory zeroed.

        Says whether the state is read at all. Nothing in SMT; it changes no weights.
        """
        return self(torch.zeros_like(deter), torch.zeros_like(logit), e_fut, a_fut)
