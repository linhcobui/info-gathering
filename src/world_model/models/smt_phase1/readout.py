"""StateReadout: the memory -> one frame embedding. STAGE 2 ONLY.

Trained on frozen stage-1 states against the true embedding (MSE), so it never shapes the
representation -- exactly the contract the pixel decoder already has. Its output feeds
smt_prelim's EmbedPixelDecoder, which reads only latents["embed"], so the yellow-pixel
detector and scripts/eval_memtest1.py work unchanged and the headline number stays
pixel-based and comparable to every earlier run.

It exists because phase 1 has no embedding predictor: the reward decoder is training-only and
the memory is a plain vector, so something has to turn an imagined state back into a frame.
"""
from torch import nn
from x_transformers import Encoder


class StateReadout(nn.Module):
    """z (..., K, d) -> embedding (..., E). Bidirectional: the K tokens are a set, not a sequence."""

    def __init__(self, embed_dim=1024, d_model=256, depth=4, n_heads=8, n_registers=10,
                 rotary_pos_emb=True, use_rmsnorm=True):
        super().__init__()
        self.n_registers, self.d_model = int(n_registers), int(d_model)
        self.encoder = Encoder(dim=d_model, depth=depth, heads=n_heads,
                               rotary_pos_emb=rotary_pos_emb, use_rmsnorm=use_rmsnorm)
        # flatten-then-project: K*d (2560) -> E (1024) loses nothing a pooled token would keep
        self.to_embed = nn.Linear(self.n_registers * self.d_model, int(embed_dim))

    def forward(self, z):
        lead = z.shape[:-2]
        out = self.encoder(z.reshape(-1, self.n_registers, self.d_model))
        return self.to_embed(out.reshape(*lead, self.n_registers * self.d_model))
