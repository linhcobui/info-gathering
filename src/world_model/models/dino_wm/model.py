"""DINO-WM (Zhou et al. 2024) on cached DINOv2 patch tokens, with proprioception.

Ported from github.com/gaoyuezhou/dino_wm (models/visual_world_model.py, vit.py, proprio.py,
vqvae.py; MIT). Same architecture, losses and optimiser split; the differences are:
  - the frozen encoder is not run here: inputs are cached DINOv2 ViT-S/14 tokens of 224x224
    frames (16x16 = 256 patches; DINO-WM resizes to 196 px for 14x14);
  - the decoder upsamples 16x16 -> 256x256 and is resized to the 224x224 target;
  - attention uses torch SDPA with the same frame-causal mask (identical maths);
  - action and proprio are normalised with dataset mean/std held as buffers.

Token layout (concat_dim = 1): every patch token is [DINO 384 | proprio emb 10 | action emb 10],
the proprio and action embeddings tiled over the 256 patches of their frame. The predictor reads
num_hist frames and, at every position, predicts the next frame's tokens (teacher forcing).
Loss: MSE on the visual + proprio parts (action part excluded), target detached. The pixel
decoder trains on DETACHED real tokens only (it never shapes the predictor); decoding the
predicted tokens is logged, not trained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------- predictor (vit.py)

def frame_causal_mask(n_patches, n_frames):
    """(T, T) bool, True = may attend: a token sees every token of its own and earlier frames."""
    f = torch.arange(n_frames).repeat_interleave(n_patches)
    return f[:, None] >= f[None, :]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads, self.dropout = heads, dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x, mask):
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in self.to_qkv(self.norm(x)).chunk(3, -1))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class ViTPredictor(nn.Module):
    def __init__(self, num_patches, num_frames, dim, depth=6, heads=16, dim_head=64, mlp_dim=2048,
                 dropout=0.1, emb_dropout=0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, heads, dim_head, dropout),
                                                    FeedForward(dim, mlp_dim, dropout)]) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("mask", frame_causal_mask(num_patches, num_frames), persistent=False)

    def forward(self, x):
        n = x.shape[1]
        x = self.emb_dropout(x + self.pos_embedding[:, :n])
        mask = self.mask[:n, :n]
        for attn, ff in self.layers:
            x = attn(x, mask) + x
            x = ff(x) + x
        return self.norm(x)


# ---------------------------------------------------------------- pixel decoder (vqvae.py, quantize=False)

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(nn.ReLU(), nn.Conv2d(in_channel, channel, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, in_channel, 1))

    def forward(self, x):
        return x + self.conv(x)


def up4(in_channel, out_channel, channel, n_res_block, n_res_channel):
    """vqvae.Decoder with stride 4: conv, res blocks, two 2x transposed convs."""
    return nn.Sequential(nn.Conv2d(in_channel, channel, 3, padding=1),
                         *[ResBlock(channel, n_res_channel) for _ in range(n_res_block)], nn.ReLU(inplace=True),
                         nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                         nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1))


class TokenDecoder(nn.Module):
    """DINO-WM's VQVAE decoder without quantisation: (B, 256, 384) tokens -> (B, 3, out, out) in [-1, 1] scale."""

    def __init__(self, emb_dim=384, channel=384, n_res_block=4, n_res_channel=128, out_size=224):
        super().__init__()
        self.upsample_b = up4(emb_dim, emb_dim, channel, n_res_block, n_res_channel)
        self.dec = up4(emb_dim, 3, channel, n_res_block, n_res_channel)
        self.out_size = out_size

    def forward(self, tokens):
        side = int(tokens.shape[1] ** 0.5)
        x = rearrange(tokens, "b (h w) e -> b e h w", h=side, w=side)
        img = self.dec(self.upsample_b(x))                       # 16 -> 256
        if img.shape[-1] != self.out_size:
            img = F.interpolate(img, size=(self.out_size, self.out_size), mode="bilinear", antialias=True,
                                align_corners=False)
        return img


# ---------------------------------------------------------------- world model (visual_world_model.py)

class DinoWM(nn.Module):
    def __init__(self, action_dim, proprio_dim, stats, num_hist=3, num_pred=1, num_patches=256, emb_dim=384,
                 action_emb_dim=10, proprio_emb_dim=10, use_proprio=True, predictor=None, decoder=None,
                 has_decoder=True):
        super().__init__()
        assert num_pred == 1, "DINO-WM only supports num_pred = 1"
        self.num_hist, self.num_pred, self.emb_dim = num_hist, num_pred, emb_dim
        self.use_proprio = bool(use_proprio)
        self.proprio_emb_dim = proprio_emb_dim if self.use_proprio else 0
        self.action_emb_dim = action_emb_dim
        # ProprioceptiveEmbedding = Conv1d(kernel 1) = a linear map per frame
        self.action_encoder = nn.Linear(action_dim, action_emb_dim)
        self.proprio_encoder = nn.Linear(proprio_dim, proprio_emb_dim) if self.use_proprio else None
        for k in ("action_mean", "action_std", "proprio_mean", "proprio_std"):
            self.register_buffer(k, torch.as_tensor(stats[k], dtype=torch.float32))
        dim = emb_dim + self.proprio_emb_dim + action_emb_dim
        self.predictor = ViTPredictor(num_patches, num_hist, dim, **(predictor or {}))
        self.decoder = TokenDecoder(emb_dim, **(decoder or {})) if has_decoder else None

    # -- encoding -----------------------------------------------------------
    def encode_act(self, action):
        return self.action_encoder((action - self.action_mean) / self.action_std)

    def encode_proprio(self, proprio):
        return self.proprio_encoder((proprio - self.proprio_mean) / self.proprio_std)

    def encode(self, visual, proprio, action):
        """visual (b, t, p, 384), proprio (b, t, 19), action (b, t, 25) -> z (b, t, p, dim)."""
        p = visual.shape[2]
        parts = [visual.float()]
        if self.use_proprio:
            parts.append(repeat(self.encode_proprio(proprio), "b t a -> b t p a", p=p))
        parts.append(repeat(self.encode_act(action), "b t a -> b t p a", p=p))
        return torch.cat(parts, dim=3)

    def split(self, z):
        """z -> (visual tokens, proprio embedding (b, t, e) or None)."""
        e = self.emb_dim
        prop = z[:, :, 0, e:e + self.proprio_emb_dim] if self.use_proprio else None
        return z[..., :e], prop

    def predict(self, z):
        t = z.shape[1]
        return rearrange(self.predictor(rearrange(z, "b t p d -> b (t p) d")), "b (t p) d -> b t p d", t=t)

    def decode(self, visual_tokens):
        """(b, t, p, 384) -> (b, t, 3, 224, 224) in [-1, 1] scale."""
        b = visual_tokens.shape[0]
        img = self.decoder(rearrange(visual_tokens, "b t p e -> (b t) p e"))
        return rearrange(img, "(b t) c h w -> b t c h w", b=b)

    # -- training -------------------------------------------------------------
    def forward(self, batch, decode_pred=False):
        """batch: visual (b, H+1, p, 384), proprio, action, rgb (b, H+1, 224, 224, 3) uint8.

        Returns (loss, logs, extras). loss = z_loss + decoder reconstruction loss (if decoder).
        """
        z = self.encode(batch["visual"], batch["proprio"], batch["action"])
        z_src, z_tgt = z[:, :self.num_hist], z[:, self.num_pred:]
        z_pred = self.predict(z_src)
        a = self.action_emb_dim
        z_loss = F.mse_loss(z_pred[..., :-a], z_tgt[..., :-a].detach())
        e = self.emb_dim
        logs = dict(z_loss=z_loss.detach(),
                    z_visual_loss=F.mse_loss(z_pred[..., :e], z_tgt[..., :e]).detach())
        if self.use_proprio:
            logs["z_proprio_loss"] = F.mse_loss(z_pred[..., e:-a], z_tgt[..., e:-a]).detach()
        # error of predicting "no change" (copy the last context frame), for scale
        logs["z_visual_copy_last"] = F.mse_loss(z_src[:, -1, :, :e], z_tgt[:, -1, :, :e]).detach()
        logs["z_visual_loss_last"] = F.mse_loss(z_pred[:, -1, :, :e], z_tgt[:, -1, :, :e]).detach()
        loss = z_loss
        extras = dict(z_pred=z_pred.detach())
        if self.decoder is not None and "rgb" in batch:
            target = batch["rgb"].permute(0, 1, 4, 2, 3).float().div(127.5).sub(1.0)   # DINO-WM: Normalize(0.5, 0.5)
            recon = self.decode(batch["visual"].float())          # cached tokens carry no grad: decoder only
            rec_loss = F.mse_loss(recon.float(), target)
            loss = loss + rec_loss
            logs["decoder_recon_loss_reconstructed"] = rec_loss.detach()
            extras["recon"] = recon.detach()
            if decode_pred:
                with torch.no_grad():
                    pred_img = self.decode(z_pred[..., :e].detach())
                logs["decoder_recon_loss_pred"] = F.mse_loss(pred_img.float(), target[:, self.num_pred:])
                extras["pred_img"] = pred_img
        logs["loss"] = loss.detach()
        return loss, logs, extras

    # -- open-loop rollout (visual_world_model.rollout) -------------------------
    @torch.no_grad()
    def rollout(self, visual0, proprio0, actions):
        """visual0/proprio0: the first n real frames (b, n, ...); actions (b, n + H, 25), action[t] leaves frame t.

        Returns predicted visual tokens (b, n + H + 1, p, 384) (first n are the real ones) and proprio embeddings.
        """
        n = visual0.shape[1]
        z = self.encode(visual0, proprio0, actions[:, :n])
        for t in range(n, actions.shape[1]):
            z_new = self.predict(z[:, -self.num_hist:])[:, -1:]
            z_new[..., -self.action_emb_dim:] = repeat(self.encode_act(actions[:, t:t + 1]), "b t a -> b t p a",
                                                       p=z.shape[2])
            z = torch.cat([z, z_new], 1)
        z = torch.cat([z, self.predict(z[:, -self.num_hist:])[:, -1:]], 1)
        return self.split(z)

    def param_groups(self, predictor_lr=5e-4, action_encoder_lr=5e-4, decoder_lr=3e-4):
        """DINO-WM's optimisers: AdamW(predictor), AdamW(action + proprio encoders), Adam(decoder)."""
        enc = list(self.action_encoder.parameters()) + (list(self.proprio_encoder.parameters()) if self.use_proprio else [])
        opts = [torch.optim.AdamW(self.predictor.parameters(), lr=predictor_lr),
                torch.optim.AdamW(enc, lr=action_encoder_lr)]
        if self.decoder is not None:
            opts.append(torch.optim.Adam(self.decoder.parameters(), lr=decoder_lr))
        return opts
