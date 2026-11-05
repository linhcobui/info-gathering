"""Pixel decoder for cached frame embeddings: r2dreamer's MultiDecoder over `embed`.

A plain autoencoder head on a frozen encoder's output -- it never shapes the
representation, and because it reads only latents["embed"] it renders real and imagined
embeddings through exactly the same path.

The architecture is r2dreamer's own ConvDecoder, fed the embedding on both of its input
branches (deterministic and stochastic). That keeps capacity and loss scale identical to
the Dreamer's stage-2 decoder, so the `recon` F1 it produces is directly comparable to the
baseline's rather than confounded by a different renderer.
"""
import copy

import torch
from torch.amp import autocast

from world_model.models.base import Decoder

from .bridge import networks, to_f32


class EmbedPixelDecoder(Decoder):
    def __init__(self, config, embed_dim, image_size):
        super().__init__()
        shapes = {"image": (image_size, image_size, 3)}
        # deter and flat_stoch both = embed_dim: BlockLinear needs embed_dim % bspace == 0
        self.net = networks.MultiDecoder(copy.deepcopy(config), embed_dim, embed_dim, shapes)

    def _dist(self, latents):
        embed = latents["embed"].float()
        return self.net(embed, embed)["image"]

    def loss(self, latents, image):
        return torch.mean(-self._dist(latents).log_prob(to_f32(image) / 255.0))

    @torch.no_grad()
    def render(self, latents):
        device_type = next(self.parameters()).device.type
        with autocast(device_type, dtype=torch.float16):
            image = self._dist(latents).mode()
        return image.float().clamp(0.0, 1.0)
