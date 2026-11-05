"""Pixel decoder for R2-Dreamer latents: r2dreamer's MultiDecoder over (stoch, deter)."""
import copy

import torch
from torch.amp import autocast

from world_model.models.base import Decoder

from .bridge import networks, to_f32


class R2DreamerDecoder(Decoder):
    def __init__(self, config, rssm, image_size):
        super().__init__()
        shapes = {"image": (image_size, image_size, 3)}
        self.net = networks.MultiDecoder(copy.deepcopy(config), rssm._deter, rssm.flat_stoch, shapes)

    def _dist(self, latents):
        return self.net(latents["stoch"].float(), latents["deter"].float())["image"]

    def loss(self, latents, image):
        return torch.mean(-self._dist(latents).log_prob(to_f32(image) / 255.0))

    @torch.no_grad()
    def render(self, latents):
        device_type = next(self.parameters()).device.type
        with autocast(device_type, dtype=torch.float16):
            image = self._dist(latents).mode()
        return image.float().clamp(0.0, 1.0)
