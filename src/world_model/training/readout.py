"""Fit a backend's decoder on frozen world-model latents."""
import torch
from torch.amp import GradScaler, autocast


class DecoderReadout:
    """Trains model.build_decoder() on latents from a frozen model.

    The model's parameters are frozen and its latents come from observe() under
    no_grad, so the decoder can never shape the representation. The optimiser is the
    same for every backend, so decoder quality is comparable across ablations.
    """

    def __init__(self, model, lr):
        self.model = model.requires_grad_(False).eval()
        self.device = model.device
        self.decoder = model.build_decoder()
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=float(lr), weight_decay=0.0)
        self.scaler = GradScaler(self.device.type)

    def step(self, batch):
        latents = self.model.observe(batch)
        with autocast(self.device.type, dtype=torch.float16):
            loss = self.decoder.loss(latents, batch["image"])
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return {"loss/decoder": loss.detach(), "opt/grad_scale": self.scaler.get_scale()}

    def state_dict(self):
        return {
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state):
        self.decoder.load_state_dict(state["decoder"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
