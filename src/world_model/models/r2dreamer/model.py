"""R2-Dreamer behind the generic WorldModel interface."""
import copy

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from torch.amp import autocast

from world_model.models.base import WorldModel

from .agent import BatchAsBuffer, OfflineDreamer
from .bridge import to_f32
from .decoder import R2DreamerDecoder
from .native_config import load_native_config


class R2DreamerWM(WorldModel):
    """Offline R2-Dreamer world model: encoder + RSSM + Barlow projector.

    Every window starts from the zero latent. With whole-episode windows (the default)
    each one begins at is_first, so no replay context is needed.
    """

    def __init__(self, action_dim, image_size, native, device, name="r2dreamer", window=None):
        super().__init__()
        self.name = name
        self.image_size = int(image_size)
        self.device = torch.device(device)
        self.native = load_native_config(native.size, native.overrides, device)
        obs_space = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (self.image_size, self.image_size, 3), np.uint8)})
        act_space = gym.spaces.Box(0.0, 1.0, (int(action_dim),), np.float32)
        # Dreamer.__init__ mutates its config, so it always gets a private copy
        self.agent = OfflineDreamer(copy.deepcopy(self.native), obs_space, act_space).to(self.device)

    def _data(self, batch):
        B, T = batch["is_first"].shape
        # r2dreamer's "action" is the action leading into each step
        data = TensorDict(
            {"image": batch["image"], "action": batch["prev_action"], "is_first": batch["is_first"]},
            batch_size=(B, T),
            device=self.device,
        )
        return data, self.agent.rssm.initial(B)

    def train_step(self, batch):
        data, initial = self._data(batch)
        return self.agent.update(BatchAsBuffer(data, initial))

    @torch.no_grad()
    def eval_step(self, batch):
        data, initial = self._data(batch)
        data = self.agent.preprocess(data)
        with autocast(self.device.type, dtype=torch.float16):
            losses, metrics, _ = self.agent.wm_losses(data, initial)
        metrics.update({f"loss/{k}": v for k, v in losses.items()})
        metrics["opt/loss"] = sum(v * self.agent._loss_scales[k] for k, v in losses.items())
        return metrics

    @torch.no_grad()
    def observe(self, batch):
        image = to_f32(batch["image"]) / 255.0
        B, T = image.shape[:2]
        with autocast(self.device.type, dtype=torch.float16):
            embed = self.agent.encoder({"image": image})
            stoch, deter, _ = self.agent.rssm.observe(
                embed, batch["prev_action"], self.agent.rssm.initial(B), batch["is_first"]
            )
        return TensorDict({"stoch": stoch, "deter": deter}, batch_size=(B, T))

    @torch.no_grad()
    def imagine(self, context, prev_action):
        with autocast(self.device.type, dtype=torch.float16):
            stoch, deter = self.agent.rssm.imagine_with_action(
                context["stoch"][:, -1], context["deter"][:, -1], prev_action
            )
        return TensorDict({"stoch": stoch, "deter": deter}, batch_size=prev_action.shape[:2])

    def build_decoder(self):
        return R2DreamerDecoder(self.native.decoder, self.agent.rssm, self.image_size).to(self.device)

    def train_state_dict(self):
        agent = self.agent
        return {
            "agent": agent.state_dict(),
            "optimizer": agent._optimizer.state_dict(),
            "scaler": agent._scaler.state_dict(),
            "scheduler": agent._scheduler.state_dict(),
        }

    def load_train_state_dict(self, state):
        agent = self.agent
        agent.load_state_dict(state["agent"])
        agent._optimizer.load_state_dict(state["optimizer"])
        agent._scaler.load_state_dict(state["scaler"])
        agent._scheduler.load_state_dict(state["scheduler"])
