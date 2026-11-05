"""Interface between the generic pipeline and a world-model backend."""
from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict
from torch import nn


class WorldModel(nn.Module, ABC):
    """A world model the pipeline can train, roll out and decode.

    Batches are dicts from info_gathering.dataset.EpisodeWindows, already on `device`:
        image (B, T, H, W, 3) uint8, action / prev_action (B, T, A), is_first (B, T),
        episode (B,), t0 (B,), meta/<key> (B, T, ...).
    Backends must never read meta/* -- those are labels for evaluation.
    """

    name: str
    device: torch.device

    @abstractmethod
    def train_step(self, batch: dict) -> dict:
        """One optimisation step; the backend owns its optimisers. Returns scalar metrics."""

    def eval_step(self, batch: dict) -> dict:
        """Training losses without an update, for holdout evaluation."""
        return {}

    @abstractmethod
    def observe(self, batch: dict) -> TensorDict:
        """Latents (B, T, ...) inferred from real frames (uses image, prev_action, is_first)."""

    @abstractmethod
    def imagine(self, context: TensorDict, prev_action: torch.Tensor) -> TensorDict:
        """Open-loop latents (B, H, ...) continuing `context`; prev_action[:, i] leads into imagined step i."""

    @abstractmethod
    def build_decoder(self) -> "Decoder":
        """A fresh, untrained decoder for this model's latents."""

    @abstractmethod
    def train_state_dict(self) -> dict:
        """Everything needed to resume training: weights and optimiser state."""

    @abstractmethod
    def load_train_state_dict(self, state: dict) -> None:
        """Inverse of train_state_dict()."""


class Decoder(nn.Module, ABC):
    """Maps latents to frames. Trained on frozen latents, so it never shapes the representation."""

    @abstractmethod
    def loss(self, latents: TensorDict, image: torch.Tensor) -> torch.Tensor:
        """Scalar loss against uint8 frames (B, T, H, W, 3)."""

    @abstractmethod
    def render(self, latents: TensorDict) -> torch.Tensor:
        """Frames (B, T, H, W, 3), float in [0, 1]."""
