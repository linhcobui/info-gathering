"""Named checkpoints (<prefix>_latest.pt, <prefix>_best.pt), each overwritten atomically."""
import os
from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, directory, prefix):
        self.directory = Path(directory)
        self.prefix = prefix

    def path(self, name):
        return self.directory / f"{self.prefix}_{name}.pt"

    def save(self, name, state):
        """Write <prefix>_<name>.pt, replacing the previous one."""
        path = self.path(name)
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)  # a crash mid-save never leaves a truncated checkpoint
        return path

    def latest(self):
        path = self.path("latest")
        return path if path.exists() else None

    @staticmethod
    def load(path, map_location="cpu"):
        return torch.load(path, map_location=map_location, weights_only=False)


def load_frozen_models(decoder_checkpoint, device):
    """Frozen world model + decoder from a train_decoder.py checkpoint, with the datasets they were trained on.

    Returns (model, decoder, train_set, holdout_set, decoder_state, world_model_state).
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    dec_state = CheckpointManager.load(decoder_checkpoint)
    wm_state = CheckpointManager.load(dec_state["wm_checkpoint"])
    wm_cfg = OmegaConf.create(wm_state["config"])
    train_set, holdout_set = instantiate(wm_cfg.data)
    model = instantiate(wm_cfg.model, action_dim=train_set.action_dim, device=device, _recursive_=False)
    model.load_train_state_dict(wm_state["model"])
    model.requires_grad_(False).eval()
    decoder = model.build_decoder()
    decoder.load_state_dict(dec_state["readout"]["decoder"])
    decoder.eval()
    return model, decoder, train_set, holdout_set, dec_state, wm_state
