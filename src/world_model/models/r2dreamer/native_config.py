"""Resolve r2dreamer's own model config (configs/model/_base_.yaml + <size>.yaml) outside Hydra."""
from omegaconf import DictConfig, OmegaConf

from .bridge import R2DREAMER_ROOT

# _base_.yaml reads the observation keys from ${env.encoder.*} and ${env.decoder.*}
_IMAGE_ONLY = {"mlp_keys": "$^", "cnn_keys": "image"}


def load_native_config(size: str, overrides, device: str) -> DictConfig:
    model_dir = R2DREAMER_ROOT / "configs" / "model"
    base = OmegaConf.load(model_dir / "_base_.yaml")
    sized = OmegaConf.load(model_dir / f"{size}.yaml")
    sized.pop("defaults", None)
    root = OmegaConf.create({"device": device, "env": {"encoder": _IMAGE_ONLY, "decoder": _IMAGE_ONLY}})
    root.model = OmegaConf.merge(base, sized, overrides or {})
    OmegaConf.resolve(root)
    return root.model
