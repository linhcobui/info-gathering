"""Cache a trained R2-Dreamer encoder's per-frame embeddings into an existing zarr.

The encoder is always loaded from a stage-1 world-model checkpoint, never built from a
random init: this backend's whole premise is that the frozen features already retain the
ball (memtest4 reconstructions scored F1 0.897-1.000) and only the prior was wrong.

The embeddings go into a NEW array inside the source store, alongside rgb/action/labels,
following the append idiom of info_gathering/memtest4.py:159-168. Nothing existing is
touched. z.attrs[embed_key] records which checkpoint produced them so a stale cache from
another run cannot be read back as if it belonged to this one.
"""
import contextlib
import datetime
from pathlib import Path

import numpy as np
import torch
import zarr

from ..r2dreamer.native_config import load_native_config
from .bridge import networks, to_f32

ATTRS_VERSION = 1
ENCODER_PREFIX = "encoder."


def build_encoder(checkpoint, native, device, image_size=64, expect_step=None):
    """The trained MultiEncoder from a stage-1 checkpoint, frozen and in eval mode.

    checkpoint: path to wm_latest.pt / wm_best.pt. native: {size, overrides} so the
    architecture is rebuilt with the very config the weights were trained under.
    """
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if state.get("kind") != "world_model":
        raise ValueError(f"{checkpoint} is a {state.get('kind')!r} checkpoint, not a world model")
    if expect_step is not None and int(state["step"]) != int(expect_step):
        raise ValueError(f"{checkpoint} is at step {state['step']}, expected {expect_step}")

    weights = {k[len(ENCODER_PREFIX):]: v for k, v in state["model"]["agent"].items()
               if k.startswith(ENCODER_PREFIX)}
    if not weights:
        raise ValueError(f"no {ENCODER_PREFIX}* weights in {checkpoint}")

    config = load_native_config(native["size"], native.get("overrides"), str(device))
    encoder = networks.MultiEncoder(config.encoder, {"image": (image_size, image_size, 3)})
    # strict: a silent partial load would leave randomly initialised filters in place
    encoder.load_state_dict(weights, strict=True)
    return encoder.to(device).eval().requires_grad_(False), state


@contextlib.contextmanager
def exact_fp32():
    """True fp32 convolutions, so the cache does not depend on the batch size it was written in.

    TF32 keeps ~10 mantissa bits and lets cuDNN pick a different algorithm per batch shape:
    measured on this encoder, the same frame embeds 1.7e-3 apart at block 512 vs block 64.
    With TF32 off the difference is exactly zero, which is what makes the cache verifiable.
    """
    cudnn, matmul = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = cudnn, matmul
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, deterministic


@torch.no_grad()
def encode_frames(encoder, rgb, device):
    """rgb (N, H, W, 3) uint8 -> (N, E) float32. No autocast, no TF32: the cache must be exact."""
    with exact_fp32():
        image = to_f32(torch.from_numpy(np.ascontiguousarray(rgb)).to(device)) / 255.0
        return encoder({"image": image}).float().cpu().numpy()


def cache_embeddings(zarr_path, checkpoint, embed_key, native, device,
                     image_size=64, block=512, overwrite=False, expect_step=None):
    """Encode every frame of <zarr_path>/rgb into a new (N, E) float32 array `embed_key`.

    Returns the provenance dict written to z.attrs[embed_key].
    """
    encoder, state = build_encoder(checkpoint, native, device, image_size, expect_step)

    z = zarr.open_group(str(zarr_path), mode="a")
    if int(z.attrs["resolution"]) != int(image_size):
        raise ValueError(f"{zarr_path} is {z.attrs['resolution']}px, encoder wants {image_size}px")
    if embed_key in z:
        if not overwrite:
            raise FileExistsError(f"{zarr_path} already has {embed_key!r}; pass overwrite=true to rebuild")
        del z[embed_key]

    n_frames = z["rgb"].shape[0]
    embed_dim = int(encoder.out_dim)
    out = z.create_dataset(embed_key, shape=(n_frames, embed_dim),
                           chunks=(512, embed_dim), dtype="float32")
    total = np.zeros(embed_dim, dtype=np.float64)
    total_sq = np.zeros(embed_dim, dtype=np.float64)
    for start in range(0, n_frames, block):
        stop = min(start + block, n_frames)
        embed = encode_frames(encoder, z["rgb"][start:stop], device)
        out[start:stop] = embed
        total += embed.astype(np.float64).sum(0)
        total_sq += (embed.astype(np.float64) ** 2).sum(0)

    mean = total / n_frames
    var = np.maximum(total_sq / n_frames - mean ** 2, 0.0)
    attrs = {
        "source_checkpoint": str(checkpoint),
        "wm_step": int(state["step"]),
        "encoder": "r2dreamer.MultiEncoder/ConvEncoder",
        "native_size": str(native["size"]),
        "embed_dim": embed_dim,
        "image_size": int(image_size),
        "n_frames": int(n_frames),
        "preprocess": "uint8/255",
        "dtype": "float32",
        "mean": mean.astype(np.float32).tolist(),
        "std": np.sqrt(var).astype(np.float32).tolist(),
        "version": ATTRS_VERSION,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    z.attrs[embed_key] = attrs
    return attrs


def read_embed_attrs(zarr_path, embed_key, checkpoint=None):
    """The provenance dict for a cached array, refusing anything stale or unlabelled."""
    z = zarr.open_group(str(zarr_path), mode="r")
    if embed_key not in z:
        raise FileNotFoundError(f"{zarr_path} has no {embed_key!r}; run cache-embeddings-prelim first")
    attrs = z.attrs.get(embed_key)
    if not attrs:
        raise ValueError(f"{zarr_path}[{embed_key}] has no provenance tag; rebuild it")
    if checkpoint is not None:
        # compare the files, not the spellings: the cache stores an absolute path while
        # configs carry repo-relative ones
        cached, wanted = Path(attrs["source_checkpoint"]).resolve(), Path(checkpoint).resolve()
        if cached != wanted:
            raise ValueError(
                f"{zarr_path}[{embed_key}] was built from {cached}, but this run expects {wanted}. "
                "Rebuild the cache or fix encoder_checkpoint.")
    return attrs
