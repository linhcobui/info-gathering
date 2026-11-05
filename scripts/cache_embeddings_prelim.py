"""PRELIMINARY stage 1: cache a trained Dreamer encoder's embeddings into the memtest zarrs.

Writes one new (n_frames, 1024) float32 array per target store and verifies it before
returning. Nothing existing is modified.

    pixi run cache-embeddings-prelim
    pixi run cache-embeddings-prelim overwrite=true
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hashlib  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from info_gathering.ball_detector import ball_pixels  # noqa: E402
from world_model.models.smt_prelim.embed_cache import (  # noqa: E402
    build_encoder, cache_embeddings, encode_frames,
)
from world_model.utils.seeding import seed_everything  # noqa: E402


def store_fingerprint(zarr_path):
    """What must not change: the set of arrays, and the frames/labels themselves."""
    z = zarr.open_group(str(zarr_path), mode="r")
    digest = hashlib.sha256()
    for key in ("rgb", "action", "episode_ends"):
        digest.update(np.ascontiguousarray(z[key][:]).tobytes())
    return sorted(z.array_keys()), digest.hexdigest()


def verify(zarr_path, embed_key, checkpoint, native, device, cfg, rng):
    """Gate A. Raises on anything that would make the cache untrustworthy."""
    z = zarr.open_group(str(zarr_path), mode="r")
    embed = z[embed_key]
    n_frames = z["rgb"].shape[0]
    checks = []

    if embed.shape != (n_frames, 1024) or embed.dtype != np.float32:
        raise AssertionError(f"{embed_key} is {embed.shape} {embed.dtype}, expected ({n_frames}, 1024) float32")
    checks.append(f"shape {embed.shape} float32")

    sample = embed[: min(4096, n_frames)]
    if not np.isfinite(sample).all():
        raise AssertionError(f"{embed_key} contains non-finite values")
    checks.append("finite")

    # a SECOND instance of the SAME TRAINED encoder, reloaded -- this tests the caching
    # path (preprocessing, dtype, no autocast drift), not the weights
    encoder, _ = build_encoder(checkpoint, native, device, int(cfg.image_size))
    idx = np.sort(rng.choice(n_frames, size=min(int(cfg.verify.reencode_samples), n_frames), replace=False))
    recomputed = encode_frames(encoder, z["rgb"][:][idx], device)
    delta = float(np.abs(recomputed - embed[:][idx]).max())
    if delta != 0.0:
        raise AssertionError(f"re-encoding {len(idx)} frames differs by {delta:.3e}, expected exactly 0")
    checks.append(f"re-encode exact on {len(idx)} frames")

    # byte-identical frames must give byte-identical embeddings (the trailing static
    # "show" frames are the natural test case)
    rgb = z["rgb"][: min(4096, n_frames)]
    same = np.flatnonzero((rgb[1:] == rgb[:-1]).all(axis=(1, 2, 3)))
    if len(same):
        pairs = same[: 256]
        drift = float(np.abs(sample[pairs + 1] - sample[pairs]).max())
        if drift != 0.0:
            raise AssertionError(f"identical frames embed differently by {drift:.3e}")
        checks.append(f"{len(pairs)} identical frame pairs agree exactly")

    return checks


def probe_visible(zarr_path, embed_key, threshold=0.15, min_pixels=3):
    """Can a linear read-out tell whether the ball is on screen in THIS frame?

    A data-quality check on the frozen encoder, not an eval metric: if this fails, the
    embedding is blind and nothing downstream can work. The target is the yellow-pixel
    detector's own verdict on the same frame, so unlike the episode-level
    ball_under_target label it is always answerable from the pixels -- during the first
    frames of the "show" phase the cup is still rising, so the ball is absent from the
    image while the episode label already says 1. Closed-form ridge, no sklearn.
    """
    z = zarr.open_group(str(zarr_path), mode="r")
    rgb = torch.from_numpy(z["rgb"][:].astype(np.float32) / 255.0)
    y = (ball_pixels(rgb, threshold).numpy() >= min_pixels).astype(np.float64) * 2.0 - 1.0
    x = z[embed_key][:].astype(np.float64)
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    order = np.random.default_rng(0).permutation(len(x))
    tr, te = order[: len(x) // 2], order[len(x) // 2:]
    gram = x[tr].T @ x[tr] + 1e-2 * np.eye(x.shape[1])
    w = np.linalg.solve(gram, x[tr].T @ y[tr])
    return float((np.sign(x[te] @ w) == y[te]).mean()), float((y > 0).mean())


@hydra.main(version_base=None, config_path="../configs/smt_prelim", config_name="cache_embeddings")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    # deliberately NOT set_float32_matmul_precision("high"): encode_frames forces exact fp32
    rng = np.random.default_rng(cfg.seed)
    checkpoint = str(ROOT / cfg.checkpoint) if not Path(cfg.checkpoint).is_absolute() else str(cfg.checkpoint)
    native = OmegaConf.to_container(cfg.native, resolve=True)

    for target in cfg.targets:
        path = ROOT / target
        before_keys, before_digest = store_fingerprint(path)
        attrs = cache_embeddings(path, checkpoint, str(cfg.embed_key), native, cfg.device,
                                 image_size=int(cfg.image_size), block=int(cfg.block),
                                 overwrite=bool(cfg.overwrite), expect_step=cfg.get("expect_step"))
        after_keys, after_digest = store_fingerprint(path)
        if after_digest != before_digest:
            raise AssertionError(f"{target}: existing data changed while writing the cache")
        if set(after_keys) - set(before_keys) - {str(cfg.embed_key)}:
            raise AssertionError(f"{target}: unexpected new arrays {set(after_keys) - set(before_keys)}")

        checks = verify(path, str(cfg.embed_key), checkpoint, native, cfg.device, cfg, rng)
        print(f"\n{target} -> {cfg.embed_key}")
        print(f"  from {attrs['source_checkpoint']} (step {attrs['wm_step']})")
        print(f"  {attrs['n_frames']} frames x {attrs['embed_dim']} dims"
              f"   |mean| {np.abs(attrs['mean']).mean():.4f}  std {np.mean(attrs['std']):.4f}")
        for check in checks:
            print(f"  ok  {check}")
        print(f"  ok  arrays unchanged ({len(before_keys)} -> {len(after_keys)})")
        if cfg.verify.probe_visible_frames:
            accuracy, rate = probe_visible(path, str(cfg.embed_key))
            flag = "ok " if accuracy > 0.99 else "!! "
            print(f"  {flag} linear read-out of 'ball on screen in this frame': "
                  f"{accuracy:.4f}  (positives {rate:.3f})")


if __name__ == "__main__":
    main()
