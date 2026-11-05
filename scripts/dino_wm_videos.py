"""Videos of a DINO-WM checkpoint on OGBar episodes, decoded to pixels by its token decoder.

Panels, left to right (one video frame = one model step = 5 raw steps):
    real            the recorded 224x224 side-view frame
    recon           decoder(real DINOv2 tokens): what the tokens hold
    1-step          decoder(prediction from the 3 real frames before it)
    open-loop H     decoder(rollout from real context, restarted every H steps; the real actions drive it)

Also prints the latent MSE of the open-loop rollout per horizon step.

    pixi run python scripts/dino_wm_videos.py --ckpt logdir/world_model/dino_wm_ogbar/v3_400_proprio/checkpoints/wm_latest.pt --episodes 0 1 2
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from info_gathering.ogbar_dino import OgbarDinoWindows  # noqa: E402
from train_dino_wm import build_model, to_u8  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402


@torch.no_grad()
def decode_all(model, tokens, bs=32):
    """(T, 256, 384) -> (T, 224, 224, 3) uint8."""
    out = []
    for i in range(0, len(tokens), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(to_u8(model.decoder(tokens[i:i + bs].float())))
    return np.concatenate(out)


@torch.no_grad()
def one_step(model, seq, device, bs=32):
    """Predicted tokens of frame t from real frames t-3..t-1, for t = 3..T-1."""
    h = model.num_hist
    z = model.encode(seq["visual"][None].to(device), seq["proprio"][None].to(device), seq["action"][None].to(device))[0]
    T = z.shape[0]
    out = []
    for s in range(h, T, bs):
        win = torch.stack([z[t - h:t] for t in range(s, min(s + bs, T))])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model.predict(win)[:, -1, :, :model.emb_dim].float())
    return torch.cat(out)


@torch.no_grad()
def open_loop(model, seq, device, horizon):
    """Rollout restarted every `horizon` steps from 3 real frames; returns tokens for t = 3..T-1 and step-in-chunk."""
    h = model.num_hist
    vis, prop, act = (seq[k].to(device) for k in ("visual", "proprio", "action"))
    T = vis.shape[0]
    toks, k_in = [], []
    for s in range(h, T, horizon):
        H = min(horizon, T - s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v, _ = model.rollout(vis[None, s - h:s], prop[None, s - h:s], act[None, s - h:s + H - 1])
        toks.append(v[0, h:h + H].float())
        k_in += list(range(1, H + 1))
    return torch.cat(toks), np.array(k_in)


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((4, 3), text, fill=(255, 255, 0))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--horizon", type=int, default=20, help="open-loop steps before restarting from real frames")
    ap.add_argument("--fps", type=int, default=8, help="model steps per second of video (real time is 4)")
    ap.add_argument("--out", default=None, help="default: <run>/videos/")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--zarr", default=None, help="other store (e.g. data/ogbar_v3_eval5.zarr); default: the training one")
    ap.add_argument("--cache-dir", default=None, help="its DINOv2 cache; default: <zarr stem>_dinov2")
    ap.add_argument("--two-panel", action="store_true", help="only real | open-loop (use --horizon 100000: from the start)")
    args = ap.parse_args()

    state = CheckpointManager.load(args.ckpt)
    cfg = OmegaConf.create(state["config"])
    device = torch.device(args.device)
    zarr_path = args.zarr or cfg.data.zarr_path
    cache_dir = args.cache_dir or (cfg.data.cache_dir if not args.zarr else args.zarr.removesuffix(".zarr") + "_dinov2")
    data = OgbarDinoWindows(ROOT / cache_dir, ROOT / zarr_path, num_frames=cfg.model.num_hist + 1,
                            frameskip=cfg.data.frameskip)
    model = build_model(cfg, state["stats"], data.action_dim, data.proprio_dim).to(device).eval()
    model.load_state_dict(state["model"])
    out = Path(args.out or Path(args.ckpt).parents[1] / "videos")
    out.mkdir(parents=True, exist_ok=True)
    h = model.num_hist
    per_k = {}
    for e in args.episodes:
        seq = data.episode_sequence(e)
        real = seq["rgb"].numpy()
        recon = decode_all(model, seq["visual"].to(device))
        one = decode_all(model, one_step(model, seq, device))
        ol_tok, k_in = open_loop(model, seq, device, args.horizon)
        ol = decode_all(model, ol_tok)
        err = ((ol_tok - seq["visual"][h:].to(device).float()) ** 2).mean((1, 2)).cpu().numpy()
        for k, v in zip(k_in, err):
            per_k.setdefault(k, []).append(v)
        frames = []
        for t in range(len(real)):
            blank = np.zeros_like(real[t])
            p1 = one[t - h] if t >= h else blank
            p2 = ol[t - h] if t >= h else blank
            k = f" (+{k_in[t - h]})" if t >= h else ""
            if args.two_panel:
                p2 = real[t] if t < h else p2                          # the 3 context frames are real
                frames.append(np.concatenate([label(real[t], f"ground truth  ep {e}  step {t}"),
                                              label(p2, "real context" if t < h else f"open-loop from start (+{k_in[t - h]})")],
                                             axis=1))
            else:
                frames.append(np.concatenate([label(real[t], f"real  ep {e}  step {t}"), label(recon[t], "recon (real tokens)"),
                                              label(p1, "1-step prediction"),
                                              label(p2, f"open-loop H={args.horizon}{k}")], axis=1))
        tag = f"{Path(zarr_path).stem}_" if args.zarr else ""
        path = out / f"{tag}ep{e:03d}_{'openloop_' if args.two_panel else ''}{Path(args.ckpt).stem}.mp4"
        imageio.mimwrite(path, frames, fps=args.fps, macro_block_size=8)
        print(f"{path}  ({len(frames)} steps)", flush=True)
    ks = sorted(per_k)
    print("open-loop latent MSE by step into the rollout: " +
          " ".join(f"+{k}:{np.mean(per_k[k]):.3f}" for k in ks if k in (1, 2, 5, 10, 15, 20) or k == ks[-1]))


if __name__ == "__main__":
    main()
