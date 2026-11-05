"""Open-loop (autoregressive) DINO-WM errors over 1..h_max steps, for the error-spike segmentation.

From the real context ending at t (frames t-2 .. t), the frozen WM is rolled forward with the logged actions
a_t .. a_{t+h-1}, feeding back its own predictions; frame t+h's prediction is scored against the real e_{t+h}:
    err_l2_mean / err_l2_topk   per-patch L2 over DINO patch tokens (visual part), mean over all 256 / the k worst
    copy_l2_mean                copy-last baseline ||e_t - e_{t+h}||, mean over patches (how much the scene changed)
One row per (episode_id, t, h), h = 1 .. h_max (fewer near the episode end). h = 1 is the one-step error of
compute_wm_errors.py.

Writes <out_dir>/wm_rollout_errors.parquet (+ .meta.json).
    pixi run python scripts/compute_rollout_errors.py --h-max 4 [--stores heldout200] [key=value ...]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from seg.config import load_config, out_dir, write_meta  # noqa: E402
from seg.episodes import load_store  # noqa: E402
from seg.metrics import to_device  # noqa: E402
from seg.wm import DinoWMAdapter  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402


@torch.no_grad()
def episode_rollout_errors(ep, wm, h_max, batch, topk):
    H, dev = wm.history_len, wm.device
    vis, prop, act = to_device(ep, dev)
    rows = []
    ts = np.arange(H - 1, ep.T - 1)
    n_of = np.minimum(h_max, ep.T - 1 - ts)                     # horizon available from each t
    for n in np.unique(n_of):
        tn = ts[n_of == n]
        for b in range(0, len(tn), batch):
            t = torch.as_tensor(tn[b:b + batch], device=dev)
            ctx = t[:, None] + torch.arange(-H + 1, 1, device=dev)
            aidx = t[:, None] + torch.arange(-H + 1, int(n), device=dev)
            pred = wm.rollout(vis[ctx], prop[ctx], act[aidx])                     # (b, n, P, D): frames t+1 .. t+n
            real = vis[t[:, None] + torch.arange(1, int(n) + 1, device=dev)].float()
            d = (pred - real).norm(dim=-1)                                         # (b, n, P)
            dc = (vis[t].float()[:, None] - real).norm(dim=-1)
            k = min(topk, d.shape[-1])
            for h in range(int(n)):
                rows.append(pd.DataFrame(dict(t=tn[b:b + batch], h=h + 1,
                                              err_l2_mean=d[:, h].mean(-1).cpu().numpy(),
                                              err_l2_topk=d[:, h].topk(k, dim=-1).values.mean(-1).cpu().numpy(),
                                              copy_l2_mean=dc[:, h].mean(-1).cpu().numpy())))
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "episode_id", ep.episode_id)
    df["episode"], df["store"], df["wm_train_episode"] = ep.index, ep.store, ep.wm_train
    df["target_raw_frame"] = ep.frame_index[df.t + df.h]
    return df.sort_values(["t", "h"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--h-max", type=int, default=4)
    ap.add_argument("--stores", nargs="*", default=None, help="store names (default: all in the config)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    wm = DinoWMAdapter(ROOT / cfg.paths.wm_ckpt, cfg.device, cfg.errors.bf16)
    stores = [s for s in cfg.stores if args.stores is None or s.name in args.stores]
    t0, parts = time.time(), []
    for s in stores:
        eps = load_store(s, cfg.frameskip)
        for i, ep in enumerate(eps):
            parts.append(episode_rollout_errors(ep, wm, args.h_max, cfg.errors.batch_size, cfg.errors.topk))
            if (i + 1) % 50 == 0:
                print(f"  {s.name}: {i + 1}/{len(eps)} episodes  {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.concat(parts, ignore_index=True)
    path = out_dir(cfg) / ("wm_rollout_errors.parquet" if args.stores is None else
                           f"wm_rollout_errors_{'_'.join(args.stores)}.parquet")
    df.to_parquet(path)
    write_meta(path, cfg, h_max=args.h_max, stores=[s.name for s in stores])
    print(df.groupby(["wm_train_episode", "h"])[["err_l2_mean", "err_l2_topk", "copy_l2_mean"]].mean().round(3))
    print(f"wrote {path} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
