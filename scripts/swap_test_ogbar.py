"""Swap test between two near-identical moments of two episodes: each belief, own actions vs the other's actions.

For episodes A (at model step tA) and B (at tB): z = the SMT teacher's belief on that episode's history up to its step,
x = that episode's real frame there. Rollouts with the prior (P chains z; frames from the decoder's first output, the
stand-in for R), for H steps:
  row A: ground truth A | prior(z_A, x_A, A's actions) | prior(z_A, x_A, B's actions)
  row B: ground truth B | prior(z_B, x_B, B's actions) | prior(z_B, x_B, A's actions)
Tilt = the bar's angle in the side-view image (line through its red pixels).

    pixi run python scripts/swap_test_ogbar.py --a 18 11 --b 272 56
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
import zarr  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from compare_from_cut import prior_rollout  # noqa: E402
from dino_wm_videos import decode_all  # noqa: E402
from dino_wm_watch_imagine import img_tilt, panel  # noqa: E402
from info_gathering.ogbar_smt_data import OgbarSmtAnchors  # noqa: E402
from train_dino_wm import build_model  # noqa: E402
from world_model.models.smt_ogbar import SmtOgbar  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

AMP = dict(device_type="cuda", dtype=torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smt", default="logdir/world_model/smt_ogbar/v3_400_gaussian/checkpoints/wm_latest.pt")
    ap.add_argument("--pixel", default="logdir/world_model/dino_wm_ogbar/v3_400_proprio/checkpoints/wm_latest.pt")
    ap.add_argument("--a", type=int, nargs=2, default=[18, 11], metavar=("EP", "STEP"))
    ap.add_argument("--b", type=int, nargs=2, default=[272, 56], metavar=("EP", "STEP"))
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)
    st = CheckpointManager.load(args.smt)
    m = SmtOgbar(action_dim=25, **st["config"]["model"]).to(dev).eval()
    m.load_state_dict(st["model"])
    pst = CheckpointManager.load(args.pixel)
    pix = build_model(OmegaConf.create(pst["config"]), pst["stats"], 25, 19).to(dev).eval()
    pix.load_state_dict(pst["model"])
    data = OgbarSmtAnchors(ROOT / "data/ogbar_v3_400_dinov2", ROOT / "data/ogbar_v3_400.zarr", 5)
    z = zarr.open_group(str(ROOT / "data/ogbar_v3_400.zarr"), "r")
    H = args.horizon
    S = {}
    for name, (e, t) in (("A", args.a), ("B", args.b)):
        p, a = data.load(e, 0)
        p, a = p.to(dev).to(torch.bfloat16), m.norm_action(a.to(dev))
        with torch.no_grad(), torch.autocast(**AMP):
            zt = m.teacher_windows(m.pool(p[None]), a[None], torch.full((1, 1), t, dtype=torch.long, device=dev))[0, 0]
        s0 = int(data.starts[e])
        real = z["rgb"][s0 + t * 5:s0 + (t + H + 1) * 5:5]
        S[name] = dict(e=e, t=t, p=p, a=a, z=zt, real=real, c=float(z["episode/com_offset_c"][e]) * 100)
    for name in ("A", "B"):
        X, Y = S[name], S["B" if name == "A" else "A"]
        h = min(H, len(X["a"]) - X["t"] - 1, len(Y["a"]) - Y["t"] - 1)
        X["self"] = decode_all(pix, prior_rollout(m, X["z"], X["p"][X["t"]], X["a"][X["t"]:X["t"] + h]).to(dev))
        X["cross"] = decode_all(pix, prior_rollout(m, X["z"], X["p"][X["t"]], Y["a"][Y["t"]:Y["t"] + h]).to(dev))
        X["h"] = h
    print(f"SMT step {st['step']}; A = ep {S['A']['e']} step {S['A']['t']} (c {S['A']['c']:+.1f}), "
          f"B = ep {S['B']['e']} step {S['B']['t']} (c {S['B']['c']:+.1f}); image tilt (deg), steps after the start")
    cols = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    tilt = lambda im: img_tilt(im[27:])
    for name in ("A", "B"):
        X, other = S[name], "B" if name == "A" else "A"
        cs = [k for k in cols if k <= X["h"]]
        print(f"  {'':34s}" + "".join(f"{'+' + str(k):>7s}" for k in cs))
        print(f"  {'GT ' + name:34s}" + "".join(f"{tilt(X['real'][k]):+7.1f}" for k in cs))
        print(f"  {'prior z_' + name + ', ' + name + chr(39) + 's actions (self)':34s}" + "".join(f"{(tilt(X['real'][0]) if k == 0 else tilt(X['self'][k - 1])):+7.1f}" for k in cs))
        print(f"  {'prior z_' + name + ', ' + other + chr(39) + 's actions (swap)':34s}" + "".join(f"{(tilt(X['real'][0]) if k == 0 else tilt(X['cross'][k - 1])):+7.1f}" for k in cs))
    h = min(S["A"]["h"], S["B"]["h"])
    frames = []
    for k in range(h + 1):
        rows = []
        for name in ("A", "B"):
            X, other = S[name], "B" if name == "A" else "A"
            lab = f"ep {X['e']} c {X['c']:+.1f}"
            rows.append(np.concatenate([
                panel(X["real"][k], f"GT {name}: {lab}", f"+{k}  "),
                panel(X["real"][0] if k == 0 else X["self"][k - 1], f"prior z_{name} + {name}'s actions", f"+{k}  "),
                panel(X["real"][0] if k == 0 else X["cross"][k - 1], f"prior z_{name} + {other}'s actions", f"+{k}  ")], 1))
        frames.append(np.concatenate(rows, 0))
    out = ROOT / "figures/ogbar_smt/swap"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"swap_ep{S['A']['e']}s{S['A']['t']}_ep{S['B']['e']}s{S['B']['t']}"
    imageio.mimwrite(out / f"{stem}.mp4", frames, fps=4, macro_block_size=8)
    pick = [k for k in (0, 8, 16, 24, 32, 40) if k <= h]
    imageio.imwrite(out / f"{stem}.png", np.concatenate([frames[k] for k in pick], 1))
    print(out / f"{stem}.mp4")
    print(out / f"{stem}.png")


if __name__ == "__main__":
    main()
