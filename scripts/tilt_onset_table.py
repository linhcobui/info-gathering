"""Tilt at the start of a probe lift: ground truth vs the OGBar-SMT decoder's autoregressive rollout, step by step.

From a cut frame the SMT decoder rolls out on its own predicted DINOv2 tokens with the recorded actions (belief fixed at
the cut), for three beliefs: no memory (teacher on x_cut only), the posterior chain, the teacher on the whole history.
Tilt = the bar's angle in the side-view image (line through its red pixels), measured the same way on real and predicted
frames. Two cuts: the first model step at/after the lift starts ("old", the frame may already tip) and the last step
before it ("clean"). Writes a table to stdout and a frame strip per episode.

    pixi run python scripts/tilt_onset_table.py --episodes 156 198
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

from dino_wm_videos import decode_all  # noqa: E402
from dino_wm_watch_imagine import img_tilt, panel  # noqa: E402
from info_gathering.ogbar_smt_data import OgbarSmtAnchors  # noqa: E402
from smt_ogbar_watch_imagine import decoder_rollout  # noqa: E402
from train_dino_wm import build_model  # noqa: E402
from world_model.models.smt_ogbar import SmtOgbar  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

AMP = dict(device_type="cuda", dtype=torch.bfloat16)
STEPS = [0, 1, 2, 3, 4, 6, 8]          # model steps after the cut (1 model step = 5 raw steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smt", default="logdir/world_model/smt_ogbar/v3_400_unif0.1/checkpoints/wm_latest.pt")
    ap.add_argument("--dino-wm", default="logdir/world_model/dino_wm_ogbar/v3_400_proprio/checkpoints/wm_latest.pt")
    ap.add_argument("--zarr", default="data/ogbar_v3_400.zarr")
    ap.add_argument("--episodes", type=int, nargs="+", default=[156, 198])
    ap.add_argument("--probe", type=int, default=2)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)
    st = CheckpointManager.load(args.smt)
    m = SmtOgbar(action_dim=25, **st["config"]["model"]).to(dev).eval()
    m.load_state_dict(st["model"])
    pst = CheckpointManager.load(args.dino_wm)
    pix = build_model(OmegaConf.create(pst["config"]), pst["stats"], 25, 19).to(dev).eval()
    pix.load_state_dict(pst["model"])
    data = OgbarSmtAnchors(ROOT / (args.zarr.removesuffix(".zarr") + "_dinov2"), ROOT / args.zarr, 5)
    z = zarr.open_group(str(ROOT / args.zarr), "r")
    out_dir = Path(args.smt).parents[1] / "videos"
    tilt = lambda im: img_tilt(im[27:])
    print(f"SMT checkpoint step {st['step']}; image tilt in degrees; columns = model steps after the cut (x5 raw steps)")
    for e in args.episodes:
        s0, e0 = int(data.starts[e]), int(data.ends[e])
        ps = z["probe_stage"][s0:e0]
        lift = [t for t in range(1, len(ps)) if ps[t] == 1 and ps[t - 1] != 1][args.probe - 1]
        patch, action = data.load(e, 0)
        patch, action = patch.to(dev).to(torch.bfloat16), m.norm_action(action.to(dev))
        T = len(patch)
        real = z["rgb"][s0:e0][::5][:T]
        c_cm = float(z["episode/com_offset_c"][e]) * 100
        strips = []
        for name, cut in (("old cut", -(-lift // 5)), ("clean cut", (lift - 1) // 5)):
            with torch.no_grad(), torch.autocast(**AMP):
                pooled = m.pool(patch[None])
                W = m.cfg["window"]
                idx = torch.arange(cut - W + 1, cut + 1, device=dev)
                prev = torch.cat([torch.zeros_like(action[:1]), action[:-1]], 0)
                beliefs = {"no memory": m.teacher.tokens(pooled[:, idx.clamp(min=0)], prev[idx.clamp(min=0)][None].to(pooled.dtype),
                                                         (idx == cut)[None])[0],
                           "teacher": m.teacher_windows(pooled, action[None], torch.full((1, 1), cut, dtype=torch.long, device=dev))[0, 0]}
                zc = m.teacher_windows(pooled, action[None], torch.zeros(1, 1, dtype=torch.long, device=dev))[0, 0]
                for t in range(cut):
                    mu, ls = m.posterior(zc[None], patch[t][None], action[t][None], patch[t + 1][None])
                    zc = m.sample(mu, ls)[0]
                beliefs["posterior"] = zc
            H = max(STEPS)
            imgs = {k: decode_all(pix, decoder_rollout(m, v.to(torch.bfloat16), patch[cut], action[cut:cut + H]).to(dev))
                    for k, v in beliefs.items()}
            gt_priv = [np.degrees(z["bar_tilt"][s0 + (cut + k) * 5]) for k in STEPS]
            print(f"\nep {e} (c {c_cm:+.1f} cm), probe {args.probe}: lift starts raw {lift}; {name} = model step {cut} (raw {cut * 5})")
            print(f"  {'':28s}" + "".join(f"{'+' + str(k):>8s}" for k in STEPS))
            print(f"  {'ground truth, privileged':28s}" + "".join(f"{v:+8.1f}" for v in gt_priv))
            print(f"  {'ground truth, image':28s}" + "".join(f"{tilt(real[cut + k]):+8.1f}" for k in STEPS))
            for k_name in ("no memory", "posterior", "teacher"):
                vals = [tilt(real[cut])] + [tilt(imgs[k_name][k - 1]) for k in STEPS[1:]]
                print(f"  {'SMT ' + k_name + ', image':28s}" + "".join(f"{v:+8.1f}" for v in vals))
            rows = [np.concatenate([panel(real[cut + k], f"GT {name}", f"+{k}  ") for k in STEPS], 1),
                    np.concatenate([panel(real[cut], f"SMT teacher {name}", "real  ")] +
                                   [panel(imgs['teacher'][k - 1], f"SMT teacher {name}", f"+{k}  ") for k in STEPS[1:]], 1)]
            strips += rows
        path = out_dir / f"{Path(args.zarr).stem}_ep{e:03d}_probe{args.probe}_onset_strip.png"
        imageio.imwrite(path, np.concatenate(strips, 0))
        print(f"  strip: {path}")


if __name__ == "__main__":
    main()
