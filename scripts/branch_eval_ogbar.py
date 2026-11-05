"""Imagination on counterfactual branches: same history, different grasp. Does the imagined tilt follow the new action?

Every branch shares the source episode's history up to the cut (branch_episode_ogbar.py), so every belief built from
that history is the same for all branches; only the recorded actions after the cut differ. From the cut, with each
branch's actions:
  SMT teacher -> prior    z_cut = teacher on the history up to the cut, then P, frames from the decoder's first output
  SMT updater -> prior    z_cut from the posterior Q chained over the real history from a step-0 teacher warm-up
  SMT no memory -> prior  z_cut = teacher on a window holding only x_cut
  DINO-WM                 open-loop from its 3 real frames ending at the cut
Score: the bar's angle in the side-view image (line through its red pixels), averaged over the model steps where the real
bar is in the probe's hold stage. Writes a tilt-vs-grasp figure, a hold-frame grid and one video per branch.

    pixi run python scripts/branch_eval_ogbar.py --branches data/ogbar_v3_ep033_branches.zarr
"""
import argparse
import json
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
from info_gathering.ogbar_dino import OgbarDinoWindows  # noqa: E402
from info_gathering.ogbar_smt_data import OgbarSmtAnchors  # noqa: E402
from train_dino_wm import build_model  # noqa: E402
from world_model.models.smt_ogbar import SmtOgbar  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

AMP = dict(device_type="cuda", dtype=torch.bfloat16)
HOLD = 2
NAMES = ["SMT teacher -> prior", "SMT updater -> prior", "SMT no memory -> prior", "DINO-WM"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smt", default="logdir/world_model/smt_ogbar/v3_400_gaussian/checkpoints/wm_latest.pt")
    ap.add_argument("--dino-wm", default="logdir/world_model/dino_wm_ogbar/v3_400_proprio/checkpoints/wm_latest.pt")
    ap.add_argument("--branches", default="data/ogbar_v3_ep033_branches.zarr")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)
    st = CheckpointManager.load(args.smt)
    m = SmtOgbar(action_dim=25, **st["config"]["model"]).to(dev).eval()
    m.load_state_dict(st["model"])
    dst = CheckpointManager.load(args.dino_wm)
    dw = build_model(OmegaConf.create(dst["config"]), dst["stats"], 25, 19).to(dev).eval()
    dw.load_state_dict(dst["model"])
    z = zarr.open_group(str(ROOT / args.branches), "r")
    cfgb = json.loads(z.attrs["config"])["branch"] if isinstance(z.attrs["config"], str) else z.attrs["config"]["branch"]
    s_list, T = list(z.attrs["branch_s"]), int(cfgb["raw_t"])
    cut = T // 5
    cache = ROOT / (args.branches.removesuffix(".zarr") + "_dinov2")
    data = OgbarSmtAnchors(cache, ROOT / args.branches, 5)
    ddata = OgbarDinoWindows(cache, ROOT / args.branches, num_frames=dw.num_hist + 1, frameskip=5)
    c_cm = float(z["episode/com_offset_c"][0]) * 100
    out_dir = ROOT / "figures/ogbar_smt/branches"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, results, z_ref = [], [], None
    tilt = lambda im: img_tilt(im[27:])
    for e, s in sorted(enumerate(s_list), key=lambda p: p[1]):
        s0, e0 = int(data.starts[e]), int(data.ends[e])
        patch, action = data.load(e, 0)
        patch, action = patch.to(dev).to(torch.bfloat16), m.norm_action(action.to(dev))
        Tn = len(patch)
        ps = z["probe_stage"][s0:e0]
        hold = sorted({t // 5 for t in range(T, e0 - s0) if ps[t] == HOLD and t // 5 > cut and t // 5 < Tn})
        H = min(max(hold) - cut + 4, Tn - 1 - cut)
        real = z["rgb"][s0:e0][::5][:Tn]
        with torch.no_grad(), torch.autocast(**AMP):
            pooled = m.pool(patch[None])
            z_t = m.teacher_windows(pooled, action[None], torch.full((1, 1), cut, dtype=torch.long, device=dev))[0, 0]
            W = m.cfg["window"]
            idx = torch.arange(cut - W + 1, cut + 1, device=dev)
            prev = torch.cat([torch.zeros_like(action[:1]), action[:-1]], 0)
            z_b = m.teacher.tokens(pooled[:, idx.clamp(min=0)], prev[idx.clamp(min=0)][None].to(pooled.dtype), (idx == cut)[None])[0]
            z_u = m.teacher_windows(pooled, action[None], torch.zeros(1, 1, dtype=torch.long, device=dev))[0, 0]
            for t in range(cut):
                mu, ls = m.posterior(z_u[None], patch[t][None], action[t][None], patch[t + 1][None])
                z_u = m.sample(mu, ls)[0]
            seq = ddata.episode_sequence(e, load_rgb=False)
            h = dw.num_hist
            v, _ = dw.rollout(seq["visual"][None, cut - h + 1:cut + 1].to(dev), seq["proprio"][None, cut - h + 1:cut + 1].to(dev),
                              seq["action"][None, cut - h + 1:cut + H].to(dev))
        if z_ref is None:
            z_ref = z_t.float()
        same = float((z_t.float() - z_ref).abs().max())
        acts = action[cut:cut + H]
        ims = [decode_all(dw, prior_rollout(m, zz, patch[cut], acts).to(dev)) for zz in (z_t, z_u, z_b)]
        ims.append(decode_all(dw, v[0, h:h + H].float().to(dev)))
        gt = float(np.nanmean([tilt(real[k]) for k in hold]))
        pred = [float(np.nanmean([tilt(im[k - cut - 1]) for k in hold if k - cut - 1 < len(im)])) for im in ims]
        priv = float(np.degrees(np.mean([z["bar_tilt"][s0 + t] for t in range(T, e0 - s0) if ps[t] == HOLD])))
        results.append(dict(s=s * 100, gt=gt, priv=priv, pred=pred))
        print(f"s {s * 100:+6.1f} cm (from COM {s * 100 - c_cm:+5.1f}): hold tilt GT {gt:+6.1f} (privileged {priv:+6.1f}) | "
              + " | ".join(f"{n.split(' ->')[0].replace('SMT ', '')} {p:+6.1f}" for n, p in zip(NAMES, pred))
              + f"   [teacher belief identical to branch 1: max diff {same:.1e}]", flush=True)
        mid = hold[len(hold) // 2]
        row = [panel(real[mid], f"GT  grasp {s * 100:+.1f} cm", f"hold  ")] + \
              [panel(im[mid - cut - 1], n, "") for n, im in zip(NAMES, ims)]
        rows.append(np.concatenate(row, 1))
        frames = []
        for k in range(H + 1):
            fr = [panel(real[cut + k], f"GT grasp {s * 100:+.1f} cm  c {c_cm:+.1f}", f"+{k}  ")]
            fr += [panel(real[cut] if k == 0 else im[k - 1], n, f"+{k}  ") for n, im in zip(NAMES, ims)]
            frames.append(np.concatenate(fr, 1))
        imageio.mimwrite(out_dir / f"ep033_branch_s{s * 100:+.1f}cm.mp4", frames, fps=4, macro_block_size=8)
    imageio.imwrite(out_dir / "ep033_branches_hold_grid.png", np.concatenate(rows, 0))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure_style import HIGHLIGHT, INK, INTERESTING, MUTED, apply
    apply()
    S = np.array([r["s"] for r in results])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(S, [r["gt"] for r in results], color=INK, marker="o", lw=2, label="ground truth (simulated branch)")
    for n, col, mk in zip(NAMES, [HIGHLIGHT, INTERESTING[0], MUTED, INTERESTING[2]], ["o", "s", "^", "D"]):
        ax.plot(S, [r["pred"][NAMES.index(n)] for r in results], color=col, marker=mk, lw=1.5, label=n)
    ax.axvline(c_cm, color=MUTED, ls=":", lw=1)
    ax.text(c_cm, ax.get_ylim()[1], " true COM", color=MUTED, va="top", fontsize=8)
    ax.axvline(-9.3, color=MUTED, ls="--", lw=0.8)
    ax.text(-9.3, ax.get_ylim()[0], " original grasp", color=MUTED, va="bottom", fontsize=8)
    ax.set_xlabel("grasp point of the branched probe (cm from the bar's middle)")
    ax.set_ylabel("bar angle in the side-view image at the hold (deg)")
    ax.set_title(f"Episode 33, branched at model step {cut}: identical history, different grasp (N = {len(S)} branches)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.text(0.01, -0.04, "Angle = line fitted through the bar's red pixels in the 224 px side view, averaged over the model steps "
             "where the real bar is in the probe's hold stage. SMT: 400-episode run, step 25,000, no uniformity.", fontsize=7)
    fig.savefig(out_dir / "ep033_branches_tilt_vs_grasp.png", bbox_inches="tight")
    print(out_dir / "ep033_branches_tilt_vs_grasp.png")
    print(out_dir / "ep033_branches_hold_grid.png")


if __name__ == "__main__":
    main()
