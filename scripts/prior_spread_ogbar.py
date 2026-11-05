"""Is the OGBar-SMT prior's spread collapsed? Per model step, on teacher beliefs:
  sigma_P        mean of P's predicted spread over the 2,560 numbers of z_{t+1} (sigma_min = 0.01)
  error          rms of (z*_{t+1} - mu_P): how far the real next belief lands from P's mean
  change         rms of (z*_{t+1} - z*_t): how much the belief actually moves in one step
A calibrated Gaussian has error ~ sigma_P. Lift onsets (probe stage 'lift' starting) are marked.
Also: N prior rollouts from one cut, to see whether samples ever diverge.

    pixi run python scripts/prior_spread_ogbar.py --ckpt <wm_latest.pt>
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

from info_gathering.ogbar_smt_data import OgbarSmtAnchors  # noqa: E402
from world_model.models.smt_ogbar import SmtOgbar  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

AMP = dict(device_type="cuda", dtype=torch.bfloat16)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#2b2a27", "#6b6a65", "#e7e6e2"


@torch.no_grad()
def per_step(m, data, e, dev):
    p, a = data.load(e, 0)
    p, a = p.to(dev).to(torch.bfloat16), m.norm_action(a.to(dev))
    T = len(p)
    with torch.autocast(**AMP):
        pooled = m.pool(p[None])
        zt = torch.cat([m.teacher_windows(pooled, a[None], torch.arange(i, min(i + 50, T), device=dev)[None])[0]
                        for i in range(0, T, 50)]).float()
        sig, err = [], []
        for i in range(0, T - 1, 64):
            j = min(i + 64, T - 1)
            mu, ls = m.prior(zt[i:j].to(torch.bfloat16), p[i:j], a[i:j])
            sig.append(ls.exp().float().mean((1, 2)))
            err.append((zt[i + 1:j + 1] - mu.float()).pow(2).mean((1, 2)).sqrt())
    change = (zt[1:] - zt[:-1]).pow(2).mean((1, 2)).sqrt()
    return torch.cat(sig).cpu().numpy(), torch.cat(err).cpu().numpy(), change.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-seen", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device(args.device)
    st = CheckpointManager.load(args.ckpt)
    m = SmtOgbar(action_dim=25, **st["config"]["model"]).to(dev).eval()
    m.load_state_dict(st["model"])
    print(f"checkpoint step {st['step']}; sigma_min {m.cfg['sigma_min']}, sigma_max {m.cfg['sigma_max']}")
    res = {}
    for store, eps in (("data/ogbar_v3_eval5.zarr", range(5)), ("data/ogbar_v3_400.zarr", range(0, 400, 400 // args.n_seen))):
        data = OgbarSmtAnchors(ROOT / (store.removesuffix(".zarr") + "_dinov2"), ROOT / store, 5)
        z = zarr.open_group(str(ROOT / store), "r")
        ps = z["probe_stage"][:]
        for e in eps:
            s0, e0 = int(data.starts[e]), int(data.ends[e])
            sig, err, ch = per_step(m, data, e, dev)
            lifts = [t // 5 for t in range(s0 + 1, e0) if ps[t] == 1 and ps[t - 1] != 1]
            lifts = [l - s0 // 5 for l in lifts] if False else [(t - s0) // 5 for t in range(s0 + 1, e0) if ps[t] == 1 and ps[t - 1] != 1]
            res[(store, e)] = (sig, err, ch, lifts)
    for name, store in (("unseen eval", "data/ogbar_v3_eval5.zarr"), ("seen training", "data/ogbar_v3_400.zarr")):
        S = np.concatenate([v[0] for k, v in res.items() if k[0] == store])
        E = np.concatenate([v[1] for k, v in res.items() if k[0] == store])
        C = np.concatenate([v[2] for k, v in res.items() if k[0] == store])
        on = np.concatenate([np.isin(np.arange(len(v[0])), [l + d for l in v[3] for d in (-1, 0, 1, 2)]) for k, v in res.items() if k[0] == store])
        print(f"{name}: steps {len(S)}")
        print(f"   sigma_P   mean {S.mean():.4f}  min {S.min():.4f}  max {S.max():.4f}   at lift onsets {S[on].mean():.4f}  elsewhere {S[~on].mean():.4f}")
        print(f"   error     mean {E.mean():.4f}  (rms distance of the real next belief from P's mean)   at onsets {E[on].mean():.4f}  elsewhere {E[~on].mean():.4f}")
        print(f"   change    mean {C.mean():.4f}  (rms one-step move of the teacher belief)            at onsets {C[on].mean():.4f}  elsewhere {C[~on].mean():.4f}")
        print(f"   share of steps with sigma_P within 1% of the floor: {np.mean(S < m.cfg['sigma_min'] * 1.01):.0%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8), facecolor=SURFACE, sharey=True)
    for e, ax in enumerate(axes):
        sig, err, ch, lifts = res[("data/ogbar_v3_eval5.zarr", e)]
        t = np.arange(len(sig))
        ax.plot(t, sig, color="#2a78d6", lw=1.6, label="prior spread sigma_P")
        ax.plot(t, err, color="#eb6834", lw=1.2, label="real next belief - mu_P (rms)")
        ax.plot(t, ch, color=INK, lw=1.0, ls="--", label="teacher one-step change (rms)")
        ax.axhline(m.cfg["sigma_min"], color=MUTED, lw=0.8, ls=":")
        for l in lifts:
            ax.axvline(l, color=GRID, lw=1.0, zorder=0)
        ax.set_yscale("log")
        ax.set_title(f"unseen eval ep {e}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("model step (grey = lift starts; dotted = sigma floor)", color=MUTED, fontsize=8)
        ax.set_facecolor(SURFACE)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.suptitle(f"OGBar-SMT prior P (checkpoint step {st['step']}): predicted spread vs actual error, per step", color=INK,
                 fontsize=12, x=0.01, ha="left")
    out = ROOT / "figures/ogbar_smt_prior_spread_eval5.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
