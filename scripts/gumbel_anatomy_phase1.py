"""Every quantity inside one straight-through Gumbel-softmax training step, at several training snapshots.

    pixi run python scripts/gumbel_anatomy_phase1.py [--steps 0 100 3000] [--out figures/gumbel_anatomy]

One real training block (a left<->right = 0<->2 swap from memtest4 training episodes), one fixed uniform draw U
shared by every column, so columns differ only by the model (and by tau, which follows the training schedule
tau = 1.0 * 0.3^(step/3000)). For each snapshot of the latent-action run (train_latent_action_prior_phase1.py
--save_at), it runs exactly the training computation in fp32:

    l = IDM(e_t..e_{t+6});  g = -log(-log U);  y_soft = softmax((l + g)/tau);  y_hard = one_hot(argmax y_soft)
    u = y_hard - detach(y_soft) + y_soft;  6-step rollout from z_t with u;  L = MSE to teacher z_{t+1..t+6}

and backpropagates L once to read dL/du and dL/dl. Last row: which code gets picked over 10,000 fresh noise draws.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

from figure_style import FAINT, HIGHLIGHT, INK, MUTED, apply  # noqa: E402
from info_gathering.swap_sequences import sequences  # noqa: E402
from train_mcl_prior_phase1 import load_model  # noqa: E402
from world_model.models.smt_phase1.latent_action_prior import LatentActionPrior  # noqa: E402

HOLD_PH, MOVE, SWAP = 1, 4, 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/readout/main/checkpoints/wm_latest.pt")
    ap.add_argument("--snapdir", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/latent_action_prior_snapshots")
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 100, 3000])
    ap.add_argument("--zarr", default="data/memtest4.zarr")
    ap.add_argument("--pair", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--skip_eps", type=int, nargs="*", default=[5], help="episodes whose swap timing fails the cross-check")
    ap.add_argument("--out", default="figures/gumbel_anatomy")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    model = load_model(args.run, dev)
    key = model.embed_key
    zg = zarr.open_group(str(ROOT / args.zarr), mode="r")
    ends = zg["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    seqs, ph = sequences(ROOT / args.zarr), zg["phase"][:]
    want = tuple(sorted(args.pair))
    e, j = next((e, j) for e in range(len(ends)) if e not in args.skip_eps for j, pr in enumerate(seqs[e][1]) if tuple(sorted(pr)) == want)
    s, n = int(starts[e]), int(ends[e] - starts[e])
    E = ((torch.from_numpy(np.asarray(zg[key][s:s + n], np.float32)).to(dev) - model.embed_mean) / model.embed_std)[None]
    PA = torch.zeros(1, n, 2, device=dev)
    PA[0, 1:] = torch.from_numpy(zg["action"][s:s + n - 1].astype(np.float32)).to(dev)
    first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
    with torch.no_grad():
        Z = model.encode_windows(E, PA, first)[0].float()
    t = int(np.flatnonzero(ph[s:s + n] == HOLD_PH)[0]) + MOVE + SWAP * j
    L = SWAP
    z0, fr, a, y = Z[t][None], E[0, t:t + L + 1][None], PA[0, t + 1:t + 1 + L][None], Z[t + 1:t + 1 + L][None]
    print(f"block: training episode {e}, swap #{j + 1} = {want[0]}<->{want[1]}, frames {t}..{t + L}")
    with torch.no_grad():
        zd, dp = z0, []
        for i in range(L):
            zd = model.prior.step(zd, a[:, i]); dp.append(zd)
        print(f"old deterministic prior, same block: MSE {float((torch.stack(dp, 1) - y).pow(2).mean()):.4f}")

    K = 16
    U = torch.rand(K, device=dev)
    U_many = torch.rand(10_000, K, device=dev)
    cols = []
    for step in args.steps:
        ck = torch.load(ROOT / args.snapdir / f"latent_action_prior_step{step}.pt", map_location=dev)
        lap = LatentActionPrior(model.prior, model.embed_dim, block=ck["block"], n_codes=ck["n_codes"]).to(dev).float()
        lap.load_state_dict(ck["lap"])
        tau = 1.0 * 0.3 ** (step / 3000)
        logits = lap.inverse(fr.flatten(1))[0]
        logits.retain_grad()
        g = -torch.log(-torch.log(U))
        y_soft = torch.softmax((logits + g) / tau, -1)
        y_hard = torch.nn.functional.one_hot(y_soft.argmax(), K).float()
        u = y_hard - y_soft.detach() + y_soft
        u.retain_grad()
        path = lap.rollout(z0, a, u[None])
        loss = (path - y).pow(2).mean()
        loss.backward()
        with torch.no_grad():
            picks = ((logits + -torch.log(-torch.log(U_many))) / tau).argmax(1)
            freq = torch.bincount(picks, minlength=K).float() / len(picks)
        col = {k: v.detach().cpu().numpy() for k, v in
               dict(l=logits, g=g, noisy=(logits + g) / tau, y_soft=y_soft, y_hard=y_hard, u=u,
                    G=u.grad, dl=logits.grad, freq=freq, p=torch.softmax(logits, -1)).items()}
        col.update(step=step, tau=tau, loss=float(loss), k=int(y_hard.argmax()))
        cols.append(col)
        print(f"step {step:5d} tau {tau:.2f} | picked u{col['k']} | loss {col['loss']:.4f} | "
              f"y_soft max {col['y_soft'].max():.4f} | max|dL/dl| {np.abs(col['dl']).max():.2e} | "
              f"u - y_hard max {np.abs(col['u'] - col['y_hard']).max():.1e}")

    rows = [("l", "logits ℓ\n(IDM output)"), ("g", "Gumbel noise g\n= −log(−log U)"),
            ("noisy", "noisy score\n(ℓ + g) / τ"), ("y_soft", "y_soft = softmax\n((ℓ + g) / τ)"),
            ("y_hard", "y_hard = one-hot\nof argmax"), ("u", "u = y_hard − detach(y_soft)\n+ y_soft  (forward value)"),
            ("G", "∂L/∂u\n(gradient arriving at u)"), ("dl", "∂L/∂ℓ\n(gradient into the IDM)"),
            ("freq", "share of 10,000\nnoise draws picking\neach code")]
    apply(font_size=8.5)
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.1 * len(cols) + 1.2, 1.55 * len(rows)), sharex=True)
    x = np.arange(K)
    for c, col in enumerate(cols):
        bar_c = [HIGHLIGHT if i == col["k"] else MUTED for i in range(K)]
        for r, (key_, label) in enumerate(rows):
            ax = axes[r, c]
            v = col[key_]
            ax.bar(x, v, width=0.75, color=bar_c if key_ != "freq" else [FAINT] * K, edgecolor="white", linewidth=0.8)
            if key_ == "freq":
                ax.bar(x, v, width=0.75, color=[HIGHLIGHT if i == col["k"] else MUTED for i in range(K)],
                       edgecolor="white", linewidth=0.8)
                ax.plot(x, col["p"], "_", color=INK, ms=9, mew=1.5)
            ax.axhline(0, color=FAINT, lw=0.8)
            if key_ in ("y_soft", "y_hard", "u", "freq"):
                ax.set_ylim(0, 1.08)
            if key_ in ("G", "dl"):
                m = np.abs(v).max()
                ax.text(0.99, 0.95, f"max |·| = {m:.1e}", transform=ax.transAxes, ha="right", va="top",
                        fontsize=7.5, color=MUTED)
            if key_ == "y_soft":
                ax.text(0.99, 0.95, f"winner {v.max():.3f}", transform=ax.transAxes, ha="right", va="top",
                        fontsize=7.5, color=MUTED)
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8.5)
            if r == 0:
                stage = {0: "random init", 3000: "converged"}.get(col["step"], "early training")
                ax.set_title(f"step {col['step']} ({stage}), τ = {col['tau']:.2f}\nrollout MSE L = {col['loss']:.4f}",
                             fontsize=9.5)
        axes[-1, c].set_xticks(x)
        axes[-1, c].set_xticklabels([f"{i}" for i in x], fontsize=7)
        axes[-1, c].set_xlabel("code k")
    fig.suptitle(f"Straight-through Gumbel-softmax, one training block (swap left↔right, training episode {e})",
                 x=0.01, y=0.998, va="top", ha="left", fontsize=11)
    fig.text(0.01, 0.982,
             "Same uniform draw U in every column. Orange = the code picked this draw. L = MSE between the 6 imagined "
             "memories ẑ and the teacher's z_{t+1..t+6}, averaged over 6 steps × 10 tokens × 256 dims (one block). "
             "Last row: bars = how often each code wins over 10,000 fresh noise draws; black ticks = softmax(ℓ).",
             fontsize=7.5, color=MUTED, ha="left", va="top", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.958))
    fig.savefig(out / "gumbel_anatomy.png", dpi=150)
    fig.savefig(out / "gumbel_anatomy.pdf")
    print(f"-> {out / 'gumbel_anatomy.png'}")


if __name__ == "__main__":
    main()
