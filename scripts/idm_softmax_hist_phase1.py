"""What the latent-action IDM's softmax looks like, per block kind, on the test episodes.

    pixi run python scripts/idm_softmax_hist_phase1.py [--lap <latent_action_prior.pt>] [--out figures/idm_softmax]

Windows e_t .. e_{t+6} start at every hold-phase frame of memtest4_eval. Training-type windows (at rest before
a swap, or idle) are split by what really happens in them (ground truth used only to group the panels);
off-schedule windows (starting mid-swap) get their own row. Slots: 0 = left, 1 = middle, 2 = right.

Left column: mean softmax probability per code (linear). Right column: every window's probability per code
(log scale), so the near-zero mass on the other codes is visible.
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

from figure_style import FAINT, HIGHLIGHT, INK, INTERESTING, MUTED, apply  # noqa: E402
from info_gathering.swap_sequences import sequences  # noqa: E402
from train_mcl_prior_phase1 import load_model  # noqa: E402
from world_model.models.smt_phase1.latent_action_prior import LatentActionPrior  # noqa: E402

HOLD_PH, LIFT_PH, MOVE, SWAP = 1, 2, 4, 6
SLOT = {0: "left", 1: "middle", 2: "right"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/readout/main/checkpoints/wm_latest.pt")
    ap.add_argument("--lap", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/latent_action_prior/latent_action_prior.pt")
    ap.add_argument("--zarr", default="data/memtest4_eval.zarr")
    ap.add_argument("--out", default="figures/idm_softmax")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.run, dev)
    ck = torch.load(ROOT / args.lap, map_location=dev)
    lap = LatentActionPrior(model.prior, model.embed_dim, block=ck["block"], n_codes=ck["n_codes"]).to(dev).eval()
    lap.load_state_dict(ck["lap"])
    L, K = ck["block"], ck["n_codes"]

    zg = zarr.open_group(str(ROOT / args.zarr), mode="r")
    ends = zg["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    ph, seqs = zg["phase"][:], sequences(ROOT / args.zarr)
    FR, kind = [], []
    for e in range(len(ends)):
        s, n = int(starts[e]), int(ends[e] - starts[e])
        E = (torch.from_numpy(np.asarray(zg[model.embed_key][s:s + n], np.float32)).to(dev) - model.embed_mean) / model.embed_std
        p = ph[s:s + n]
        h0, on, sw = int(np.flatnonzero(p == HOLD_PH)[0]), int(np.flatnonzero(p == LIFT_PH)[0]), seqs[e][1]
        b0 = h0 + MOVE
        swap_at = {b0 + SWAP * j: tuple(sorted(pr)) for j, pr in enumerate(sw)}
        for t in range(h0, on - L):
            FR.append(E[t:t + L + 1])
            kind.append(swap_at[t] if t in swap_at else "idle" if t >= b0 + SWAP * len(sw) else "off")
    FR = torch.stack(FR)
    with torch.no_grad():
        logits = torch.cat([lap.inverse(FR[i:i + 512].flatten(1)) for i in range(0, len(FR), 512)]).float()
    P = logits.softmax(-1).cpu().numpy()
    kind = np.array(kind, dtype=object)

    rows = [((0, 2), "swap left ↔ right (0↔2)"), ((0, 1), "swap left ↔ middle (0↔1)"),
            ((1, 2), "swap middle ↔ right (1↔2)"), ("idle", "idle (nothing moves)"),
            ("off", "off-schedule (window starts mid-swap)")]
    masks = [np.array([k == key for k in kind]) for key, _ in rows]
    used = sorted({int(P[m].argmax(1)[i]) for m, (key, _) in zip(masks, rows) if key != "off"
                   for i in range(m.sum())})
    # highlight = the left<->right code (the figure's subject); the other used codes in the INTERESTING order
    lr = int(np.bincount(P[masks[0]].argmax(1)).argmax())
    color = {lr: HIGHLIGHT, **{c: INTERESTING[i] for i, c in enumerate(c for c in used if c != lr)}}
    bar_c = [color.get(c, FAINT) for c in range(K)]

    apply()
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 2.1 * len(rows)), sharex=True,
                             gridspec_kw={"width_ratios": [1, 1.25]})
    rng = np.random.default_rng(0)
    x = np.arange(K)
    for r, ((key, title), m) in enumerate(zip(rows, masks)):
        p = P[m]
        a0, a1 = axes[r]
        mean = p.mean(0)
        a0.bar(x, mean, width=0.8, color=bar_c, edgecolor="white", linewidth=1)
        a0.set_ylim(0, 1.05)
        a0.set_ylabel("mean prob.")
        a0.set_title(f"{title}  —  N = {len(p)} windows", loc="left", color=INK, fontsize=9.5)
        for c in range(K):
            if mean[c] > 0.03:
                a0.text(c, mean[c] + 0.02, f"{mean[c]:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        lp = np.log10(np.clip(p, 1e-16, 1))
        for c in range(K):
            a1.scatter(c + rng.uniform(-0.28, 0.28, len(lp)), lp[:, c], s=4, color=bar_c[c], alpha=0.35,
                       linewidths=0)
            a1.plot([c - 0.35, c + 0.35], [np.median(lp[:, c])] * 2, color=INK, lw=1.5)
        a1.set_ylim(-16.5, 0.5)
        a1.set_yticks([0, -4, -8, -12, -16])
        a1.set_yticklabels(["1", "1e-4", "1e-8", "1e-12", "≤1e-16"])
        a1.set_ylabel("prob. (log)")
        for a in (a0, a1):
            a.yaxis.grid(True)
            a.set_axisbelow(True)
    for a in axes[-1]:
        a.set_xticks(x)
        a.set_xticklabels([f"u{c}" for c in range(K)], fontsize=8)
        a.set_xlabel("code")
    for c in used:
        axes[-1][0].get_xticklabels()[c].set_color(color[c])
        axes[-1][0].get_xticklabels()[c].set_fontweight("bold")
    fig.suptitle("Latent-action IDM: softmax over the 16 codes, test episodes (memtest4_eval)", x=0.01, y=0.995, ha="left",
                 fontsize=11, color=INK)
    fig.text(0.01, 0.962, "Value = softmax of the IDM's 16 logits for one 7-frame window e_t..e_{t+6} (sums to 1 "
             "per window). Left: mean over windows. Right: every window (dot), black line = median. "
             "Coloured = the 4 codes ever chosen on training-type windows (orange = the left↔right code); grey = never.",
             fontsize=8, color=MUTED, ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out / "idm_softmax_by_kind.png", dpi=150)
    fig.savefig(out / "idm_softmax_by_kind.pdf")
    np.savez(out / "idm_softmax_by_kind.npz", probs=P, kind=np.array([str(k) for k in kind]))
    print(f"-> {out / 'idm_softmax_by_kind.png'}")
    for (key, title), m in zip(rows, masks):
        p = P[m]
        top = p.argmax(1)
        print(f"{title:40s} N {len(p):5d} | argmax: " + ", ".join(f"u{c} {np.mean(top == c):.2f}" for c in np.unique(top))
              + f" | winning prob median {np.median(p.max(1)):.4f} min {p.max(1).min():.4f}")


if __name__ == "__main__":
    main()
