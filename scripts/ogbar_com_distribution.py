"""The OGBar v3 bar and where its hidden COM can be: a to-scale sketch of the bar (40 cm capsule, point masses at
+-18 cm) above the per-episode COM offset c of the training store (histogram), with the held-out store and the 5 eval
episodes marked. c = (W_R - W_L) / 200 g x 18 cm, W_R ~ U[30, 170] g, so c ~ U[-12.6, +12.6] cm (docs/OGBAR.md).

    pixi run python scripts/ogbar_com_distribution.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

from figure_style import HIGHLIGHT, INTERESTING, OBVIOUS, apply  # noqa: E402


def load(store):
    z = zarr.open_group(str(ROOT / store), "r")
    return z["episode/com_offset_c"][:] * 100, z["episode/w_right"][:] * 1000


def main():
    apply()
    c_tr, w_tr = load("data/ogbar_v3_400.zarr")
    c_ho = load("data/ogbar_v3_heldout200.zarr")[0] if (ROOT / "data/ogbar_v3_heldout200.zarr").exists() else None
    c_ev = load("data/ogbar_v3_eval5.zarr")[0]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2], hspace=0.08))

    # to-scale bar: 36 cm straight part + 2 cm-radius round ends = 40 cm, 4 cm across
    ax0.add_patch(FancyBboxPatch((-18, -2), 36, 4, boxstyle="round,pad=0,rounding_size=2", fc="#e34948", ec="none", alpha=0.35))
    ax0.add_patch(FancyBboxPatch((-20, -2), 40, 4, boxstyle="round,pad=0,rounding_size=2", fc="none", ec=OBVIOUS[1], lw=0.8))
    for x, lab in ((-18, "W_L"), (18, "W_R")):
        ax0.add_patch(Circle((x, 0), 1.0, fc=OBVIOUS[0], ec="none"))
        ax0.text(x, 3.0, f"{lab}\n(hidden, {x:+d} cm)", ha="center", va="bottom", fontsize=7.5)
    ax0.axvspan(-12.6, 12.6, ymin=0.25, ymax=0.75, color=HIGHLIGHT, alpha=0.25, lw=0)
    ax0.plot([0], [0], marker="+", color=OBVIOUS[1], ms=8)
    ax0.text(0, -3.2, "bar middle", ha="center", va="top", fontsize=7.5, color=OBVIOUS[1])
    ax0.text(12.9, -0.2, "COM can be\nanywhere here", ha="left", va="center", fontsize=7.5, color=HIGHLIGHT)
    ax0.set_ylim(-6, 8.5)
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.axis("off")
    ax0.set_title("OGBar v3 bar: 40 cm capsule, 200 g, all mass in two hidden points; both ends look identical",
                  loc="left", fontsize=9)

    bins = np.linspace(-14, 14, 29)
    ax1.hist(c_tr, bins=bins, color=HIGHLIGHT, alpha=0.85, label=f"training store, {len(c_tr)} episodes")
    if c_ho is not None:
        ax1.hist(c_ho, bins=bins, histtype="step", color=INTERESTING[0], lw=1.4, label=f"held-out store, {len(c_ho)} episodes")
    exp = len(c_tr) * (bins[1] - bins[0]) / 25.2
    ax1.plot([-12.6, -12.6, 12.6, 12.6], [0, exp, exp, 0], color=OBVIOUS[1], lw=1, ls="--", label="uniform on ±12.6 cm (expected)")
    ax1.plot(c_ev, np.full(len(c_ev), -1.2), "v", color=OBVIOUS[0], ms=6, clip_on=False, label="5 eval episodes")
    ax1.set_ylim(-2, None)
    ax1.set_xlim(-21, 21)
    ax1.set_xlabel("COM offset c from the bar's middle (cm, along the bar's body axis)")
    ax1.set_ylabel("episodes per 1 cm bin")
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax1.text(20.5, ax1.get_ylim()[1] * 0.95,
             f"training: min {c_tr.min():+.1f}, max {c_tr.max():+.1f} cm\nmean {c_tr.mean():+.2f}, sd {c_tr.std():.2f} cm"
             f" (uniform: sd {25.2 / np.sqrt(12):.2f})\nW_R {w_tr.min():.0f}-{w_tr.max():.0f} g",
             ha="right", va="top", fontsize=7.5)
    out = ROOT / "figures/ogbar_com_distribution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out, f"train c min {c_tr.min():.2f} max {c_tr.max():.2f} mean {c_tr.mean():.2f} sd {c_tr.std():.2f}",
          "" if c_ho is None else f"heldout mean {c_ho.mean():.2f} sd {c_ho.std():.2f}")


if __name__ == "__main__":
    main()
