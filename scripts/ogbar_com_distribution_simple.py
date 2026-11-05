"""Minimal bar + COM-distribution pictures for slides: the stylised bar (40 cm capsule) with a density drawn above it.
No axes, no labels; every picture has the same canvas, so they line up on a slide.

  uniform    the training distribution: uniform on +-12.5 cm (400 training episodes: sd 7.25 cm vs 7.27 for the
             exact uniform), smoothed by a 1.5 cm Gaussian so the edges taper
  middle     a tighter Gaussian on the middle (sd 3 cm)
  right      a tighter Gaussian on the right (mean +8 cm, sd 2.5 cm)
  bimodal    two Gaussians, left and right (+-8 cm, sd 2.5 cm each)
  point      collapsed to a point at +8 cm (sd 0.25 cm)
Illustrative shapes (except uniform), all drawn with the same peak height. Writes
figures/ogbar_com_<name>.{png,svg}.

    pixi run python scripts/ogbar_com_distribution_simple.py
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
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from scipy.special import erf  # noqa: E402

from figure_style import HIGHLIGHT, apply  # noqa: E402

X = np.linspace(-20, 20, 2000)
PEAK, BASE = 6.0, 3.5                         # curve height and its baseline, in bar units (cm)


def gauss(mu, sd):
    return np.exp(-0.5 * ((X - mu) / sd) ** 2)


def draw(dens, name):
    dens = dens / dens.max() * PEAK
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.fill_between(X, BASE, BASE + dens, color=HIGHLIGHT, alpha=0.25, lw=0)
    ax.plot(X, BASE + dens, color=HIGHLIGHT, lw=2)
    bar = "#e34948"                                                          # 40 cm capsule, 4 cm across
    ax.add_patch(Rectangle((-18, -2), 36, 4, fc=bar, ec="none"))
    for xe in (-18, 18):
        ax.add_patch(Circle((xe, 0), 2, fc=bar, ec="none"))
    ax.set_xlim(-21, 21)
    ax.set_ylim(-2.5, BASE + PEAK + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    for ext in ("png", "svg"):
        fig.savefig(ROOT / f"figures/ogbar_com_{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    print(ROOT / f"figures/ogbar_com_{name}.png")


def main():
    apply()
    c = zarr.open_group(str(ROOT / "data/ogbar_v3_400.zarr"), "r")["episode/com_offset_c"][:] * 100
    lo, hi, s = float(c.min()), float(c.max()), 1.5
    draw(0.5 * (erf((X - lo) / (s * np.sqrt(2))) - erf((X - hi) / (s * np.sqrt(2)))), "uniform")
    draw(gauss(0, 3.0), "middle")
    draw(gauss(8, 2.5), "right")
    draw(gauss(-8, 2.5) + gauss(8, 2.5), "bimodal")
    draw(gauss(8, 0.25), "point")


if __name__ == "__main__":
    main()
