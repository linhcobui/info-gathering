"""How sparse are the OGBar play episodes? Observable start states and whole-trajectory coverage.

Top row (one point per episode, N = 400 training + 5 eval): the bar's start position (x, y) with its long axis drawn,
the start yaw (folded to 0-180 deg: the ends look identical), the gripper's start position, and each start's nearest
other start (distance between bar centres, and the yaw difference to that neighbour).
Bottom row (every 5th frame of every training episode): density of the bar's and the gripper's x-y position, and of their
heights. All quantities are simulator state (privileged), used only to describe the data.

    pixi run python scripts/ogbar_start_density.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
import zarr  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from figure_style import FAINT, HIGHLIGHT, INK, INTERESTING, MUTED, apply  # noqa: E402


def starts(store):
    z = zarr.open_group(str(ROOT / store), "r")
    ends = z["episode_ends"][:]
    s0 = np.r_[0, ends[:-1]]
    return z, s0, ends, dict(bar=z["bar_pos"][:][s0], yaw=np.degrees(z["bar_yaw_mod_pi"][:][s0]),
                             eff=z["effector_pos"][:][s0])


def main():
    apply()
    cfg = yaml.safe_load(open(ROOT / "configs/data/ogbar_slot_v3_flat_noslip.yaml"))
    z, s0, ends, S = starts("data/ogbar_v3_400.zarr")
    _, _, _, E = starts("data/ogbar_v3_eval5.zarr")
    xy = S["bar"][:, :2] * 100                                                  # cm
    D = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    nn_d = D[np.arange(len(xy)), nn]
    dyaw = np.abs(S["yaw"] - S["yaw"][nn])
    nn_yaw = np.minimum(dyaw, 180 - dyaw)
    close = (D < 2.0) & (np.abs(((S["yaw"][:, None] - S["yaw"][None]) + 90) % 180 - 90) < 10)
    print(f"bar start x {xy[:, 0].min():.1f}..{xy[:, 0].max():.1f} cm, y {xy[:, 1].min():.1f}..{xy[:, 1].max():.1f} cm "
          f"(area {np.ptp(xy[:, 0]) * np.ptp(xy[:, 1]):.0f} cm^2), yaw 0..180 deg; z {S['bar'][:, 2].min()*100:.1f}-"
          f"{S['bar'][:, 2].max()*100:.1f} cm")
    print(f"nearest other start: median {np.median(nn_d):.1f} cm (10-90% {np.percentile(nn_d, 10):.1f}-"
          f"{np.percentile(nn_d, 90):.1f}); its yaw differs by median {np.median(nn_yaw):.0f} deg")
    print(f"starts with another start within 2 cm AND 10 deg: {close.any(1).sum()} of {len(xy)}")
    print(f"gripper start x {S['eff'][:, 0].min()*100:.1f}..{S['eff'][:, 0].max()*100:.1f}, y {S['eff'][:, 1].min()*100:.1f}.."
          f"{S['eff'][:, 1].max()*100:.1f}, z {S['eff'][:, 2].min()*100:.1f}..{S['eff'][:, 2].max()*100:.1f} cm")

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6), gridspec_kw=dict(width_ratios=[1.3, 1, 1.3, 1]))
    (bx0, by0), (bx1, by1) = np.array(cfg["bar"]["boundary"]) * 100
    sl = cfg["slot"]
    hl = cfg["bar"]["half_length"] * 100

    def floor(ax):
        ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec=FAINT, lw=1))
        ax.add_patch(Rectangle((sl["x"] * 100 - 21, sl["y"] * 100 - 6), 42, 12, fc=FAINT, ec="none", alpha=0.6))
        ax.text(sl["x"] * 100, sl["y"] * 100, "slot", ha="center", va="center", color=MUTED, fontsize=8)
        ax.set_aspect("equal"); ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
        ax.set_xlim(bx0 - 3, bx1 + 3); ax.set_ylim(by0 - 3, by1 + 3)

    ax = axes[0, 0]
    floor(ax)
    for (x, y), yaw in zip(xy, np.radians(S["yaw"])):
        dx, dy = hl * np.cos(yaw), hl * np.sin(yaw)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=MUTED, lw=0.4, alpha=0.35)
    ax.scatter(xy[:, 0], xy[:, 1], s=9, color=HIGHLIGHT, label="training start (bar centre)", zorder=3)
    ax.scatter(E["bar"][:, 0] * 100, E["bar"][:, 1] * 100, s=60, marker="*", color=INTERESTING[0], label="eval start", zorder=4)
    ax.set_title(f"bar start pose, N = {len(xy)} (grey = bar's long axis)", loc="left")
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[0, 1]
    ax.hist(S["yaw"], bins=np.arange(0, 181, 10), color=HIGHLIGHT)
    ax.axhline(len(xy) / 18, color=MUTED, ls="--", lw=1, label="uniform")
    ax.set_xlabel("start yaw, folded to 0-180 deg"); ax.set_ylabel("episodes"); ax.legend(fontsize=7)
    ax.set_title("start yaw", loc="left")

    ax = axes[0, 2]
    floor(ax)
    ax.scatter(S["eff"][:, 0] * 100, S["eff"][:, 1] * 100, c=S["eff"][:, 2] * 100, s=10,
               cmap=LinearSegmentedColormap.from_list("h", [FAINT, INTERESTING[0]]))
    ax.set_title("gripper start position (colour = height)", loc="left")

    ax = axes[0, 3]
    ax.scatter(nn_d, nn_yaw, s=9, color=HIGHLIGHT)
    ax.axvline(np.median(nn_d), color=MUTED, ls="--", lw=1)
    ax.text(np.median(nn_d), 88, f" median {np.median(nn_d):.1f} cm", color=MUTED, fontsize=8, va="top")
    ax.set_xlabel("distance to nearest other start (cm, bar centres)")
    ax.set_ylabel("yaw difference to that start (deg)")
    ax.set_title("how far apart starts are", loc="left")

    fr = np.concatenate([np.arange(a, b, 5) for a, b in zip(s0, ends)])
    bar_all, eff_all = z["bar_pos"][:][fr] * 100, z["effector_pos"][:][fr] * 100
    cmap = LinearSegmentedColormap.from_list("d", ["#ffffff", FAINT, INTERESTING[0]])
    for ax, P, name in ((axes[1, 0], bar_all, "bar centre"), (axes[1, 2], eff_all, "gripper")):
        floor(ax)
        h = ax.hist2d(P[:, 0], P[:, 1], bins=[np.arange(bx0, bx1 + 1, 1), np.arange(by0, by1 + 1, 1)], cmap=cmap,
                      norm=LogNorm(), cmin=1)
        fig.colorbar(h[3], ax=ax, fraction=0.04, label="frames per 1 cm cell")
        occ = (h[0] > 0).mean()
        ax.set_title(f"{name} x-y over all frames (every 5th), {occ:.0%} of 1 cm cells visited", loc="left")
    ax = axes[1, 1]
    ax.hist(bar_all[:, 2], bins=np.arange(0, 40, 1), color=HIGHLIGHT, log=True)
    ax.set_xlabel("bar centre height (cm)"); ax.set_ylabel("frames (log)"); ax.set_title("bar height, all frames", loc="left")
    ax = axes[1, 3]
    ax.hist(eff_all[:, 2], bins=np.arange(0, 40, 1), color=INTERESTING[0], log=True)
    ax.set_xlabel("gripper height (cm)"); ax.set_ylabel("frames (log)"); ax.set_title("gripper height, all frames", loc="left")
    fig.suptitle(f"OGBar v3 play data, 400 training episodes ({len(fr)} frames at stride 5): start states (top) and coverage "
                 "over whole trajectories (bottom). Simulator state, used only to describe the data.", x=0.01, ha="left")
    fig.tight_layout()
    out = ROOT / "figures/ogbar_smt/start_density.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
