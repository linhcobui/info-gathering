"""ALL3 figures, all read from the saved result tables / label files:

    figures/all3_by_swaps.png       accuracy and F1 by number of swaps, all five runs
    figures/all3_runs_diagram.png   how each run is built (same data and model; what differs)
    figures/all3_labels.png         the two self-made label sets: how right, and how much they cover

    python3 scripts/all3_figures.py        (system python3: matplotlib, numpy)

Sources: logdir/world_model/smt_phase1_all3/depth_table_handcrafted_backtracking.md (scripts/onset_depth_table.py),
logdir/world_model/dreamerv3_memtest4_all3/main/depth_table.md (scripts/lift_eval_depth_table.py),
data/labels_psr_self_all3naive.npz and data/labels_backtrack_all3naive.npz.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURF, INK, INK2, MUTED, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f0efe9"
# fixed categorical order; a run keeps its colour in every figure
COL = {"Naive": "#2a78d6", "Align-self-predictions": "#eb6834", "Align-privileged": "#1baf7a",
       "Hand-crafted-backtracking": "#c98500", "DreamerV3": "#d55181"}
STYLE = {"Hand-crafted-backtracking": dict(ls=(0, (5, 3)), marker="o", ms=8, hollow=True),   # lies on Align-privileged
         "DreamerV3": dict(ls=(0, (1, 2)), marker="s", ms=5.5, hollow=False)}                   # different read-out
LABEL = {"DreamerV3": "DreamerV3 (imagined pixels)"}


def read_smt():
    text = (ROOT / "logdir/world_model/smt_phase1_all3/depth_table_handcrafted_backtracking.md").read_text()
    out = {}
    parts = re.split(r"\*\*(F1|accuracy)\*\*", text)
    for metric, body in zip(parts[1::2], parts[2::2]):
        lines = body.splitlines()
        header = [h.strip() for h in next(l for l in lines if l.startswith("| swaps")).split("|")[2:-1]]
        rows = [l for l in lines if l.startswith("| ") and l[2].isdigit()]
        out[metric] = {"swaps": [int(r.split("|")[1].split()[0]) for r in rows]}
        for i, name in enumerate(header):
            out[metric]["always ball" if name.startswith("always") else name] = [float(r.split("|")[2 + i]) for r in rows]
    return out


def add_dreamer(data):
    rows = [l.split("|") for l in (ROOT / "logdir/world_model/dreamerv3_memtest4_all3/main/depth_table.md").read_text().splitlines()
            if l.startswith("| ") and l[2].isdigit()]
    by = {int(r[1]): (float(r[3]), float(r[4])) for r in rows}
    for metric, k in (("accuracy", 0), ("F1", 1)):
        data[metric]["DreamerV3"] = [by[s][k] for s in data[metric]["swaps"]]
    return data


def style_axes(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def chart(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), dpi=170, sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, metric in zip(axes, ("accuracy", "F1")):
        d = data[metric]
        x = d["swaps"]
        ax.set_facecolor(SURF)
        ax.axvspan(-0.4, 3.5, color=BAND, zorder=0)
        ax.text(1.55, 0.06, "in training\n(every order of 0-3 swaps)", ha="center", fontsize=9, color=INK2)
        ax.text(6.0, 0.06, "never trained (OOD)", ha="center", fontsize=9, color=INK2)
        ax.plot(x, d["always ball"], color=MUTED, lw=1.5, ls=(0, (3, 3)), zorder=2, label='always "ball"')
        names = [n for n in COL if n in d]
        for name in names:
            st = STYLE.get(name, dict(ls="-", marker="o", ms=5.5, hollow=False))
            ax.plot(x, d[name], color=COL[name], lw=2.2, ls=st["ls"], marker=st["marker"], ms=st["ms"],
                    mfc=SURF if st["hollow"] else COL[name], mec=COL[name] if st["hollow"] else SURF, mew=1.6,
                    zorder=4 if st["hollow"] else 3, label=LABEL.get(name, name))
        # end-of-line labels, nudged apart (>= 0.065); runs ending on the same value share one label
        top = {}
        for n in names:
            top.setdefault(f"{d[n][-1]:.2f}", d[n][-1])
        ends = sorted([(v, t, INK) for t, v in top.items()] + [(d["always ball"][-1], 'always "ball"', MUTED)])
        ys = []
        for v, _, _ in ends:
            ys.append(max(v, ys[-1] + 0.065) if ys else v)
        for (v, t, c), y in zip(ends, ys):
            ax.text(8.2, y, t, fontsize=8.8, color=c, va="center")
        ax.set_xlim(-0.4, 9.4)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xlabel("number of swaps in the test episode", color=INK2, fontsize=10)
        ax.set_title("Accuracy" if metric == "accuracy" else "F1", loc="left", fontsize=12, color=INK, fontweight="bold")
        style_axes(ax)
    axes[0].set_ylabel("63 test episodes per point", color=INK2, fontsize=10)
    handles = [l for l in axes[0].lines if l.get_label() != 'always "ball"']
    fig.legend(handles=handles, labels=[l.get_label() for l in handles], loc="upper center", ncol=5, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.text(0.5, 0.005, "SMT runs: memory read by the reward decoder at the lift onset.   DreamerV3: lift imagined from "
             "the actions, decoded, ball found by a colour detector.", ha="center", fontsize=8.8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    out = ROOT / "figures/all3_by_swaps.png"
    fig.savefig(out, facecolor=SURF)
    print(out)


def box(ax, x, y, w, h, text, edge=GRID, fill="#ffffff", size=9.0, weight="normal", color=INK, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01", fc=fill, ec=edge, lw=1.4))
    tx = x + w / 2 if align == "center" else x + 0.01
    ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=size, color=color, weight=weight, linespacing=1.35)


def arrow(ax, pts, color=INK2):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.3, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, shrinkA=0, shrinkB=1))


def diagram(data):
    fig, ax = plt.subplots(figsize=(15, 9.2), dpi=170)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 0.995, "ALL3: five runs on the same data", fontsize=14, weight="bold", color=INK, va="top")
    ax.text(0.0, 0.962, "Four SMT runs share one model (WINDOW60 phase 1: 60-frame teacher, posterior + prior updaters, "
            "reward decoder over the memory and the next 60 frames, 20,000 steps) and differ only in the alignment term. "
            "DreamerV3 is a separate baseline model.", fontsize=9.2, color=INK2, va="top", wrap=True)
    for x, t in ((0.0, "TRAINING DATA"), (0.17, "WHAT SHAPES THE MEMORY / WHERE ALIGNMENT LABELS COME FROM"),
                 (0.60, "RUN"), (0.775, "ACCURACY AT THE LIFT")):
        ax.text(x, 0.905, t, fontsize=8.3, color=MUTED, weight="bold")
    box(ax, 0.0, 0.52, 0.14, 0.36, "memtest4_all3\n\nevery swap order\nof 0-3 swaps\n3 + 9 + 27 + 81\n= 120 orders\n\n"
        "384 episodes\n96 per depth\n(+24 held out)", fill="#f4f3ee", size=9)
    HX, HW, MX, MW = 0.17, 0.37, 0.60, 0.14
    rows = [
        ("Naive", 0.80, 0.07, "no alignment term"),
        ("Align-privileged", 0.665, 0.09, "pull-only on the TRUE ball slot\nleft 0.1 · middle 0.1 · right 0.1"),
        ("Align-self-predictions", 0.36, 0.115, "pull-only on the model's OWN classes, from IMAGINED swaps:\n"
         "decoder says 'ball' → 'here';  else imagine latent swap A / B\nand ask again → 'after A' / "
         "'after B'      weights 0.1 · 0.03 · 0.03"),
        ("Hand-crafted-backtracking", 0.215, 0.115, "pull-only on slots carried through REAL swaps:\nanchors where the "
         "decoder is sure 'ball here', walked back and forth\nthrough real swaps named by latent actions      all 0.1"),
        ("DreamerV3", 0.04, 0.10, "separate model: native DreamerV3, pixel reconstruction\nshapes the latent; scored by "
         "imagining the lift and detecting\nthe ball in the decoded frames (different read-out)"),
    ]
    for name, y, h, how in rows:
        c = COL[name]
        box(ax, HX, y, HW, h, how, edge=c, size=8.4 if h > 0.1 else 8.8,
            align="left" if h > 0.1 else "center")
        nm = {"Align-self-predictions": "Align-self-\npredictions", "Hand-crafted-backtracking": "Hand-crafted-\nbacktracking"}
        box(ax, MX, y + h / 2 - 0.04, MW, 0.08, nm.get(name, name), edge=c, fill=c + "22", size=9.4, weight="bold")
        arrow(ax, [(HX + HW, y + h / 2), (MX, y + h / 2)], color=c)
        arrow(ax, [(0.14, 0.70), (0.155, 0.70), (0.155, y + h / 2), (HX, y + h / 2)])
    # the frozen Naive (+ its latent actions) feeds both self-labelling runs
    box(ax, HX, 0.52, HW, 0.085, "frozen Naive  +  its latent actions (learned from frames only:\nnothing, 0↔1, 0↔2, 1↔2) "
        "→ labels for the two runs below", edge=COL["Naive"], size=8.4)
    lane = 0.565
    arrow(ax, [(MX, 0.81), (lane, 0.81), (lane, 0.5625), (HX + HW, 0.5625)], color=COL["Naive"])
    ax.text(lane + 0.004, 0.63, "frozen\ncopy", fontsize=8, color=COL["Naive"], va="center")
    arrow(ax, [(0.27, 0.52), (0.27, 0.475)], color=COL["Naive"])
    arrow(ax, [(0.45, 0.52), (0.45, 0.48), (0.555, 0.48), (0.555, 0.2725), (HX + HW, 0.2725)], color=COL["Naive"])
    # results
    acc = data["accuracy"]
    idx = {s: i for i, s in enumerate(acc["swaps"])}
    box(ax, 0.775, 0.04, 0.225, 0.84, "", fill="#ffffff")
    ax.text(0.785, 0.85, "63 test episodes per depth;\n0-3 swaps trained, 4-8 never seen;\n0-3 column = lowest of the four",
            fontsize=8.4, color=INK2, va="top")
    ax.text(0.785, 0.76, "0-3" + " " * 5 + "4" + " " * 6 + "6" + " " * 6 + "8", fontsize=8.6, color=MUTED,
            family="monospace", va="center")
    order = ["Naive", "Align-privileged", "Align-self-predictions", "Hand-crafted-backtracking", "DreamerV3"]
    for i, name in enumerate(order):
        y0 = 0.715 - 0.115 * i
        v = acc[name]
        lo = min(v[idx[s]] for s in range(4))
        vals = f"{lo:.2f}" + "   " + "   ".join(f"{v[idx[s]]:.2f}" for s in (4, 6, 8))
        ax.plot([0.785, 0.80], [y0] * 2, color=COL[name], lw=3, solid_capstyle="round")
        ax.text(0.806, y0, LABEL.get(name, name), fontsize=8.3, color=INK, va="center")
        ax.text(0.785, y0 - 0.037, vals, fontsize=8.6, color=INK, family="monospace", va="center")
    for name, y, h, _ in rows:
        arrow(ax, [(MX + MW, y + h / 2), (0.775, y + h / 2)], color=COL[name])
    out = ROOT / "figures/all3_runs_diagram.png"
    fig.savefig(out, facecolor=SURF, bbox_inches="tight", pad_inches=0.15)
    print(out)


def labels_figure():
    sets = [("Align-self-predictions\n(imagined swaps)", "data/labels_psr_self_all3naive.npz", COL["Align-self-predictions"]),
            ("Hand-crafted-backtracking\n(real swaps)", "data/labels_backtrack_all3naive.npz", COL["Hand-crafted-backtracking"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=170)
    fig.patch.set_facecolor(SURF)
    w = 0.36
    for j, (name, path, c) in enumerate(sets):
        d = np.load(ROOT / path)
        lab, tru, done = d["label"], d["truth"], d["swaps_done"]
        has = lab >= 0
        mp = {k: int(np.bincount(tru[lab == k], minlength=3).argmax()) for k in (0, 1, 2) if (lab == k).any()}
        right = np.array([mp.get(int(l), -9) == int(t) for l, t in zip(lab, tru)])
        acc = [np.mean(right[has & (done == k)]) for k in range(4)]
        cov_ok = np.array([np.mean((right & has)[tru == s]) for s in range(3)])
        cov_bad = np.array([np.mean((~right & has)[tru == s]) for s in range(3)])
        off = (j - 0.5) * w
        xs = np.arange(4)
        axes[0].bar(xs + off, acc, width=w - 0.04, color=c, label=name.replace("\n", " "), zorder=3)
        for x_, v in zip(xs + off, acc):
            axes[0].text(x_, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.6, color=INK)
        xs = np.arange(3)
        axes[1].bar(xs + off, cov_ok, width=w - 0.04, color=c, zorder=3)
        axes[1].bar(xs + off, cov_bad, bottom=cov_ok, width=w - 0.04, color=SURF, edgecolor=c, hatch="////", lw=1.2,
                    zorder=3, label="labelled WRONG" if j == 0 and cov_bad.any() else None)
        for x_, a_, b_ in zip(xs + off, cov_ok, cov_bad):
            axes[1].text(x_, a_ + b_ + 0.015, f"{a_:.2f} right" + (f"\n+{b_:.2f} wrong" if b_ > 0 else ""), ha="center", va="bottom",
                         fontsize=8.2, color=INK)
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels([f"{k}" for k in range(4)])
    axes[0].set_xlabel("swaps already done at the labelled frame", color=INK2, fontsize=10)
    axes[0].set_title("How often a label is right", loc="left", fontsize=12, color=INK, fontweight="bold")
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["slot 0\n(lifted cup)", "slot 1", "slot 2"])
    axes[1].set_xlabel("where the ball really is", color=INK2, fontsize=10)
    axes[1].set_title("How many still frames get the RIGHT label", loc="left", fontsize=12, color=INK, fontweight="bold")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
    for ax in axes:
        ax.set_facecolor(SURF); ax.set_ylim(0, 1.12); style_axes(ax)
    axes[0].set_ylabel("share of labelled frames", color=INK2, fontsize=10)
    axes[1].set_ylabel("share of still frames", color=INK2, fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.text(0.5, 0.005, "Training episodes of memtest4_all3; true slot used only to score the labels.", ha="center",
             fontsize=8.8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    out = ROOT / "figures/all3_labels.png"
    fig.savefig(out, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    data = add_dreamer(read_smt())
    chart(data)
    diagram(data)
    labels_figure()
