"""F1 and accuracy BY NUMBER OF SWAPS, posterior at the lift onset, for runs trained on short sequences.

    pixi run python scripts/onset_depth_table.py "NAME=<stage-1 run dir>" ... [--md out.md]

Reads <run>/onset_probe_EVAL{,5,6,7,8}/per_episode.csv (run onset-probe-phase1 first; missing deep sets are
skipped for every run). EVAL (memtest4_eval) is split by its swap count 0-4; EVAL5..8 are one depth each.
Every row is the same episodes for every run. Score = posterior_r1, the frozen reward decoder's memory-only
read-out from the posterior chained to the onset. No confidence intervals; n = 63 per row.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def f1(y, p):
    tp = (y & p).sum()
    return float(2 * tp / max(2 * tp + (~y & p).sum() + (y & ~p).sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=<stage-1 run dir>")
    ap.add_argument("--trained", default="0-3", help="swap counts seen in training, for the row labels")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    runs = [a.split("=", 1) for a in args.runs]
    sets = ["EVAL"] + [s for s in ("EVAL5", "EVAL6", "EVAL7", "EVAL8")
                       if all((Path(r) / f"onset_probe_{s}" / "per_episode.csv").exists() for _, r in runs)]
    lo, hi = (int(x) for x in args.trained.split("-"))
    data = {}
    for name, run in runs:
        rows = {}
        for sp in sets:
            for r in csv.DictReader((Path(run) / f"onset_probe_{sp}" / "per_episode.csv").open()):
                rows.setdefault(int(r["cup_swaps"]), []).append((r["label"] == "True", r["posterior_r1"] == "True"))
        data[name] = rows
    depths = sorted(data[runs[0][0]])
    out = []
    for metric in ("F1", "accuracy"):
        out.append(f"\n**{metric}**, posterior at the lift onset (63 test episodes per row, the same for every run)\n")
        out.append("| swaps | " + " | ".join(n for n, _ in runs) + " | always \"ball\" |")
        out.append("|---|" + "---|" * (len(runs) + 1))
        for d in depths:
            tag = "trained" if lo <= d <= hi else "**OOD**"
            cells = []
            for name, _ in runs:
                g = data[name][d]
                y, p = np.array([a for a, _ in g]), np.array([b for _, b in g])
                cells.append(f"{f1(y, p):.2f}" if metric == "F1" else f"{np.mean(y == p):.2f}")
            y = np.array([a for a, _ in data[runs[0][0]][d]])
            base = 2 * y.mean() / (1 + y.mean()) if metric == "F1" else y.mean()
            out.append(f"| {d} ({tag}) | " + " | ".join(cells) + f" | {base:.2f} |")
    text = "\n".join(out)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
