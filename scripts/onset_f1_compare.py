"""Posterior-at-onset F1 table for stage-1 runs, from their onset probes (no pixel decoder needed).

    pixi run python scripts/onset_f1_compare.py "NAME=<stage-1 run dir>:<training zarr>:<n trained>" ... [--md out.md]

Reads <run>/onset_probe_EVAL/per_episode.csv (0-4 swaps), <run>/onset_probe_EVAL5/per_episode.csv
(5 swaps) and, when EVERY run has them, onset_probe_EVAL6 / EVAL7 (6 / 7 swaps); run them first with `pixi run onset-probe-phase1 wm_run=<run> outdir=<run>/onset_probe_EVAL{,5}
zarr_path=data/memtest4_eval{,5}.zarr`. The score is `posterior_r1`: the frozen reward decoder's
memory-only read-out (lift offset 1) from the posterior chained to the lift onset.

Rows: seen / unseen / unseen 3 / unseen 4 judged against EACH run's own training set (n differs when
the training stores differ), plus two rows on the SAME episodes for every run: 5 swaps, and the
0-4 episodes unseen by ALL runs. An always-"ball" row gives the F1 a constant guess would get.
Plain F1 score, no confidence intervals; with n < ~60, differences under ~0.1 are usually noise.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from info_gathering.swap_sequences import sequences  # noqa: E402


def f1(y, p):
    tp = (y & p).sum()
    return float(2 * tp / max(2 * tp + (~y & p).sum() + (y & ~p).sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=<stage-1 run dir>:<training zarr>:<number of trained episodes>")
    ap.add_argument("--md", default=None, help="also write the markdown table here")
    args = ap.parse_args()
    deep = {"EVAL5": 5, "EVAL6": 6, "EVAL7": 7}                 # deeper-than-training sets, same episodes for all
    specs = [(n, *r.split(":")) for n, r in (a.split("=", 1) for a in args.runs)]
    sets = ["EVAL", "EVAL5"] + [sp for sp in ("EVAL6", "EVAL7")
                                if all((Path(run) / f"onset_probe_{sp}" / "per_episode.csv").exists() for _, run, _, _ in specs)]
    test = {sp: sequences(ROOT / f"data/memtest4_{sp.lower()}.zarr") for sp in sets}
    runs = []
    for spec in args.runs:
        name, rest = spec.split("=", 1)
        run, trz, ntr = rest.split(":")
        rows = {}
        for sp in sets:
            f = Path(run) / f"onset_probe_{sp}" / "per_episode.csv"
            if not f.exists():
                sys.exit(f"missing {f}: run onset-probe-phase1 first")
            for r in csv.DictReader(f.open()):
                rows[(sp, int(r["episode"]))] = (r["label"] == "True", r["posterior_r1"] == "True")
        runs.append((name, int(ntr), trz, set(sequences(ROOT / trz)[:int(ntr)]), rows))
    keys = sorted(runs[0][4])
    unseen_all = [k for k in keys if k[0] == "EVAL" and all(test["EVAL"][k[1]] not in s for _, _, _, s, _ in runs)]

    def bucket(seen, k):
        seq = test[k[0]][k[1]]
        d = len(seq[1])
        if k[0] in deep:
            return [f"{d} swaps"]
        return ["seen"] if seq in seen else ["unseen", f"unseen {d}"]

    deep_rows = [f"{deep[sp]} swaps" for sp in sets if sp in deep]
    order = ["seen", "unseen", "unseen 3", "unseen 4", "unseen by all"] + deep_rows
    label = {"seen": "**seen**", "unseen": "**unseen**", "unseen 3": "↳ unseen 3", "unseen 4": "↳ unseen 4",
             "unseen by all": f"**unseen by all** (same {len(unseen_all)})"} | {
             r: f"**{r}** (same 63 for all)" for r in deep_rows}
    cells = {b: [] for b in order}
    for name, ntr, trz, seen, rows in runs:
        groups = {b: [] for b in order}
        for k in keys:
            for b in bucket(seen, k):
                if b in groups:                      # unseen 0-2 still count in the pooled "unseen" row
                    groups[b].append(rows[k])
        for k in unseen_all:
            groups["unseen by all"].append(rows[k])
        for b in order:
            g = groups[b]
            if not g:
                cells[b].append("–")
                continue
            y, p = np.array([a for a, _ in g]), np.array([c for _, c in g])
            same = b in deep_rows or b == "unseen by all"
            cells[b].append(f"{f1(y, p):.2f}" + ("" if same else f" (n={len(y)})"))
    base = {}
    for b in ["unseen by all"] + deep_rows:
        ks = unseen_all if b == "unseen by all" else [k for k in keys if k[0] in deep and deep[k[0]] == int(b.split()[0])]
        y = np.array([runs[0][4][k][0] for k in ks])
        base[b] = 2 * y.mean() / (1 + y.mean()) if len(y) else float("nan")
    lines = ["| | " + " | ".join(n for n, *_ in runs) + " |", "|---|" + "---|" * len(runs),
             "| trained on | " + " | ".join(f"{ntr} episodes ({Path(trz).stem})" for _, ntr, trz, _, _ in runs) + " |"]
    lines += [f"| {label[b]} | " + " | ".join(cells[b]) + " |" for b in order]
    lines.append(f"| always \"ball\" ({' / '.join(base)}) | " + " / ".join(f"{v:.2f}" for v in base.values()) + " |"
                 + " |" * (len(runs) - 1))
    text = "F1 score, posterior at the lift onset (memory-only read-out). n = test episodes.\n\n" + "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
