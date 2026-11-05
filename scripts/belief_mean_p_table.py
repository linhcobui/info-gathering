"""Average decoded p and pass rate on the closed-start test set, by honest answer (1/3, 1/2, 1, 0) and swap range.

    python3 scripts/belief_mean_p_table.py "NAME=<run dir>" ... [--md out.md]

Reads <run>/belief_eval/per_episode.csv (scripts/belief_eval_phase1.py): p = the frozen decoder's answer at the final
lift onset from the posterior chained over every frame; p* = the exact Bayes answer. pass = |p - p*| <= 0.1.
"""
import csv
import sys
import numpy as np

CATS = [("1/3: nothing seen", (3,)), ("1/2: one of two", (5, 6)), ("1: ball known under the lifted cup", (0,)),
        ("0: ball known elsewhere / empty peek", (1, 2, 4))]
runs = [a.split("=", 1) for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
rows = {n: list(csv.DictReader(open(f"{r}/belief_eval/per_episode.csv"))) for n, r in runs}
out = []
for title, key in (("average decoded p", "mean"), ("pass rate |p - p*| <= 0.1", "pass")):
    out.append(f"\n**{title}** (closed-start test set; 0-3 swaps trained / 4-8 swaps unseen)\n")
    out.append("| honest answer p* | swaps | n | " + " | ".join(n for n, _ in runs) + " |")
    out.append("|---|---|---|" + "---|" * len(runs))
    for cat, cls in CATS:
        for lo, hi, lab in ((0, 3, "0-3"), (4, 8, "4-8")):
            vals, n = [], 0
            for nm, _ in runs:
                sel = [r for r in rows[nm] if int(r["belief_class"]) in cls and lo <= int(r["swaps"]) <= hi]
                n = len(sel)
                p = np.array([float(r["p_posterior"]) for r in sel]); ps = np.array([float(r["p_star"]) for r in sel])
                vals.append(f"{p.mean():.2f}" if key == "mean" else f"{np.mean(np.abs(p - ps) <= 0.1):.2f}")
            out.append(f"| {cat} | {lab} | {n} | " + " | ".join(vals) + " |")
text = "\n".join(out)
print(text)
if md:
    open(md, "w").write(text + "\n")
