"""By-depth accuracy / F1 of a pixel world model's IMAGINED lift (eval_memtest1.py with eval_memtest4.yaml).

    pixi run python scripts/lift_eval_depth_table.py <decoder run> [--md out.md]

Reads <decoder run>/eval_EVAL{,5,6,7,8}/per_episode.csv. The model watches the real frames up to the lift onset,
imagines the lift from the actions alone, the decoder renders it, and a colour detector looks for the ball
(`imagined_ball`); `recon_ball` (decoded posterior of the real lift frames) is the detector/decoder sanity check.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from info_gathering.swap_sequences import sequences  # noqa: E402


def f1(y, p):
    tp = (y & p).sum()
    return float(2 * tp / max(2 * tp + (~y & p).sum() + (y & ~p).sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decoder_run")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    rows = {}
    for sp in ("", "5", "6", "7", "8"):
        f = Path(args.decoder_run) / f"eval_EVAL{sp}" / "per_episode.csv"
        if not f.exists():
            continue
        # the csv's `swaps` column counts only swaps that moved the ball; group by the episode's real swap count
        depth = [len(q[1]) for q in sequences(ROOT / f"data/memtest4_eval{sp}.zarr")]
        for r in csv.DictReader(f.open()):
            if r["segment"] == "lift":
                rows.setdefault(depth[int(r["episode"])], []).append(
                    (r["label"] == "True", r["imagined_ball"] == "True", r["recon_ball"] == "True"))
    out = ["| swaps | n | imagined: accuracy | imagined: F1 | decoded real lift (sanity): accuracy |", "|---|---|---|---|---|"]
    for d in sorted(rows):
        y, p, rc = (np.array([g[i] for g in rows[d]]) for i in range(3))
        out.append(f"| {d} | {len(y)} | {np.mean(y == p):.2f} | {f1(y, p):.2f} | {np.mean(y == rc):.2f} |")
    text = "\n".join(out)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
