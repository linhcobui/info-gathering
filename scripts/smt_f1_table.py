"""Imagined-lift F1 for SMT runs, split the standard way: seen / unseen swap sequences and 5 swaps.

    pixi run python scripts/smt_f1_table.py NAME=<pixel decoder run dir> [NAME=<dir> ...] [--md out.md]

Unseen is split into unseen 3 and unseen 4 swaps: a pooled "unseen" hid that every unaligned run
is at chance on unseen 4 (the pooled number was carried by ~10 easy 3-swap episodes). Each
training store gets an always-"ball" row to read the buckets against.

For each pixel-decoder run (smt_prelim or smt_phase1) it
  * finds its pixel evals: the newest per_episode.csv under the run whose metrics.json says
    memtest4_eval.zarr (0-4 swaps) and memtest4_eval5.zarr (5 swaps). Run them first with
    eval-smt-phase1 / eval-smt-prelim (`zarr_path=data/memtest4_eval5.zarr outdir=<run>/eval_EVAL5`);
  * reads the run's TRAINING store and anchor holdout from the world-model config stored in the
    decoder checkpoint, and calls a test episode "seen" iff its exact swap sequence (start slot +
    slot pair of each swap, info_gathering.swap_sequences) occurs in the trained episodes;
  * scores only segment == "lift" (the reveal rows are a control, F1 0.5 even on real frames).

F1 = harmonic mean of precision and recall on "ball under the lifted cup". Always-"ball" scores
~0.52 on the unseen/5-swap buckets, always-"empty" scores 0. CIs: 95% bootstrap over episodes.
"""
import argparse
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from info_gathering.swap_sequences import sequences  # noqa: E402

EVAL, EVAL5 = "memtest4_eval.zarr", "memtest4_eval5.zarr"


def f1(y, p):
    tp = (y & p).sum()
    return float(2 * tp / max(2 * tp + (~y & p).sum() + (y & ~p).sum(), 1))


def ci(y, p, n=2000):
    rng = np.random.default_rng(0)
    s = [f1(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(s, [2.5, 97.5])


def find_evals(run):
    """{eval zarr name: newest lift rows} for every pixel eval under the run."""
    found, own = {}, (Path(run) / "checkpoints" / "decoder_latest.pt").resolve()
    for m in Path(run).rglob("metrics.json"):
        try:
            meta = json.loads(m.read_text())
        except json.JSONDecodeError:
            continue
        # only evals of THIS run: finetune exports etc. nest their own evals inside a run dir
        if Path(meta.get("decoder_checkpoint", "")).resolve() != own:
            continue
        name = Path(meta.get("eval_zarr", "")).name
        csv_path = m.parent / "per_episode.csv"
        if name in (EVAL, EVAL5) and csv_path.exists():
            if name not in found or m.stat().st_mtime > found[name][0]:
                found[name] = (m.stat().st_mtime, csv_path)
    return {k: [r for r in csv.DictReader(v[1].open()) if r["segment"] == "lift"] for k, v in found.items()}


@lru_cache(None)
def trained_sequences(zarr_path, holdout):
    seqs = sequences(ROOT / zarr_path)
    return frozenset(seqs[:len(seqs) - holdout] if holdout else seqs)


def train_info(run):
    ck = Path(run) / "checkpoints" / "decoder_latest.pt"
    state = torch.load(ck, map_location="cpu", weights_only=False)
    wm = state["config"]["world_model"]
    anchors = wm.get("data_anchors") or wm["data"]
    return anchors["zarr_path"], int(anchors.get("holdout_episodes", 0)), wm["model"]


def score(run):
    zarr_path, holdout, model = train_info(run)
    seen_set = trained_sequences(zarr_path, holdout)
    evals = find_evals(run)
    out = {"train": f"{Path(zarr_path).name} ({len(sequences(ROOT / zarr_path)) - holdout} eps)",
           "warmup": model.get("eval_warmup", "-"), "ctx": model.get("ctx_len", "-"), "always_ball": {}}
    rows = []
    for name, lift in evals.items():
        test = sequences(ROOT / "data" / name)
        for r in lift:
            seq = test[int(r["episode"])]
            rows.append((len(seq[1]), seq in seen_set, r["label"] == "True", r["imagined_ball"] == "True"))
    y = np.array([r[2] for r in rows], bool)
    p = np.array([r[3] for r in rows], bool)
    d = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows], bool)
    buckets = {"all 0-4": d <= 4, "seen": (d <= 4) & s, "unseen": (d <= 4) & ~s,
               "unseen 3": (d == 3) & ~s, "unseen 4": (d == 4) & ~s, "5 swaps": d == 5}
    for k in range(5):
        buckets[f"{k} sw"] = d == k
    for key, m in buckets.items():
        if m.sum():
            lo, hi = ci(y[m], p[m])
            out[key] = (f1(y[m], p[m]), int(m.sum()), lo, hi)
            out["always_ball"][key] = 2 * y[m].mean() / (1 + y[m].mean())     # F1 of always answering "ball"
    out["missing"] = [n for n in (EVAL, EVAL5) if n not in evals]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=<pixel decoder run dir>")
    ap.add_argument("--md", help="also write the table as markdown here")
    args = ap.parse_args()
    cols = ["seen", "unseen 3", "unseen 4", "5 swaps", "0 sw", "1 sw", "2 sw", "3 sw", "4 sw"]
    lines = ["| run | trained on | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    scored = [(spec.split("=", 1)[0], score(spec.split("=", 1)[1])) for spec in args.runs]
    # "unseen" depends on each run's training set, so the always-"ball" row is per training store
    for store in dict.fromkeys(r["train"] for _, r in scored):
        ref = next(r for _, r in scored if r["train"] == store)
        lines.append(f"| *always say \"ball\"* | {store} | " + " | ".join(
            f"*{ref['always_ball'][c]:.2f}*" if c in ref["always_ball"] else "–" for c in cols) + " |")
    for name, r in scored:
        cells = []
        for c in cols:
            if c in r:
                v, n, lo, hi = r[c]
                cells.append(f"{v:.2f} ({n})" + (f" [{lo:.2f}–{hi:.2f}]" if c in ("unseen 3", "unseen 4", "5 swaps") else ""))
            else:
                cells.append("–")
        note = f" ⚠ no eval for {', '.join(r['missing'])}" if r["missing"] else ""
        lines.append(f"| {name}{note} | {r['train']}, warmup {r['warmup']} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
