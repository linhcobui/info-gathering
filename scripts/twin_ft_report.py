"""TWIN_FINETUNE results: every arm on the same test episodes, split by depth and by seen/unseen.

Re-run at any time; arms that have not finished are listed as pending. Writes
    docs/TWIN_FINETUNE_RESULTS.md
    figures/twin_ft_seen_unseen.png

"Seen" = the episode's exact swap sequence (start slot + slot pair of each swap) occurs in S1's
training episodes OR in the finetune pool -- the same buckets for every arm, so every row compares
the same episodes. At depth 4 nothing is in S1, so "seen" there means "in the finetune pool".

Two read-outs per arm, both on the posterior chain warmed up once at frame 0:
    pixel   the frozen decoder's imagined lift, yellow-pixel detector (eval-smt-phase1). PRIMARY:
            it is what the finetune optimises, and what every earlier pixel table reports.
    probe   the frozen stage-1 reward head read at the lift onset (onset-probe-phase1,
            posterior_r1). For finetuned arms the head never saw the new states -- read it as a
            check on whether the representation changed, not just the decoder's output.

    pixi run twin-ft-report
"""
import csv
import math
import os
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from info_gathering.swap_sequences import sequences  # noqa: E402

BASE = ROOT / "logdir/world_model/smt_phase1_twinft"
C0 = BASE / "s3_noobs_phase1"
C1 = BASE / "s3ft_noobs_phase1"
PD = "readout/main/pixel_decoder_phase1/main"
OUT_MD = ROOT / "docs/TWIN_FINETUNE_RESULTS.md"
OUT_FIG = ROOT / "figures/twin_ft_seen_unseen.png"
NOTES_START, NOTES_END = "<!-- notes:start -->", "<!-- notes:end -->"

# (key, label, pixel-eval dir, onset-probe dir (wm_run), finetune logdir or None, what it is)
ARMS = [
    ("C0", "C0 · stage 1 on 0–3", C0 / PD, C0, None, "no finetune (reference)"),
    ("C1", "C1 · retrain on S1+FT", C1 / PD, C1, None, "stage 1 from scratch on S1 + the finetune pool, WITH labels"),
    ("F1", "F1 · reveal", C0 / "twin_ft/f1/export/pixel_decoder", C0 / "twin_ft/f1/export",
     C0 / "twin_ft/f1", "finetune, L_reveal only"),
    ("F2", "F2 · reveal+recon", C0 / "twin_ft/f2/export/pixel_decoder", C0 / "twin_ft/f2/export",
     C0 / "twin_ft/f2", "finetune, L_reveal + L_recon"),
    ("F3", "F3 · reveal+recon+twin", C0 / "twin_ft/f3/export/pixel_decoder", C0 / "twin_ft/f3/export",
     C0 / "twin_ft/f3", "finetune, L_reveal + L_recon + L_twin (the full method)"),
]
TWINS = [("F3-twin", "F3 · EMA twin", C0 / "twin_ft/f3/export_twin/pixel_decoder"),
         ("F2-twin", "F2 · EMA twin", C0 / "twin_ft/f2/export_twin/pixel_decoder"),
         ("F1-twin", "F1 · EMA twin", C0 / "twin_ft/f1/export_twin/pixel_decoder")]

# buckets, in display order: (key, label, depth, seen)
BUCKETS = [("d0-2", "0–2 swaps (all seen)", (0, 1, 2), None),
           ("d3s", "3 swaps · seen", (3,), True),
           ("d4s", "4 swaps · seen in FT", (4,), True),
           ("d4u", "4 swaps · UNSEEN", (4,), False),
           ("d5", "5 swaps (all unseen)", (5,), None)]


def f1_acc(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
    return (2 * tp / max(2 * tp + fp + fn, 1)), float((y == p).mean()) if len(y) else float("nan")


def boot(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    s = [f1_acc(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    f, a = np.array(s).T
    return np.percentile(f, [2.5, 97.5]), np.percentile(a, [2.5, 97.5])


def read_rows(path, segment_col=True):
    f = Path(path) / "per_episode.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(f.open()))
    return [r for r in rows if r.get("segment", "lift") == "lift"] if segment_col else rows


def main():
    s1 = sequences(ROOT / "data/memtest4_s3.zarr")[:-25]          # anchors' 25-episode holdout excluded
    ft = sequences(ROOT / "data/memtest4_ft.zarr")
    seen = set(s1) | set(ft)
    split = json.loads((ROOT / "data/memtest4_ft_split.json").read_text())
    held = {(h[0], tuple(tuple(p) for p in h[1])) for h in split["held_out_4swap"]}
    test = {"EVAL": sequences(ROOT / "data/memtest4_eval.zarr"), "EVAL5": sequences(ROOT / "data/memtest4_eval5.zarr")}

    def bucket_of(seq):
        d, s = len(seq[1]), seq in seen
        for key, _, depths, want in BUCKETS:
            if d in depths and (want is None or want == s):
                return key
        return None

    # per arm: list of (bucket, label, pixel_pred, probe_pred, in_H)
    results, status = {}, {}
    for key, label, pix_dir, wm_dir, ft_dir, _ in ARMS:
        rows, missing = [], []
        for split_name in ("EVAL", "EVAL5"):
            pix = read_rows(Path(pix_dir) / f"eval_{split_name}")
            probe = read_rows(Path(wm_dir) / f"onset_probe_{split_name}", segment_col=False)
            if probe is None:
                missing.append(f"probe {split_name}")
            if pix is None:
                missing.append(f"pixel {split_name}")
                continue
            probe_by = {int(r["episode"]): r["posterior_r1"] == "True" for r in probe} if probe else {}
            for r in pix:
                e = int(r["episode"])
                seq = test[split_name][e]
                rows.append({"bucket": bucket_of(seq), "label": r["label"] == "True",
                             "pixel": r["imagined_ball"] == "True", "probe": probe_by.get(e),
                             "held": seq in held, "depth": len(seq[1]), "ep": (split_name, e)})
        results[key] = rows
        status[key] = "done" if not missing else ("pending" if len(missing) == 4 else "partial: missing " + ", ".join(missing))
    twin_results = {}
    for key, label, pix_dir in TWINS:
        rows = []
        for split_name in ("EVAL", "EVAL5"):
            pix = read_rows(Path(pix_dir) / f"eval_{split_name}")
            for r in pix or []:
                seq = test[split_name][int(r["episode"])]
                rows.append({"bucket": bucket_of(seq), "label": r["label"] == "True",
                             "pixel": r["imagined_ball"] == "True"})
        twin_results[key] = rows

    def cell(rows, bucket, source, ci=False):
        sel = [r for r in rows if r["bucket"] == bucket and r.get(source) is not None]
        if not sel:
            return "–", None
        y, p = [r["label"] for r in sel], [r[source] for r in sel]
        f, a = f1_acc(y, p)
        txt = f"{f:.2f} / {a:.2f}"
        if ci:
            (flo, fhi), _ = boot(y, p)
            txt += f" [{flo:.2f}–{fhi:.2f}]"
        return txt, (f, a, len(sel))

    md = ["# TWIN-FINETUNE — results", "",
          "Auto-generated by `scripts/twin_ft_report.py` (re-run: `pixi run twin-ft-report`). "
          "Plan: [TWIN_FINETUNE.md](TWIN_FINETUNE.md).", "",
          "**Set-up.** Stage 1 (unaligned baseline, no-obs decoder) trained on **0–3 swaps** "
          "(`data/memtest4_s3.zarr`, 378 episodes). Its readout + pixel decoder are frozen. The finetune "
          "pool (`data/memtest4_ft.zarr`, 598 episodes, 0–4 swaps, fresh seed) is used **without labels**: "
          "the only training signal is the frozen decoder's imagined lift against the real frames' pixels. "
          "It contains 4-swap episodes from only **half** of the 243 possible 4-swap sequences; the other half "
          "is held out. Test = `memtest4_eval` (0–4, 315 episodes) + `memtest4_eval5` (5 swaps, 63), untouched.", ""]
    # a hand-written reading, kept verbatim across re-runs (edit it between the two markers)
    notes = ""
    if OUT_MD.exists():
        text = OUT_MD.read_text()
        if NOTES_START in text and NOTES_END in text:
            notes = text[text.index(NOTES_START):text.index(NOTES_END) + len(NOTES_END)]
    if notes:
        md += [notes, ""]
    md += ["## Arms", "", "| arm | what | status |", "|---|---|---|"]
    for key, label, _, _, _, what in ARMS:
        md.append(f"| {key} | {what} | {status[key]} |")
    md += ["", "## Headline: imagined-lift F1 / accuracy by kind of episode (pixel eval)", "",
           "Each cell: **F1 / accuracy** on the same episodes for every arm. n and positive rate in the header. "
           "Brackets: 95% bootstrap CI of F1, on the two buckets that decide the question.", ""]
    ref = next((results[k] for k in results if results[k]), [])
    header = "| arm |"
    for b, lab, _, _ in BUCKETS:
        sel = [r for r in ref if r["bucket"] == b]
        rate = np.mean([r["label"] for r in sel]) if sel else float("nan")
        header += f" {lab}<br>n={len(sel)}, pos {rate:.2f} |"
    md += [header, "|---|" + "---|" * len(BUCKETS)]
    table = {}
    for key, label, *_ in ARMS:
        line = f"| {key} |"
        for b, *_ in BUCKETS:
            txt, val = cell(results[key], b, "pixel", ci=b in ("d4u", "d5"))
            table[(key, b, "pixel")] = val
            line += f" {txt} |"
        md.append(line)
    if any(twin_results.values()):
        md += ["", "EMA twins (the slow copy, exported separately; pixel eval only):", "",
               "| twin |" + "".join(f" {lab} |" for _, lab, _, _ in BUCKETS), "|---|" + "---|" * len(BUCKETS)]
        for key, label, _ in TWINS:
            if twin_results[key]:
                md.append(f"| {key} |" + "".join(f" {cell(twin_results[key], b, 'pixel')[0]} |" for b, *_ in BUCKETS))

    md += ["", "## Accuracy can mislead here: balance, bias, and a paired test against C0 (pixel eval)", "",
           "About a third of reveals show a ball, so answering \"empty\" every time already scores 0.65 on the "
           "hard buckets. **balanced** = mean of recall on ball episodes and recall on empty episodes (0.50 = "
           "chance). **says ball** = how often the arm predicts a ball (true rate in the header). **vs C0** = on the "
           "same episodes, how many C0 got wrong that this arm gets right (fixed) and the reverse (broke), with an "
           "exact McNemar p-value.", "",
           "| arm | bucket | acc | balanced | says ball | fixed / broke vs C0 | p |", "|---|---|---|---|---|---|---|"]
    for b_keys, b_lab in ((("d4s",), "4 seen in FT"), (("d4u",), "4 UNSEEN"), (("d5",), "5"),
                          (("d4u", "d5"), "4 unseen + 5")):
        base = {r["ep"]: r for r in results["C0"] if r["bucket"] in b_keys}
        if not base:
            continue
        for key, *_ in ARMS:
            sel = sorted((r for r in results[key] if r["bucket"] in b_keys), key=lambda r: r["ep"])
            if [r["ep"] for r in sel] != sorted(base):
                continue
            c0_rows = [base[r["ep"]] for r in sel]
            y = np.array([r["label"] for r in sel]); pr = np.array([r["pixel"] for r in sel])
            bal = 0.5 * ((pr & y).sum() / max(y.sum(), 1) + (~pr & ~y).sum() / max((~y).sum(), 1))
            row = f"| {key} | {b_lab} (pos {y.mean():.2f}) | {(pr == y).mean():.2f} | {bal:.2f} | {pr.mean():.2f} |"
            if key == "C0":
                md.append(row + " – | – |")
                continue
            c_arm = pr == y                                   # paired by episode
            c_0 = np.array([r["pixel"] for r in c0_rows]) == np.array([r["label"] for r in c0_rows])
            fixed, broke = int((c_arm & ~c_0).sum()), int((~c_arm & c_0).sum())
            n = fixed + broke
            pval = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(fixed, broke) + 1)) / 2 ** n) if n else 1.0
            md.append(row + f" {fixed} / {broke} | {pval:.3f} |")

    md += ["", "## Onset probe (frozen stage-1 reward head on the posterior at the lift onset)", "",
           "Same cells, F1 / accuracy. The reward head was never trained on finetuned states, so a gap "
           "between this and the pixel table means the decoder's output changed more than the state did.", "",
           "| arm |" + "".join(f" {lab} |" for _, lab, _, _ in BUCKETS), "|---|" + "---|" * len(BUCKETS)]
    for key, *_ in ARMS:
        line = f"| {key} |"
        for b, *_ in BUCKETS:
            txt, val = cell(results[key], b, "probe")
            table[(key, b, "probe")] = val
            line += f" {txt} |"
        md.append(line)

    # decoder-fooling check: does the frozen decoder's call agree with the frozen reward head's?
    md += ["", "## Decoder-fooling check", "",
           "Fraction of test episodes where the pixel verdict (frozen decoder) and the onset-probe verdict "
           "(frozen reward head) agree, and each one's accuracy on the hard episodes (4 unseen + 5). If "
           "finetuning only taught the updater to make the frozen decoder draw a ball, agreement falls and "
           "pixel accuracy rises while probe accuracy does not.", "",
           "| arm | agreement, all | agreement, 4 unseen + 5 | pixel acc, 4 unseen + 5 | probe acc, 4 unseen + 5 |",
           "|---|---|---|---|---|"]
    for key, *_ in ARMS:
        rows = [r for r in results[key] if r["probe"] is not None]
        if not rows:
            md.append(f"| {key} | – | – | – | – |")
            continue
        hard = [r for r in rows if r["bucket"] in ("d4u", "d5")]
        ag = lambda rs: np.mean([r["pixel"] == r["probe"] for r in rs]) if rs else float("nan")
        acc = lambda rs, s: np.mean([r[s] == r["label"] for r in rs]) if rs else float("nan")
        md.append(f"| {key} | {ag(rows):.2f} | {ag(hard):.2f} | {acc(hard, 'pixel'):.2f} | {acc(hard, 'probe'):.2f} |")

    # held-out half vs merely-not-drawn among the unseen 4-swap episodes
    md += ["", "## 4-swap unseen, split by why it is unseen", "",
           "`held out` = its sequence is in the half that was removed from the finetune pool; `not drawn` = allowed "
           "but no finetune episode happened to use it. Pixel F1 / accuracy.", "",
           "| arm | held out | not drawn |", "|---|---|---|"]
    for key, *_ in ARMS:
        rows = [r for r in results[key] if r["bucket"] == "d4u"]
        parts = []
        for h in (True, False):
            sel = [r for r in rows if r["held"] == h]
            if sel:
                f, a = f1_acc([r["label"] for r in sel], [r["pixel"] for r in sel])
                parts.append(f"{f:.2f} / {a:.2f} (n={len(sel)})")
            else:
                parts.append("–")
        md.append(f"| {key} | {parts[0]} | {parts[1]} |")

    # finetune training curves
    md += ["", "## Finetune training (from metrics.jsonl, batch = 16 finetune episodes)", "",
           "| arm | steps | reveal BCE first → last | batch acc first → last | recon first → last | twin dist last | sec/step |",
           "|---|---|---|---|---|---|---|"]
    for key, _, _, _, ft_dir, _ in ARMS:
        if ft_dir is None:
            continue
        f = Path(ft_dir) / "metrics.jsonl"
        if not f.exists():
            md.append(f"| {key} | pending | | | | | |")
            continue
        m = [json.loads(line) for line in f.open()]
        first, last = m[0], m[-1]
        tail = m[-5:]
        mean = lambda k: np.mean([x[k] for x in tail if k in x])
        md.append(f"| {key} | {last['step']} | {first['loss/reveal']:.3f} → {mean('loss/reveal'):.3f} | "
                  f"{first['batch/acc']:.2f} → {mean('batch/acc'):.2f} | {first['loss/recon']:.4f} → "
                  f"{mean('loss/recon'):.4f} | {last['twin/dist']:.4f} | {mean('sec_per_step'):.2f} |")

    # predictions and kill criteria, evaluated where the numbers exist
    md += ["", "## Predictions and kill criteria (docs/TWIN_FINETUNE.md §8), checked", ""]

    def get(key, b, src="pixel"):
        v = table.get((key, b, src))
        return None if v is None else v[1]          # accuracy

    def pooled(key, buckets, src="pixel"):
        sel = [r for r in results[key] if r["bucket"] in buckets and r.get(src) is not None]
        if not sel:
            return None
        return f1_acc([r["label"] for r in sel], [r[src] for r in sel])

    lines = []
    for arm in ("F1", "F2", "F3"):
        a, c0, c1 = pooled(arm, ("d4u", "d5")), pooled("C0", ("d4u", "d5")), pooled("C1", ("d4u", "d5"))
        if a is None:
            lines.append(f"- {arm}: pending.")
            continue
        txt = f"- **{arm}** on 4-unseen + 5 (pooled, n={sum(r['bucket'] in ('d4u', 'd5') for r in results[arm])}): F1 {a[0]:.3f}, acc {a[1]:.3f}"
        if c0:
            txt += f" | vs C0 {c0[0]:.3f} / {c0[1]:.3f} ({a[1] - c0[1]:+.3f} acc)"
        if c1:
            txt += f" | vs C1 {c1[0]:.3f} / {c1[1]:.3f} ({a[1] - c1[1]:+.3f} acc)"
        lines.append(txt)
        s, u = get(arm, "d4s"), get(arm, "d4u")
        s0, u0 = get("C0", "d4s"), get("C0", "d4u")
        if None not in (s, u, s0, u0):
            lines.append(f"  - 4 swaps, gain over C0: seen-in-FT {s - s0:+.3f} acc, unseen {u - u0:+.3f} acc "
                         f"({'gain is mostly on SEEN sequences → memorising the pool' if (s - s0) > 2 * max(u - u0, 0.0) + 0.05 else 'gain transfers to unseen sequences'})")
    lines += ["", "Kill criteria: **F3 ≈ C1** → it's just data. **F3 improves the pool but not unseen test "
              "sequences** → memorising. **Decoder-fooling check fails** → the frozen-decoder anchor isn't trustworthy."]
    md += lines

    md += ["", "## Where everything is", "", "| arm | pixel eval | onset probe |", "|---|---|---|"]
    for key, _, pix, wm, _, _ in ARMS:
        md.append(f"| {key} | `{os.path.relpath(pix, ROOT)}/eval_EVAL{{,5}}` | `{os.path.relpath(wm, ROOT)}/onset_probe_EVAL{{,5}}` |")
    md += ["", f"Chain logs: `logdir/twin_ft/chain_*.log`. Figure: `{os.path.relpath(OUT_FIG, ROOT)}`.", ""]
    OUT_MD.write_text("\n".join(md))
    print("\n".join(md))
    plot(results, ref)


def plot(results, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5df"
    colours = {"C0": "#898781", "C1": "#1baf7a", "F1": "#eb6834", "F2": "#a35dd6", "F3": "#2a78d6"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=170, gridspec_kw={"wspace": 0.18})
    fig.patch.set_facecolor(SURF)
    rng = np.random.default_rng(0)
    for ax, src, title in ((axes[0], "pixel", "Imagined lift (frozen decoder) — accuracy"),
                           (axes[1], "probe", "Onset probe (frozen reward head) — accuracy")):
        ax.set_facecolor(SURF)
        ax.set_ylim(0.3, 1.04)
        ax.grid(axis="y", color=GRID, lw=1, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(length=0, labelcolor=INK2, labelsize=9.5)
        ax.axvspan(2.5, 4.5, color=INK, alpha=0.035, lw=0, zorder=0)
        for i, (b, *_r) in enumerate(BUCKETS):         # what answering "empty" every time would score
            sel = [r for r in ref if r["bucket"] == b]
            if sel:
                base = 1.0 - np.mean([r["label"] for r in sel])
                ax.plot([i - 0.32, i + 0.32], [base, base], color=INK, lw=1.4, ls=(0, (3, 2)), alpha=0.55, zorder=1)
        ax.plot([], [], color=INK, lw=1.4, ls=(0, (3, 2)), alpha=0.55, label='always say "empty"')
        ax.text(3.5, 0.32, "sequence never trained\nor finetuned on", ha="center", fontsize=8.5, color=MUTED, linespacing=1.2)
        offs = np.linspace(-0.2, 0.2, len(colours))
        for (key, col), dx in zip(colours.items(), offs):
            rows = results.get(key) or []
            xs, ys = [], []
            for i, (b, *_rest) in enumerate(BUCKETS):
                sel = [r for r in rows if r["bucket"] == b and r.get(src) is not None]
                if not sel:
                    continue
                y = np.array([r["label"] for r in sel]); p = np.array([r[src] for r in sel])
                acc = (y == p).mean()
                bs = [(y[j] == p[j]).mean() for j in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                x = i + dx
                ax.plot([x, x], [lo, hi], color=col, lw=1.3, alpha=0.45, zorder=2)
                ax.scatter(x, acc, s=46, color=col, zorder=4, edgecolor="white", lw=0.8)
                xs.append(x); ys.append(acc)
            if xs:
                ax.plot(xs, ys, color=col, lw=1.8, alpha=0.9, zorder=3,
                        label=next(l for k, l, *_ in ARMS if k == key))
        short = {"d0-2": "0–2 swaps\nall seen", "d3s": "3 swaps\nseen", "d4s": "4 swaps\nseen in FT",
                 "d4u": "4 swaps\nUNSEEN", "d5": "5 swaps\nall unseen"}
        ticks = []
        for b, lab, *_ in BUCKETS:
            sel = [r for r in ref if r["bucket"] == b]
            ticks.append(f"{short[b]}\nn={len(sel)}")
        ax.set_xticks(range(len(BUCKETS)))
        ax.set_xticklabels(ticks, fontsize=8.8, color=INK2)
        ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=10)
    axes[0].set_ylabel("accuracy (whisker: 95% bootstrap CI)", fontsize=10.5, color=INK2)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Stage 1 on 0–3 swaps, label-free finetune on 0–4 (half the 4-swap sequences held out)",
                 x=0.07, ha="left", fontsize=13.5, color=INK, y=0.99)
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG, facecolor=SURF, bbox_inches="tight")
    print(f"-> {OUT_FIG}")


if __name__ == "__main__":
    main()
