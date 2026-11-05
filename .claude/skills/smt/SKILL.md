---
name: smt
description: Score and compare SMT world-model runs (smt_prelim / smt_phase1) on memtest4 — F1 split by seen vs unseen swap sequences and 5 swaps (pixel eval or onset probe), the Naive / Align-self / Align-privileged comparison table and t-SNE, plus where runs live and how to evaluate them. Use when asked for F1 / results / a comparison table / t-SNE for any SMT run, or to evaluate a new one.
---

# SMT runs on memtest4: how to get the numbers

## The standard table (one command)

```bash
pixi run python scripts/smt_f1_table.py "NAME=<pixel decoder run dir>" ["NAME2=<dir>" ...] [--md out.md]
```

For each run it prints imagined-lift **F1** on **seen**, **unseen 3**, **unseen 4** and **5 swaps**,
plus F1 by swap count 0–4, and an always-"ball" row per training store. 95% bootstrap CIs are given
for the unseen and 5-swap buckets. Never quote a pooled "unseen": the 500-episode store's pooled
0.59 was ten easy 3-swap episodes, while unseen 4 was at chance.
- **Seen** means the test episode's exact swap sequence occurs in the run's training episodes.
  The sequence is the start slot plus the slot pair of each swap. The training store and its
  25-episode anchor holdout are read from the decoder checkpoint, and slack is ignored.
  Implementation: `src/info_gathering/swap_sequences.py`. It replays the generator from each episode's seed.
- **F1** is on "ball under the lifted cup", from the yellow-pixel detector on decoded imagined frames.
  Only `segment == "lift"` rows count; the reveal rows score 0.5 even on real frames.
  Baselines: always "empty" = 0; always "ball" ≈ 0.52 on unseen and 5-swap.
- Say **F1 score**, not just "F1": the finetune arms are also named F1/F2/F3.

Pass a **pixel decoder** run dir, not the stage-1 dir:
- phase 1: `<stage1>/readout/<ts>/pixel_decoder_phase1/<ts>`
- prelim: `<stage1>/pixel_decoder_prelim/<ts>`

If the table shows `⚠ no eval for memtest4_eval5.zarr`, run the 5-swap eval first (next section).

## Onset-probe F1 comparison (no pixel decoder needed)

For stage-1 runs without a pixel decoder (WINDOW60 and later), compare on the posterior at the lift
onset:

```bash
pixi run python scripts/onset_f1_compare.py "NAME=<stage-1 run dir>:<training zarr>:<n trained>" ... [--md out.md]
```

- **Score:** `posterior_r1` from the onset probes (`<run>/onset_probe_EVAL{,5}/per_episode.csv`; run
  `onset-probe-phase1` first). That's the frozen reward decoder's memory-only read-out (lift offset 1)
  from the posterior chained to the onset.
- **Rows, judged per run:** seen / unseen / unseen 3 / unseen 4, each against **that run's own**
  training set, so n differs between stores.
- **Rows on the same episodes for every run:** 5 swaps, and the 0–4 episodes unseen by all runs.
  6 and 7 swaps are added automatically when every run has `onset_probe_EVAL6` / `EVAL7`
  (`data/memtest4_eval6.zarr`, `memtest4_eval7.zarr`: 63 episodes each, configs
  `configs/data/memtest4_eval{6,7}.yaml`, embeddings cached).
- **Runs worth including:** Align-self-predictions = `$M/window60_alignselfpred_phase1`
  (`data/memtest4.zarr:225`); Align-privileged (1 swap) =
  `logdir/world_model/smt_phase1_memtest4_depth1/2026-09-24_14-22-12_memtest4_depth1_alignlift_phase1`
  (`data/memtest4_depth1.zarr:500`), which is 100% correct on 0–7 swaps.
- **Baseline:** an always-"ball" row.
- **Format:** plain F1 score, no confidence intervals (the user asked for it that way).
- **Naming:** use the user's names. Naive = WINDOW60; Align-self = W60-ALIGNSELF; Align-privileged =
  ALIGN-LIFT.

```bash
M=logdir/world_model/smt_phase1_memtest4; B=logdir/world_model/smt_phase1_memtest4_big
pixi run python scripts/onset_f1_compare.py "Naive=$M/window60_phase1:data/memtest4.zarr:225" \
  "Align-self=$M/window60_alignself_phase1:data/memtest4.zarr:225" \
  "Align-privileged=$B/2026-09-22_21-12-04_memtest4_big_alignlift_phase1:data/memtest4_big.zarr:475"
```

## t-SNE of rest states (ball slot / swap count / start slot)

```bash
pixi run python scripts/rest_states_extract.py --out <npz> --device cuda:N "NAME=<run dir>:<training zarr>:<n trained>" ...
python3 scripts/rest_states_tsne.py <npz> figures/<name>.png "NAME=subtitle" ...       # system python3 (sklearn)
```

- **What's plotted:** posterior states at motionless hold frames of `memtest4_eval`, one column per
  run.
- **Rows:** coloured by the ball's slot now, the episode's swap count, and the start slot. × marks a
  sequence that run never trained on.
- **Neighbour scores:** each panel prints the share of each state's 10 nearest states from OTHER
  episodes that share the colour. Higher is better for slot; lower is better for swap count and start
  slot, which measure history.
- **Renaming:** column names come from `names` in the npz. Rename by rewriting that array, without
  recomputing.

## Mid-hold probe (does the state track the ball? does it keep history?)

```bash
pixi run midhold-probe-phase1 "NAME=<stage-1 run dir>" ["NAME2=<dir>"] --device cuda:N [--json out.json]
```

Linear probes on teacher and posterior states at hold frames of `memtest4_eval`, excluding the
middle of a swap. They're fitted on even episodes and tested on odd ones.
- **Ball slot now, by swaps DONE.** 0 done is just the start slot, which is visible at the reveal,
  so read the later columns. Grouping by swaps still to come is misleading for the same reason.
- **Start slot and swaps so far**, each against what the current slot alone predicts. Above it means
  history is kept.

## Reward-decoder layout

`model.arch.decoder_layout`:
- `interleaved` is the default and every run before WINDOW60: `[z | a_t, x_t, a_{t+1}, x_{t+1}, …]`,
  newest frame x_{t+k-2}.
- `paired` is WINDOW60: `[z | (x_t,a_t), (x_{t+1},a_{t+1}), …]`, one MLP token per step, r_{t+k} at
  position K−1+k, newest frame x_{t+k-1}. At the lift, only offset 1 is memory-only.

## Evaluating a run

| what | command |
|---|---|
| phase-1 pixel eval, 0–4 | `pixi run eval-smt-phase1 decoder_run=<pd> outdir=<pd>/eval_EVAL` |
| phase-1 pixel eval, 5 swaps | `pixi run eval-smt-phase1 decoder_run=<pd> zarr_path=data/memtest4_eval5.zarr outdir=<pd>/eval_EVAL5` |
| prelim pixel eval | `pixi run eval-smt-prelim decoder_run=<pd> ...` (same overrides) |
| **old** prelim checkpoints (Sept 17–18) | `pixi run python scripts/eval_prelim_legacy.py decoder_run=<pd> ...` |
| posterior/teacher onset probe | `pixi run onset-probe-phase1 wm_run=<stage1> outdir=<stage1>/onset_probe_EVAL{,5} zarr_path=...` |
| full chain (stage 1 → readout → decoder → evals) | `scripts/twin_ft_chain.sh baseline <tag> <cuda:N>` (fixed dirs, resume + retry) |

About the legacy wrapper: old prelim checkpoints lack modules the current code always builds,
but which those runs never used (`teacher_dec.to_ball`, the smtcell `updater`). The wrapper loads
non-strictly and only tolerates those keys. It reproduced the original 0–4 eval exactly (0
differences in 630 rows).

The GPUs are shared with another user's jobs, so check `nvidia-smi` first. Use the scratchpad for
throwaway scripts. `sklearn` is in system `python3`, not the pixi env.

## Do not propose no-obs ("noobs") reward decoders

The user has ruled them out. In memtest4 the action carries only the phase ([0,0] reveal / [1,0] hold
/ [0,1] lift), never which cups swap. So a reward decoder without future frames can't predict the
reveal from any state before the last swap: those targets are noise. The existing P1-500-noobs run
stays as a historical result, but don't design new arms or controls around it.

## Comparability caveats (state them when comparing)

- **smt_prelim's eval is easier.** It warms up with a 60-frame teacher window, and the lift onset
  is always within frame 58. So its memory at the lift is the teacher's encoding of the whole
  prefix, and the recurrent updater is barely tested. Phase 1 warms up on frame 0 and chains the
  posterior (`eval_warmup: 1`).
- **Training sets differ.**

  | store | episodes | swaps | used by |
  |---|---|---|---|
  | `memtest4.zarr` | 250 (225 trained) | 0–4 | prelim, P1-250 |
  | `memtest4_big.zarr` | 500 (475 trained) | 0–4 | P1-500 and every ALIGN/DELTA run |
  | `memtest4_s3.zarr` | 378 | 0–3 | twin-finetune C0 |

  Compare only within a store, or say so.
- **`memtest4.zarr` has old x-based ball-slot labels.** `swap_sequences` handles it (replay
  checked on resting frames only). Don't use its `ball_slot` mid-swap.
- **Sample sizes are small:** unseen is n=44 (500-episode store) or 110 (250), and 5 swaps is n=63.
  Differences under about 0.1 are usually noise; use the CIs, or a paired test on the same episodes.

## Reference results (pixel F1 score; computed 2026-10-05 with `smt_f1_table.py`)

**Decoder target, before any alignment.** SMT decoding future frame **embeddings** (prelim) vs a
**ball-present reward** (phase 1):

| run | decoder input | decoder target (training signal) | trained on | seen | unseen | 5 swaps |
|---|---|---|---|---|---|---|
| prelim ctx60 | memory + future frames | future frame embeddings | 225 eps | 1.00 | 0.30 [0.17–0.42] | 0.24 |
| prelim ctx60, reweighted | memory + future frames | embeddings, memory-weighted | 225 | 0.99 | 0.26 | 0.33 |
| prelim ctx60, SMT cell | memory + future frames | embeddings | 225 | 1.00 | 0.32 | 0.12 |
| prelim ctx60 + ball head | memory + future frames | embeddings + ball | 225 | 0.96 | 0.28 | 0.18 |
| P1-250 | memory + actions + future frames | ball reward only | 225 | 0.96 | 0.30 [0.16–0.42] | 0.37 |
| P1-500 | memory + actions + future frames | ball reward only | 475 | 0.93 | 0.48 [0.20–0.70] | 0.53 |
| P1-500-noobs | memory + actions | ball reward only | 475 | 0.93 | 0.59 [0.32–0.79] | 0.54 |

Frames as decoder INPUT (P1-500 vs -noobs) is a different question from frames as the TARGET
(prelim vs P1-250). No phase-1 run predicts frames; its pixel decoder is fitted after stage 1.

At equal data (225 episodes), the two targets tie on unseen (0.30 vs 0.30), even though prelim
had the easier teacher-prefix eval. The reward target is ahead at 5 swaps (0.37 vs 0.24, with
overlapping CIs). Doubling the data is what moved unseen.

## Reference results: Naive vs Align-self vs Align-privileged (onset-probe F1; 2026-10-07)

From `onset_f1_compare.py`:

| | Naive | Align-self | Align-privileged |
|---|---|---|---|
| trained on | 225 episodes | 225 episodes | 475 episodes |
| **seen** | 0.93 (n=205) | 0.89 (n=205) | 0.94 (n=271) |
| **unseen** | 0.46 (n=110) | **0.82** (n=110) | 0.89 (n=44) |
| ↳ unseen 3 | 0.47 (n=44) | **0.94** (n=44) | 0.00\* (n=10) |
| ↳ unseen 4 | 0.50 (n=55) | 0.71 (n=55) | 0.96 (n=34) |
| **5 swaps** (same 63 for all) | 0.53 | **0.74** | 0.67 |
| **unseen by all three** (same 37) | 0.34 | 0.71 | **0.95** |
| always "ball" (5 swaps / unseen by all three) | 0.52 / 0.46 | | |

\* 10 episodes with 2 balls, both missed; accuracy there is 0.80.

**The three runs:**
- **Naive** = `smt_phase1_memtest4/window60_phase1`: WINDOW60, paired reward decoder with a 60-step
  window, no alignment.
- **Align-self** = `smt_phase1_memtest4/window60_alignself_phase1`:
  - positives-only alignment at **every at-rest hold frame**, mid-hold included;
  - labels are the W60-ALIGNPOS model's own decoder answer to a spliced "no swap, then lift" future
    (`scripts/build_self_slot0_labels.py`, `align_label: file`).
- **Align-privileged** = ALIGN-LIFT, `smt_phase1_memtest4_big/2026-09-22_21-12-04_memtest4_big_alignlift_phase1`:
  alignment on the hidden 3-way slot.

**Always say this:** Align-self's labels match the hidden "ball in slot 0 now" bit 98.7% of the time.
They're label-free in how they were obtained, but close to privileged in information.

t-SNE: `figures/tsne_naive_self_privileged.png`. Neighbour scores:

| | Naive | Align-self | Align-privileged |
|---|---|---|---|
| same slot, seen / unseen | 0.89 / 0.65 | 0.97 / 0.89 | 1.00 / 0.98 |
| same swap count | 0.58 | 0.40 | 0.45 |
| same start slot | 1.00 | 0.65 | 0.74 |

Other result docs: `docs/SMT_PHASE1_RESULTS.md` (ALIGN-*), `docs/SYMBOLIC_ABLATIONS.md`, `docs/DELTA_RESULTS.md`,
`docs/TWIN_FINETUNE_RESULTS.md`, `docs/WINDOW60_RESULTS.md`, `docs/WINDOW60_ALIGNPOS.md`.
