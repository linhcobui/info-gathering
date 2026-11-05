# WINDOW60 — results (interim: onset probe only)

Plan: [WINDOW60.md](WINDOW60.md). Stage 1 and the readout finished. The pixel decoder was **paused
by request** on 2026-10-06 before its first checkpoint, so there's no pixel eval yet. To resume:
`CONFIG=train_memtest4_window60_phase1 RUN=logdir/world_model/smt_phase1_memtest4/window60_phase1 NO_REPORT=1 scripts/twin_ft_chain.sh baseline window60 cuda:N`,
then `scripts/window60_report.sh` (it overwrites this file).

Read-out here: the frozen stage-1 reward head at the lift onset (`onset-probe-phase1`, `*_r1`).
Both runs were trained on `memtest4.zarr` (225 episodes); P1-250 is the control.

## F1 score [95% CI]

| | seen (205) | unseen 3 (44) | unseen 4 (55) | 5 swaps (63) |
|---|---|---|---|---|
| *always say "ball"* | *0.50* | *0.67* | *0.51* | *0.52* |
| P1-250 teacher | 0.99 | 0.31 [0.08–0.50] | 0.31 [0.12–0.49] | 0.38 [0.20–0.54] |
| WINDOW60 teacher | 1.00 | 0.38 [0.16–0.57] | 0.56 [0.36–0.72] | 0.41 [0.21–0.59] |
| P1-250 posterior | 0.96 | 0.20 [0.00–0.40] | 0.34 [0.14–0.53] | 0.32 [0.12–0.50] |
| WINDOW60 posterior | 0.93 | 0.47 [0.25–0.65] | 0.50 [0.30–0.65] | 0.53 [0.35–0.68] |

## Why the F1 rise is mostly a bias shift (posterior)

| | says "ball" (true rate) unseen 3 / unseen 4 / 5 | balanced accuracy unseen 3 / unseen 4 / 5 |
|---|---|---|
| P1-250 | 0.18 (0.50) / 0.29 (0.35) / 0.25 (0.35) | 0.45 / 0.52 / 0.51 |
| WINDOW60 | 0.36 (0.50) / 0.53 (0.35) / 0.49 (0.35) | 0.55 / 0.58 / 0.61 |

On the same episodes, accuracy is unchanged:

| bucket | P1-250 acc | WINDOW60 acc | WINDOW60 fixed / broke | McNemar p |
|---|---|---|---|---|
| seen | 0.98 | 0.95 | 3 / 8 | 0.23 |
| unseen 3 | 0.45 | 0.55 | 9 / 5 | 0.42 |
| unseen 4 | 0.58 | 0.56 | 10 / 11 | 1.00 |
| 5 swaps | 0.59 | 0.60 | 12 / 11 | 1.00 |
| unseen 4 + 5 | 0.58 | 0.58 | 22 / 22 | 1.00 |

## Reading

- **No significant composition gain.** WINDOW60 says "ball" about twice as often on the hard
  buckets, which is what lifts its F1. Balanced accuracy rises 0.06–0.10, but nowhere near
  significance at these n, and every F1 is still at or below always-"ball".
- **Seen sequences dip slightly** (0.98 → 0.95, not significant).
- **Still to run:** the mid-hold probe (does the teacher now carry the slot after 2–4 swaps?) and
  the pixel eval. The probe needs no pixel decoder: `pixi run midhold-probe-phase1
  "P1-250=…" "WINDOW60=logdir/world_model/smt_phase1_memtest4/window60_phase1"`.
