# WINDOW60-ALIGNPOS: WINDOW60 + alignment on the positive lift bit, then finetune

Status: plan (2026-10-06). Runs after the current BPTT finetune finishes.

## The run it builds on

**ALIGN-POS** (`logdir/world_model/smt_phase1_memtest4_big/2026-09-28_14-20-42_memtest4_big_alignpos_phase1`,
config `train_memtest4_big_alignpos_phase1.yaml`):
- **Alignment:** at motionless hold frames, the states where the ball IS under the cup that will
  lift are pulled together (`align_label: target`, `align_positive_only: true`, `coef_align: 0.1`).
- **Other losses:** no uniformity (`coef_unif: 0`), no separation hinge (`coef_sep: 0`).
- **Data and decoder:** the 500-episode `memtest4_big`; the old reward decoder (actions only, 20-step
  window).
- **Results:** posterior F1 score seen 0.79, unseen 0.55, 5 swaps 0.60 (onset probe).

The label is the reveal itself, carried back. Motionless hold frames only occur after the last
swap, so "ball under the cup that will lift" at those frames is exactly the episode's own answer at
the lift.

## The new run: WINDOW60 + ALIGN-POS

Config `train_memtest4_window60_alignpos_phase1.yaml` = `train_memtest4_window60_phase1.yaml` plus
ALIGN-POS's loss block:

| | WINDOW60 | **WINDOW60-ALIGNPOS** |
|---|---|---|
| data | `memtest4.zarr`, 225 trained | same |
| teacher window | 60 | 60 |
| reward decoder | paired (frame, action) tokens, window 60 | same |
| alignment | none | **positives of the lift bit, coef 0.1** |
| uniformity | 1e-3 | **0** (as ALIGN-POS) |
| separation hinge | 0 | 0 |
| steps / batch | 20k / 128 | same |

`memtest4.zarr`'s old ball-slot labels differ from the new ones only mid-swap, and alignment only
uses still frames, so the label is correct there.

## Steps

1. **Smoke tests.**
   - `verify-smt-phase1` with these overrides must show the align loss reaching the teacher and
     nothing else.
   - 100 training steps: the align loss is nonzero and RAM is OK. Use 2 loader workers (the machine
     is near its RAM limit).
2. **Stage 1:** 20k steps (about 40 min on a free GPU; longer while the GPUs are shared).
3. **Evaluate before finetuning**, all without the pixel decoder:
   - **onset probe:** seen 0–4, unseen 3, unseen 4, 5 swaps;
   - **judge test:** decoder from cups-down (frame 13) on unseen 4 and 5 swaps;
   - **mid-hold probe:** slot by swaps done, and history.
4. **Finetune with BPTT exactly as for WINDOW60:**
   - trains the posterior only, against the frozen reward decoder's memory-only read-out at the onset;
   - trained on seen 0–4 + unseen 4 test episodes (260);
   - **held out: unseen 3 and 5 swaps**;
   - lr 1e-6, one optimiser step per full pass, evaluate every 25 steps.
5. **Compare with WINDOW60's finetune** on the held-out sets.

## What would count

- **Does alignment change the starting point?** Onset-probe unseen 4 / 5 swaps, and the judge
  test, against WINDOW60 (0.56 / 0.60 onset; 0.64 / 0.60 judge).
- **Does finetuning now transfer?** Unseen 3 and 5 swaps rise during finetuning, instead of falling
  or staying flat as they did for WINDOW60.
