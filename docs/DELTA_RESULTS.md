# DELTA — results

Plan: [DELTA.md](DELTA.md). Everything below was run on 2026-10-02 (one seed per arm, 20k steps).

Eval set `data/memtest4_eval.zarr` (315 episodes, 63 per swap count, positive rate 0.359) and
`data/memtest4_eval5.zarr` (63 episodes, five swaps, all unseen). Seen / unseen = whether the
episode's exact (start slot, swap-pair sequence) occurs in the arm's training split: 271 / 44 for
0–4-swap-trained arms, 126 / 189 for depth-1 arms (every 2+ swap episode is unseen).

**Headline: two of the three kill criteria in DELTA.md 8 are met (tracking drops wherever the mask
binds; history leakage doesn't fall), and only the static-frame prediction holds.**
The gate does learn the structure DELTA describes: it is shut on every non-swap frame and
writes each swap once, into a few pair-dependent tokens. But tracking gets worse, history
leakage rises in the posterior, and D-depth1 doesn't compose (posterior F1 0.32 at five swaps).

## Arms

| id | train data | mask | λ | run |
|---|---|---|---|---|
| D-base | memtest4_big (0–4) | off | — | `smt_phase1_memtest4_big/2026-09-21_11-49-55_memtest4_big_noobs_phase1` (existing BASELINE-noobs) |
| D-tok-lo | memtest4_big (0–4) | token | 0.01 | `smt_phase1_memtest4_big/2026-10-01_23-49-41_big_delta_tok_lo_phase1` |
| D-tok-hi | memtest4_big (0–4) | token | 0.1 | `smt_phase1_memtest4_big/2026-10-01_23-49-46_big_delta_tok_hi_phase1` |
| D-dim | memtest4_big (0–4) | dim | 0.1 | `smt_phase1_memtest4_big/2026-10-02_00-52-50_big_delta_dim_phase1` |
| D-depth1-base | memtest4_depth1 (0–1) | off | — | `smt_phase1_memtest4_depth1/2026-10-02_00-08-41_memtest4_depth1_noobs_phase1` (**new control**) |
| D-depth1 | memtest4_depth1 (0–1) | token | 0.1 | `smt_phase1_memtest4_depth1/2026-10-02_00-52-53_memtest4_depth1_delta_phase1` |

All arms use the D-base recipe: no-obs reward decoder, binary lift head, `coef_unif 1e-3`, no
alignment. λ warms up linearly over 5k steps, and the gate bias starts at 3.0 (sigmoid 0.95).

- λ = 0.1 was picked as "best" for D-dim and D-depth1 because D-tok-hi beats D-tok-lo on
  overall posterior F1 (0.753 vs 0.698) and at five swaps (0.489 vs 0.378).
- D-depth1 uses the token mask, DELTA.md 4's starting granularity, and ran in parallel with
  D-dim rather than after it.
- D-depth1-base was added because no unlabelled depth-1 control existed. P1-DEPTH1 used dense
  privileged slot labels.

## Implementation

| file | change |
|---|---|
| `src/world_model/models/smt_phase1/updater.py` | `MaskNet` (MLP on the event → 10 or 2560 sigmoid gates, output layer zero-init with bias `delta_mask_init`); `gate()` = `(1−m)⊙z + m⊙ẑ`; `step_with_mask` on both updaters; `step` returns the gated state, so `observe()`, `imagine()`, DMT and every probe run the training equation |
| `src/world_model/models/smt_phase1/losses.py` | `step_with_mask`; `delta_l1 = mean(m)`, `delta_l1_prior = mean(m_p)`; `delta/*` metrics (mean, fraction > 0.5, static vs moving) |
| `src/world_model/models/smt_phase1/model.py` | `arch.delta_mask / delta_mask_init / delta_mask_hidden`, `loss.coef_delta / delta_warmup_steps`; λ_t in `_total` |
| `src/world_model/models/smt_phase1/dmt.py` | note only: rollouts already go through the gated `step` |
| `configs/smt_phase1/model/smt_phase1.yaml` | the five keys, inert defaults (`delta_mask: "off"` — **quoted**: a bare YAML `off` parses as `False`) |
| `scripts/verify_smt_phase1.py` | gate 4 (below); `delta_l1*` added to the gradient contract |
| `scripts/delta_extract.py`, `scripts/delta_analyze.py` | diagnostics (stage 1 under pixi/GPU, stage 2 under system python with scikit-learn) |

Smoke tests (DELTA.md 5), all passing:
1. `delta_mask: off` vs the pre-change code, same weights and batch: every loss term and the
   total are **bit-identical**.
2. Gate forced to 0 returns `z_t` exactly. Gate forced to 1 returns `ẑ` exactly. Checked for
   both updaters, token and dim.
3. `delta_l1` reaches only `posterior.mask_net` and `delta_l1_prior` only `prior.mask_net`.
   The existing four-term contract is unchanged.

## Onset probe — reward head at the lift onset (r1 = memory + lift action, no frame)

The eval pipeline reproduces the documented D-base numbers exactly (SYMBOLIC_ABLATIONS.md 4).

**Posterior** (encode once at t=0, then the gated posterior over every frame) — F1:

| arm | all | seen | UNSEEN | 0 | 1 | 2 | 3 | 4 | 5 swaps | acc all |
|---|---|---|---|---|---|---|---|---|---|---|
| D-base | **0.887** | **0.928** | **0.593** | 1.000 | 1.000 | **1.000** | **0.857** | **0.625** | **0.542** | **0.921** |
| D-tok-lo | 0.698 | 0.720 | 0.538 | 1.000 | 1.000 | 0.593 | 0.372 | 0.409 | 0.378 | 0.797 |
| D-tok-hi | 0.753 | 0.796 | 0.400 | 1.000 | 1.000 | 0.471 | 0.717 | 0.478 | 0.489 | 0.819 |
| D-dim | 0.758 | 0.791 | 0.500 | 1.000 | 1.000 | 0.692 | 0.489 | 0.524 | 0.488 | 0.838 |
| D-depth1-base | 0.661 | 1.000 | 0.431 | 1.000 | 1.000 | 0.324 | 0.500 | 0.436 | 0.314 | 0.740 |
| D-depth1 | 0.679 | 1.000 | 0.429 | 1.000 | 1.000 | 0.452 | 0.400 | 0.440 | 0.320 | 0.771 |

**Teacher** (60-frame window re-encoded at the onset, no updater) — F1:

| arm | all | seen | UNSEEN | 2 | 3 | 4 | 5 swaps | acc 5 |
|---|---|---|---|---|---|---|---|---|
| D-base | 0.944 | 1.000 | 0.400 | 1.000 | 0.980 | 0.732 | 0.316 | 0.587 |
| D-tok-lo | 0.921 | 0.995 | 0.452 | 1.000 | 0.945 | 0.694 | 0.449 | 0.571 |
| D-tok-hi | 0.907 | 0.979 | 0.333 | 0.960 | 0.875 | 0.711 | 0.474 | 0.683 |
| D-dim | 0.900 | 0.984 | 0.296 | 0.917 | 0.960 | 0.625 | 0.359 | 0.603 |
| D-depth1-base | 0.652 | 1.000 | 0.388 | 0.364 | 0.383 | 0.408 | 0.348 | 0.524 |
| D-depth1 | 0.667 | 1.000 | 0.492 | 0.318 | 0.588 | 0.507 | 0.447 | 0.333 |

**Prior** (posterior, then 9 gated prior steps) equals the posterior to three decimals in every
DELTA arm, because the prior's gate at the show action is ~0. It is therefore omitted. Five-swap
always-ball F1 is 0.518.

**D-depth1 vs the bit-only ceiling**, accuracy by depth 0–5 (ceiling = best a perfectly played
1-bit state can do, SMT_PHASE1_RESULTS.md):

| | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| D-depth1 posterior | 1.000 | 1.000 | 0.730 | 0.571 | 0.556 | 0.460 |
| D-depth1-base posterior | 1.000 | 1.000 | 0.603 | 0.587 | 0.508 | 0.444 |
| bit-only ceiling | 1.000 | 0.667 | 0.825 | 0.603 | 0.603 | 0.651 |

## Is the answer in the state? Decoder-free probe on the onset state

Logistic regression (PCA-64) fit on each arm's own training-split onset states, separately per
source, then tested on eval. The reward decoder only ever trained on teacher states, so this
separates "the answer is missing from the state" from "the decoder misreads an off-manifold state".

| arm | posterior all | posterior 2 / 3 / 4 | posterior 5 | teacher all | teacher 5 |
|---|---|---|---|---|---|
| D-base | **0.893** | 1.000 / 0.840 / 0.680 | 0.455 | 0.940 | 0.308 |
| D-tok-lo | 0.775 | 0.867 / 0.533 / 0.519 | 0.431 | 0.920 | 0.300 |
| D-tok-hi | 0.779 | 0.818 / 0.545 / 0.531 | 0.537 | 0.907 | 0.389 |
| D-dim | 0.746 | 0.643 / 0.561 / 0.528 | 0.426 | 0.903 | 0.293 |
| D-depth1-base | 0.661 | 0.324 / 0.500 / 0.436 | 0.314 | 0.652 | 0.340 |
| D-depth1 | 0.676 | 0.452 / 0.400 / 0.417 | 0.292 | 0.667 | 0.447 |

The gated posteriors lose the answer itself, not just its readability. Their probe F1 is
0.11–0.15 below D-base's.

## Closure at the onset — R² of the posterior state against the teacher state

| arm | 0 swaps | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| D-base | 0.865 | 0.745 | 0.567 | 0.206 | −0.082 | −0.675 |
| D-tok-lo | 0.534 | 0.002 | −0.436 | −0.664 | −0.390 | −0.334 |
| D-tok-hi | 0.345 | −0.509 | −1.320 | −1.727 | −1.977 | −2.317 |
| D-dim | 0.550 | −0.046 | −0.919 | −1.675 | −2.150 | −2.812 |
| D-depth1-base | 0.265 | 0.377 | 0.299 | 0.235 | 0.047 | 0.217 |
| D-depth1 | **0.936** | **0.984** | **0.977** | **0.961** | **0.956** | **0.953** |

On 0–4-swap data the gated chain leaves the teacher's manifold even at **zero** swaps, and the
gap grows with every swap. D-depth1's chain is the only one that tracks its teacher, but that
teacher is wrong at depth ≥ 2.

## History leakage (DELTA.md 1 re-run)

Rest states = motionless hold frames after the cups are down; 3775 states. Probes are fit *within*
each ball-slot group (episode-grouped 5-fold CV, PCA-64 + logistic regression) and pooled.
Majority baselines are 0.172 for swaps-so-far and 0.475 for start slot. This reimplementation
differs from the one behind DELTA.md 1's table (its majority baselines were 0.273 / 0.515), so
compare rows here with each other, not with that table.

| arm | state | swaps-so-far | start slot | within-slot variance from depth |
|---|---|---|---|---|
| D-base | teacher | 0.870 | 0.998 | 38.6% |
| D-base | posterior | 0.870 | 0.990 | 39.2% |
| D-tok-lo | teacher | 0.863 | 1.000 | 40.7% |
| D-tok-lo | posterior | **0.946** | 1.000 | **52.3%** |
| D-tok-hi | teacher | 0.848 | 1.000 | **27.9%** |
| D-tok-hi | posterior | **0.930** | 1.000 | 38.0% |
| D-dim | teacher | 0.839 | 0.997 | 31.3% |
| D-dim | posterior | **0.945** | 1.000 | 39.2% |
| D-depth1-base | teacher | 0.682 | 1.000 | 32.3% |
| D-depth1 | teacher | 0.768 | 1.000 | 25.0% |

The teacher's depth share falls by up to 11 points (D-tok-hi), but its history *probes* don't
move, and the posterior's rise to 0.93–0.95.

## The gate (DELTA.md 7.1)

Mean posterior gate per (event × memory token), `memtest4_eval`:

| event | D-tok-hi tokens 0 / 2 / 5 / 9 (all others 0.000) | D-depth1 token 5 (all others 0.000) |
|---|---|---|
| reveal, descent, static, lift onset, lift, show static | 0.000 / 0.000 / 0.000 / 0.000 | 0.000 |
| swap(0,1) | 0.052 / 0.015 / 0.112 / 0.042 | 0.188 |
| swap(0,2) | 0.046 / 0.014 / 0.119 / 0.033 | 0.189 |
| swap(1,2) | 0.004 / 0.019 / 0.124 / 0.000 | 0.024 |

By frame position within a swap (0 = first moving frame):

| arm | position 0: max-token m | tokens > 0.5 | positions 1–4: max-token m |
|---|---|---|---|
| D-tok-lo | 0.916 | 1.67 | ≤ 0.017 |
| D-tok-hi | 0.591 | 1.00 | 0.000 |
| D-depth1 | 0.676 | 0.67 | 0.000 |
| D-dim | 0.191 | 0.00 | ≤ 0.030 |

- Every swap is written **once, on its first moving frame, into one or two tokens**. Every other
  frame of every episode leaves `z` exactly unchanged.
- Swaps touching slot 0 (the slot that is always lifted) open tokens 0 and 9 in D-tok-hi, and
  swap(1,2) does not. In D-depth1, swap(1,2) barely opens anything (0.024).
- The prior's gate is a function of the action alone, so it is a constant during the whole hold:
  D-tok-hi opens token 5 at 0.127 on every hold step, and D-depth1 opens nothing at all.
- D-dim never opens a dimension past 0.5 on average. It also leaks slightly on static frames
  (token 8: 0.016).

## Blocks (DELTA.md 7.2) — one token at a time, posterior rest states

Slot probe from that token alone / swaps-so-far within slot (majority 0.48 / 0.36 for a
constant token):

| arm | tok 0 | tok 1 | tok 2 | tok 3 | tok 4 | tok 5 | tok 6 | tok 7 | tok 8 | tok 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| D-base | .78/.83 | .74/.86 | .73/.80 | .74/.82 | .78/.85 | .76/.83 | .73/.82 | .73/.77 | .70/.81 | .67/.78 |
| D-tok-hi | **.74/.80** | .48/.36 | **.69/.90** | .48/.36 | .48/.36 | **.78/.87** | .48/.36 | .48/.36 | .48/.36 | **.71/.72** |
| D-depth1 | .48/.36 | .48/.36 | .48/.36 | .48/.36 | .48/.36 | **.59/.75** | .48/.36 | .48/.36 | .48/.36 | .48/.36 |

The latent does break into parts, but not the per-place blocks DELTA.md 2 predicted. In
D-tok-hi it splits into **four live tokens and six dead ones** that are never written after t=0.
Each live token carries the slot *and* the swap count together. D-depth1 keeps one live token.

## t-SNE

All in `logdir/delta_diag/final/figures/`:

- `tsne_onset_all_arms.png` — lift-onset states (eval + eval5), teacher (top) vs posterior
  (bottom), every arm; colour = true slot; ● seen, × unseen 2–4, ▲ five swaps.
- `<arm>_tsne_rest.png` — rest states, teacher vs posterior, coloured by ball slot / swaps so far /
  start slot.
  - In the gated arms each posterior episode collapses to a single point at rest, since the gate
    is shut.
  - Those points sort cleanly by **start slot** and **swap count**, while the current ball slot is
    mixed. The posterior is organised by history, the opposite of DELTA's prediction.
  - D-base's teacher and posterior both form ball-slot-coherent strands for seen sequences.
- `<arm>_gate.png`, `<arm>_dz_posterior.png`, `<arm>_dz_teacher.png` — the event × token tables
  above as heatmaps. `|Δz|` exists for the ungated arms too.

## Predictions (DELTA.md 8) vs outcome

| prediction | outcome |
|---|---|
| static-frame mask → ~0 | **held**: exactly 0.000 in every token arm (D-dim ≤ 0.016) |
| history probes fall substantially at λ = 0.1 | **failed**: teacher swaps-so-far 0.848 vs 0.870; posterior *rises* to 0.930; start slot 1.000 everywhere |
| composition error at 4-unseen falls from BASELINE's 0.59 | **failed** (proxy): posterior F1 at 4-unseen 0.300 (tok-hi) / 0.500 (dim) vs 0.522 |
| D-depth1 ≥ 0.9 at five swaps | **failed**: 0.320 F1 / 0.460 acc, ≈ its control (0.314 / 0.444) and below the bit-only ceiling |

Kill criteria:
- **"The mask is fully open where tracking works"**: not met. The mask is very sparse at every λ.
- **"The mask shuts and tracking collapses at every λ where it binds"**: **met**. λ = 0.01
  already binds, and posterior F1 falls from 0.887 to 0.70–0.76 at every λ and granularity.
- **"History leakage doesn't fall at any working λ"**: **met**. It rises in the posterior.

## Why it fails (from the measurements above)

1. **λ barely matters; the live `L_post` target shuts the gate.** At step 500, with λ_t ≤ 0.01,
   the posterior gate was already ~0.01. Once the gate can hold `z` exactly, the cheapest way to
   zero `L_post` is to make the teacher stationary and close the gate. λ 0.01 and 0.1 differ
   only in how many tokens a swap writes (1.7 vs 1.0 above 0.5).
2. **A shut gate removes error correction.** D-base's posterior rewrites every token from
   `(z, e)` on every frame, which pulls the state back toward an observation-consistent one. The
   gated posterior writes once per swap and then holds that write untouched. Any error in the
   write therefore persists, and the teacher's residual per-step drift (which `L_post` tolerates)
   is never followed. Closure falls below zero after one swap, and the decoder-free probe shows
   the information is lost, not just misread.
3. **Stationary and sparse do not rule out history.** DELTA.md 2 rejected history encoding on the
   grounds that a history buffer writes to count-dependent locations. A *counter* inside a fixed
   token, bumped by every swap event, is stationary and as sparse as the slot update. The
   swaps-so-far probe rising to 0.93–0.95 is exactly that.
4. **With 0–1-swap data, the sparsest code is the 1-bit one.** D-depth1 found a single token
   written only by swaps touching the lifted slot. That is ALIGN-TARGET's {0} vs {1,2} partition,
   which isn't closed under the swaps, reached without any label. Sparse updates did not force
   code reuse at depth 2+.

## Caveats

- The one gain is the teacher at five swaps: F1 0.449 / 0.474 for D-tok-lo / D-tok-hi vs 0.316.
  But accuracy is mixed (0.571 / 0.683 vs 0.587), n = 63, and it doesn't carry over to
  the posterior.
- One seed per arm.
- No stage-2 readout, pixel decoder or pixel eval was run for the DELTA arms. The numbers above
  are onset-probe numbers, not pixel F1.
- DELTA.md 7.4 (composition/drift error split) and 7.5 (swap-operator transfer R²) were not run.
  The `swapop` / `errsplit` code is not in the repository. Closure R² by depth is the
  substitute.
- The history-probe implementation is new (see its section), so its absolute values aren't
  comparable to DELTA.md 1.
- D-depth1 used λ 0.1 and the token mask, chosen before D-dim finished. D-dim did not beat the
  token arms, so the choice stands.

## Reproducing

```bash
pixi run python scripts/verify_smt_phase1.py "gates='2,3,4'" model.arch.delta_mask=token \
    model.loss.coef_delta=0.1 model.arch.decoder_obs=false
pixi run train-smt-phase1-delta-tok-lo            # D-tok-lo
pixi run train-smt-phase1-delta-tok-hi            # D-tok-hi
pixi run train-smt-phase1-delta-dim               # D-dim
pixi run train-smt-phase1-depth1-noobs            # D-depth1-base
pixi run train-smt-phase1-delta-depth1            # D-depth1
pixi run delta-extract --out logdir/delta_diag/final --device cuda:0 \
    --arm D-base=<run dir> --arm D-tok-lo=<run dir> ...   # GPU; dumps states, gates, onset reads
python3 scripts/delta_analyze.py logdir/delta_diag/final  # system python (scikit-learn)
```

Full report with every table: `logdir/delta_diag/final/report.md`; raw numbers in `results.json`.
