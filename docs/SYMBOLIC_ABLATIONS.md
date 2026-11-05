# Symbolic observation ablations

Does the frozen R2-Dreamer embedding limit memtest4 phase 1? Four observation formats, same
model, same objective, same data. Answer: **no.** Nothing beats the learned embedding, and the
result that appeared to say otherwise was a leak in my own encoding.

Builders: [scripts/build_oracle_state.py](../scripts/build_oracle_state.py),
[scripts/build_scene_state.py](../scripts/build_scene_state.py),
[scripts/build_symbol_obs.py](../scripts/build_symbol_obs.py).
Model hook: `oracle_obs: true` swaps `observe()` from the pixel encoder to a cached per-frame
vector ([model.py](../src/world_model/models/smt_phase1/model.py)); everything downstream is unchanged.

---

## 0. What the embedding actually is

Measured before replacing anything:

| property | value |
|---|---|
| nominal width | 1024 |
| 90% of variance | 13 dims |
| 99.999% of variance | 27 dims |
| distinct frames in the whole dataset | **28** |
| swap pair decodable from ONE frame, cups moving | **1.000** |
| swap pair decodable from ONE frame, cups stationary | 0.311 (chance 0.333) |
| per-dimension raw std | min 3e-4, median 0.12 |

Two things follow. The rendered world is a 28-symbol alphabet embedded in ~27 effective
dimensions, so there is no information bottleneck. And the embedding is *perfectly* informative
about which pair is swapping whenever the cups are in motion; it says nothing on the two frames
per swap block where the cups are stationary, because the pixels show nothing there either.

Normalisation was checked for numerical damage (`(e-mean)/(std+1e-6)` amplifying near-null
dimensions) and is clean: the smallest raw std is 3e-4, three orders above the epsilon floor.

---

## 1. ORACLE v1 / v2 — 13-D hand-built. **INVALID, results discarded.**

```
[0:3]   phase one-hot:          {reveal, hold, show}
[3]     cup height:             scalar, 1.00 = cups up, 0.29 = cups down
[4:8]   swapping-pair one-hot:  {none, slots(0,1), slots(0,2), slots(1,2)}
[8]     swap progress:          scalar 0.0 -> 1.0 across the swap, 0 when no swap
[9]     ball_shown:             binary flag                      ] v2 split; in v1 these
[10:13] ball slot one-hot:      {slot0, slot1, slot2}            ] were one 4-way one-hot
                                all-zero unless ball_shown       ] {not visible, slot0/1/2}
```

Channel by channel:

- **`[0:3]` phase** — which stage the episode is in. `reveal` = cups up, ball on display;
  `hold` = cups down, swaps happen, ball hidden; `show` = the cup standing at slot 0 rises.
  Observable: you can see whether cups are up, down, or one is rising.
- **`[3]` cup height** — `cup_z_max` rescaled to [0, 1]. 1.00 with cups up, 0.29 with them down.
- **`[4:8]` swapping-pair** — which two **slots** are currently exchanging cups. Slots, not cup
  identities: identity is the hidden variable and never appears anywhere in the vector.
- **`[8]` progress** — how far through the current swap. A swap takes 6 frames, so this runs
  `0.0, 0.2, 0.4, 0.6, 0.8, 1.0` and sits at 0 otherwise. It is load-bearing: swap blocks are
  **back-to-back with no gap**, and 35.1% of consecutive pairs are the *same* pair, so without it
  two consecutive `(1,2)` swaps look like one continuous 12-frame `(1,2)` signal — and a double
  swap is the identity, i.e. the opposite answer. The reset from 1.0 to 0.0 is the only boundary
  marker. (This also broke the first verification tracker, which looked for "the previous frame
  had no swap" — a condition that never occurs between adjacent blocks.)
- **`[9]` / `[10:13]` ball** — v2 splits visibility (an observable fact about the scene) from the
  slot (the hidden variable it occasionally exposes). v1 folded both into one 4-way one-hot where
  "not visible" was an active category rather than an absence.

Real rows, from a 4-swap episode (hold 10–45, lift onset 46):

```
  t |  phase  | cupZ | swapping-pair | prog | shown | slot
  0 | [1 0 0] | 1.00 |   [1 0 0 0]   | 0.0  |   1   | [0 0 1]  reveal: ball on show at slot 2
 13 | [0 1 0] | 0.29 |   [1 0 0 0]   | 0.0  |   0   | [0 0 0]  cups down, ball hidden
 14 | [0 1 0] | 0.29 |   [0 1 0 0]   | 0.0  |   0   | [0 0 0]  swap (0,1) begins
 19 | [0 1 0] | 0.29 |   [0 1 0 0]   | 1.0  |   0   | [0 0 0]  swap (0,1) ends
 20 | [0 1 0] | 0.29 |   [0 0 0 1]   | 0.0  |   0   | [0 0 0]  swap (1,2) begins immediately
 38 | [0 1 0] | 0.29 |   [1 0 0 0]   | 0.0  |   0   | [0 0 0]  slack, nothing happening
 46 | [0 0 1] | 0.29 |   [1 0 0 0]   | 0.0  |   0   | [0 0 0]  lift onset, cup still down
```

**Both defects are visible in those rows.** At t=19 and t=20 the cups are *stationary* — the
render is the generic cups-down picture and the embedding scores 0.311 (chance) — yet the vector
announces which pair is swapping. That is defect (b) below. And `[9] ball_shown` at t=46 is the
leak: shown here as 0 (the array has since been rebuilt), but in the version the arms trained on
it read **1** whenever the ball was under the target cup, at the exact frame the probe reads.

Verified sufficient (a hand-written tracker scores 878/878) and non-leaking *during the hold*
(0/878 episodes expose the slot while the cups are down).

**Two defects, found only by pushing on the encoding:**

**(a) It leaks the answer at the lift onset.** `ball_shown` was gated on
`phase == show AND ball_under_target`. The generator lowers the action onto the first show frame
*while the cup is still down* — measured, `cup_z` at the onset equals the previous frame — so the
pixels reveal nothing. The flag fired anyway, in exactly the 113/315 eval episodes where the ball
is under the target (and read 0 for the other 202, which is equally diagnostic). The onset probe
reads the state *at that frame*, so the arms were handed their answer at the point of measurement.

**(b) It removes the segmentation problem.** It labels `swapping=(1,2)` on the two stationary
frames per block where the embedding scores 0.311. Episode-level information is the same, but
per-frame it is strictly more informative than the pixels.

Both defects inflate performance. Every oracle number is discarded.

**One finding from them is still worth keeping.** v1 and v2 carry identical information — the only
change was "not visible" going from an active one-hot category to plain absence — yet the no-align
arm moved from 0.638 to 0.952 F1 at five swaps. **Information-identical re-encodings are not
performance-identical**, which is a caution for every cross-arm comparison in this project.

---

## 2. SCENE-STATE — 16-D, the cups' actual coordinates

```
OBSERVABLE
  [0:3]    phase one-hot
  [3:12]   three cup positions, SORTED BY X        <- cup identity never handed over
  [12]     ball_shown, read from the RENDER via ball_detector
NON-OBSERVABLE
  [13:16]  ball xyz, zero unless shown
```

Three deliberate choices, each fixing a defect above:

- **Cup identity is hidden.** `cup_xyz` is indexed by identity in the generator, which *is* the
  hidden variable. Sorting by x each frame yields the observable multiset; the model must track
  cups by continuity, as a viewer does.
- **No swap-pair or progress channel.** The swap is implicit in cup motion, so stationary boundary
  frames carry no swap information — matching the pixels.
- **Visibility comes from the render**, not from a rule. `ball_pixels(frame) >= 30`. This makes
  defect (a) structurally impossible rather than something to reason about correctly.

Poses are reconstructed exactly from stored episode seeds, verified per episode against
`ball_slot`. Two RNG protocols exist and both are handled: `memtest4_big` samples `hold_frames`
inside `sample_episode`, the eval sets pass it in, which shifts the stream. **878/878 episodes
reconstruct exactly; ball visible at the onset in 0/878.**

---

## 3. SYMBOL — 28-D one-hot over the visual alphabet

A one-hot over the 28 distinct rendered frames: a **bijection** with what the renderer shows, so
information is held exactly fixed against the embedding and only geometry varies.

Building it surfaced a third trap. Rounding embeddings to 3 decimals gave 34 "distinct" frames,
six of them split-specific (`big`-only and `eval`-only). The pairwise distance histogram is
bimodal by three orders of magnitude — 9 pairs below 0.009, then nothing until 2.84 — so those
were float-noise duplicates. Merging at any threshold in that gap gives 28, and all three splits
then use the same 28 symbols. Without the merge, eval would carry one-hot dimensions never active
in training.

---

## 4. Results

Onset probe, no alignment in any arm, `memtest4_eval` (271 seen / 44 unseen) and `memtest4_eval5`.

**Posterior (the recurrent chain) — F1:**

| arm | observation | all | seen | UNSEEN | depths 0–4 | 5 swaps |
|---|---|---|---|---|---|---|
| BASELINE-noobs | R2D embedding, 1024-D | 0.887 | 0.928 | **0.593** | 1.00 1.00 1.00 0.86 0.63 | **0.542** |
| SCENE-noalign | cup xyz, identity hidden | **0.891** | **0.968** | 0.400 | 1.00 1.00 1.00 **0.94** 0.55 | 0.457 |
| SYMBOL-noalign | 28-way one-hot | 0.755 | 0.814 | 0.286 | 1.00 0.88 0.87 0.50 0.49 | 0.276 |
| *ALIGN-LIFT (reference, privileged label)* | R2D embedding | *0.934* | *0.941* | *0.889* | *1.00 1.00 1.00 0.76 0.91* | *0.667* |

**Teacher (one-shot bidirectional encode) — F1:**

| arm | all | seen | UNSEEN | 5 swaps |
|---|---|---|---|---|
| BASELINE-noobs | 0.944 | 1.000 | 0.400 | 0.316 |
| SCENE-noalign | 0.933 | 1.000 | 0.444 | 0.263 |
| SYMBOL-noalign | 0.893 | 0.968 | 0.320 | 0.513 |

### What is okay

- **SCENE-STATE ties the embedding overall** (0.891 vs 0.887) and is *better on seen sequences*
  (0.968 vs 0.928) and at 3 swaps (0.94 vs 0.86). A faithful, disentangled, 16-D physical state
  is a perfectly good substrate — just not a better one.

### What failed

- **SYMBOL loses badly** — 0.755 vs 0.887 overall, and half the five-swap score (0.276 vs 0.542) —
  despite carrying *exactly* the same information by construction. The only difference is
  geometry: a one-hot makes every frame orthogonal to every other, destroying the similarity
  structure the embedding has (two mid-swap frames of one swap are near each other in embedding
  space, orthogonal as symbols). **The learned metric structure is doing useful work.**
- **SCENE is worse on unseen sequences** (0.400 vs 0.593) and at five swaps (0.457 vs 0.542).
- **No observation format reaches ALIGN-LIFT's 0.889 unseen.**

---

## 5. Conclusions

1. **Perception is not the bottleneck.** Every observation carries the task perfectly at the
   episode level, and none beats the learned embedding on the chain.
2. **Metric structure beats information-equivalence.** The one-hot is the control that proves it:
   same information, orthogonal geometry, materially worse.
3. **The apparent "perfect observations fix it" result was a leak.** Retracted.
4. **The bottleneck is composition, and it is unchanged by the observation.** The error split
   (`swapop`/`errsplit`, see SMT_PHASE1_RESULTS.md) puts BASELINE-noobs at 0.59 *composition*
   error at both 4-unseen and 5 swaps, versus ALIGN-LIFT at **0.00 composition error at every
   depth including the unseen depth 5**. What buys composition is the 3-way alignment label, not
   the input format.
5. **Method note.** Three separate encoding bugs were caught only by adversarial checking — an
   onset leak, an over-informative swap channel, and split-specific symbols. The habit that caught
   all three: derive the observable from the artefact itself (the render, the seed-reconstructed
   pose) rather than from a rule about what *ought* to be observable, and verify with a
   leak test at the exact frame the metric is read.

## 6. Reproducing

```bash
pixi run python scripts/build_scene_state.py      # 878/878 exact, 0 onset leaks
pixi run python scripts/build_symbol_obs.py       # 34 raw rows -> 28 merged symbols
pixi run train-smt-phase1-scene-noalign
pixi run train-smt-phase1-symbol-noalign
pixi run onset-probe-phase1 wm_run=<run> zarr_path=data/memtest4_eval.zarr \
  'meta_keys=[ball_under_target,ball_slot,phase,embed_scene_state]'
```
