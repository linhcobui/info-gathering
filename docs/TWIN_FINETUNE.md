# TWIN-FINETUNE — teaching the updater new situations without labels

**One-line version.** Train SMT as usual. Then, on a fresh batch of episodes, finetune *only the
updater* so that its states (a) predict the reveal correctly when decoded by the **frozen**
decoder, and (b) stay close to a slow-moving copy of itself. The frozen decoder fixes what
states *mean*. The reveal says whether the updater got it right. The slow copy stops it from
cheating.

Status: **run 2026-10-05, all five arms finished 05:20.** Kill criterion 2 met: the finetune
improves the 4-swap sequences it practised on (0.58 → 0.85, p = 0.016), not held-out ones (no
change) or 5 swaps (no change in balanced accuracy). C1 was undertrained at equal steps, so the
"just data" question is still open. Results and reading: [TWIN_FINETUNE_RESULTS.md](TWIN_FINETUNE_RESULTS.md).
§0 lists what changed from the plan below; where they disagree, §0 is what ran.

## 0. As run (2026-10-05)

**Depths, per your change.** Stage 1 trains on **0–3 swaps**. The finetune pool has 0–4 swaps, but
its 4-swap episodes come from only **half** of the 243 possible 4-swap sequences. Test 4-swap
episodes then split into *seen in the finetune pool* and *never seen*, and 5 swaps stays pure
extrapolation.

| store | what | episodes |
|---|---|---|
| `data/memtest4_s3.zarr` (S1) | `memtest4_big` minus its 4-swap episodes, byte-identical frames | 378 (depths 63/63/126/126) |
| `data/memtest4_ft.zarr` (FT) | fresh seed 10; 4-swap episodes from the allowed half only | 598 (102/102/101/99/194; 92 distinct 4-swap sequences) |
| `data/memtest4_s3ft.zarr` (C1) | S1 train + FT + S1's 25 holdout last | 976 |
| test, 4 swaps | `memtest4_eval` | 26 seen in FT · 37 unseen (30 held out, 7 just not drawn) |
| test, 5 swaps | `memtest4_eval5` | 63, all unseen |

The held-out half is stratified so half of the test set's 4-swap sequences fall in it
(`data/memtest4_ft_split.json`). Sequences are read by replaying the generator from each episode's
seed, cross-checked against the ball-slot labels (`src/info_gathering/swap_sequences.py`; it
reproduces every earlier seen/unseen mask exactly).

**Changes to the losses, each forced by a measurement:**

- **L_reveal uses the 3rd-largest yellowness, not a soft pixel count.** The eval calls "ball" when
  at least 3 pixels have yellowness > 0.15, which is the same as "the 3rd-largest yellowness
  exceeds 0.15". So `logit = (3rd-largest yellowness − 0.15) / 0.05`, and the loss's call matches
  the eval's exactly. The soft count in §5 failed: ~4000 background pixels each leak about 0.05,
  so empty frames scored about 30 "pixels", and soft and hard verdicts agreed on only 12–37% of
  frames.
- **L_recon is in embedding space:** ‖readout(z_t) − e_t‖², the readout's own training loss.
  That's cheaper than going through the pixel decoder, and it says the same thing: "this state
  is still readable".
- **lr = 1e-6, not 3e-5.** The gradient through the ~50-step chain is about 170 at step 1, and Adam
  moves every weight by about the lr each step. On the 0–4 baseline, 3e-5 broke tracking within 10
  steps (batch accuracy 0.94 → 0.71). 3e-6 faded the rendered ball. 1e-6 held accuracy at
  0.92–0.97 while the reveal loss fell.
- **w_recon = w_twin = 10** (not 0.1 / 1). At lr 1e-6 both terms are tiny (recon ~0.005, twin
  gap ~5e-4). Weight 10 makes a doubling of either cost about 10% of the reveal loss. These
  weights are **not tuned**.

**Smoke tests passed** (§10):

1. With lr 0, the export reproduces the original model's pixel eval exactly: 0 differences over
   630 rows, including float MSE.
2. The training rollout reproduces `observe()`/`imagine()` exactly: state gap 0.0 at the onset
   and over the imagination.
3. Only the online updaters receive gradients (asserted at step 1 of every run).

**Queue** (`scripts/twin_ft_chain.sh`, logs in `logdir/twin_ft/`):

| GPU | job |
|---|---|
| 0 | C0: stage 1 → readout → decoder → evals |
| 1 | C1: the same, on S1 + FT |
| 0 | F1 and F3, once C0's decoder exists |
| 1 | F2, once C0's decoder exists |

Each finetune arm: 3000 steps at ~1.5 s/step on the shared GPUs. Each arm then runs the pixel
eval (0–4 and 5), the onset probes, and a pixel eval of its EMA twin.


---

## 1. Why

**The failure we keep seeing.** When the chain meets a swap sequence it hasn't trained on, it
lands on the wrong state. In plain terms: from state z2, a swap should bring it back to z1 (the
state meaning "ball under cup a"), but it goes to z3 instead.

**What we learned last time.** On the baseline, those wrong states were *not* in unfamiliar
territory. At depth 4 they sat as close to familiar training states as the right ones did (or
closer — 5.92 vs 6.38). Snapping to the nearest familiar state didn't help. So the fix can't be
"pull states back toward familiar ones". The updater needs to learn to land on the *right*
familiar state, and that requires a signal saying what's right.

**Where that signal comes from, for free.** Every episode ends with a reveal: a cup lifts, and
the ball is either there or not. That's visible in the pixels. No hidden-state labels are needed.

## 2. The idea, in plain words

| part | role | trained? |
|---|---|---|
| teacher, frozen encoder, reward head | from stage 1 | **frozen** |
| readout + pixel decoder | the "dictionary": turns a state into an image. Defines what a state *means* | **frozen** |
| online updater (posterior + prior) | carries the belief through the episode | **trained** |
| slow updater ("twin") | a lagging copy of the online updater, refreshed as a running average of its weights | not trained by gradients; follows the online one slowly |

**Why freeze the dictionary.** We never have to say what the correct state z1 *is*. Any state the
frozen decoder renders as "ball under cup a" counts as meaning that. The updater just has to land
somewhere that decodes to the truth.

**Why the twin.** If you train hard against a fixed decoder, the updater can find odd states that
make the decoder draw a ball without genuinely tracking anything — adversarial examples for the
decoder. Keeping the online updater close to its slow twin makes those sudden exploits expensive.

## 3. Data — three separate sets

| set | contents | labels used? | role |
|---|---|---|---|
| **S1** | `memtest4_big`, the existing 475 training episodes | yes (stage 1, as before) | already done |
| **FT** | **new** `memtest4_ft`, ~500 episodes, 0–4 swaps, fresh seed | **no** — only pixels | finetuning |
| **TEST** | `memtest4_eval` (315) + `memtest4_eval5` (63), untouched | — | evaluation only |

**Two rules that keep the test honest:**
1. **Filter FT against TEST.** Drop any FT episode whose exact start slot and swap order appears
   in TEST. Otherwise "unseen" test sequences become seen during finetuning.
2. **FT stays at 0–4 swaps.** Five swaps stays a genuine extrapolation test.

Every TEST episode then falls into one of two buckets: **seen in S1**, or **novel to both S1 and
FT**. The headline number is the second bucket, plus five swaps.

FT is generated with the existing generator: copy `configs/data/memtest4_big.yaml`, give it a
fresh master seed (S1 = 2, eval = 1; use e.g. 10), write `data/memtest4_ft.zarr`, run
`pixi run generate-memtest4 --config configs/data/memtest4_ft.yaml`.

## 4. What one training step does

For a batch of FT episodes:

1. **Warm up** the state from the first frame with the frozen teacher — exactly as in evaluation.
2. **Observe:** run the *online* posterior updater frame by frame up to the lift onset, feeding
   in real frames. Keep every state.
3. **Imagine:** from the onset state, run the *online* prior updater forward through the show
   frames, with no frames fed in — exactly as in evaluation.
4. **Decode** the states with the frozen readout and frozen pixel decoder, to get images.
5. Run steps 1–3 again with the **slow twin**, with no gradients.
6. Compute the losses below, update the online updater, then nudge the twin toward it:

```
twin_weights  ←  0.999 × twin_weights  +  0.001 × online_weights
```

Gradients flow back through the whole rollout (~50 observe steps + 9 imagine steps), so the
reveal can reach the swaps that caused it. Episodes are short, so this is affordable.

## 5. The losses

Notation, once:
- `x_t` — the real frame at time t.
- `z_t` — the online updater's state at time t; `z̃_t` — the twin's state at the same time.
- `D(z)` — the frozen decoder: readout then pixel decoder, state → image.
- `ball(img)` — a soft "is there a ball in this image" score in [0, 1] (defined below).

### L_reveal — be right (the main signal)

At show frames from offset 2 onward (the cup has risen enough that the real image shows the
ball, if it's there — measured: offset 2 in every case):

```
target   = 1 if the REAL frame shows a ball (ball_pixels(x_t) ≥ 30), else 0
L_reveal = binary cross-entropy( ball(D(z_t)), target )      averaged over those frames
```

`ball(img)` is a soft version of our evaluation detector, so it can be differentiated:

```
yellowness(pixel) = min(R, G) − B                          # the ball is yellow, cups/floor aren't
soft_count        = Σ_pixels sigmoid( (yellowness − 0.15) / 0.05 )
ball(img)         = sigmoid( (soft_count − 15) / 5 )        # ≈1 when the ~30+ ball pixels appear
```

The target comes from the real pixels, so no hidden-state label is ever used.

### L_recon — stay readable (keeps states inside the dictionary)

While observing, the decoded state should look like the frame actually seen:

```
L_recon = mean over observed frames of ‖ D(z_t) − x_t ‖²
```

During the hold the ball is hidden, so this carries no ball information. Its job is to keep
every state, not just the final one, in the region the frozen decoder understands.

### L_twin — stay honest

```
L_twin = mean over all steps of ‖ z_t − z̃_t ‖²        (z̃ is a fixed target: no gradient into the twin)
```

The online updater may move, but only as fast as the twin can follow. A sudden jump to an odd
state that fools the decoder costs a lot here.

### L_fam — optional, stay familiar

Distance from each rolled state to its nearest state in a bank of stage-1 states. Last run
showed wrong states were *not* far from familiar ones, so this won't fix errors. It's included
only as an extra guard against decoder-fooling, and only if the diagnostics show fooling.

### Total

```
L = 1.0 · L_reveal  +  w_recon · L_recon  +  w_twin · L_twin   (+ w_fam · L_fam)
```

Starting values: `w_recon = 0.1`, `w_twin = 1.0`. Learning rate 3e-5 (it's a finetune),
3,000 steps, batch 16 episodes grouped by length.

## 6. Experiments

Start from the **BASELINE (unaligned)** stage-1 model: it fails at composition and uses no
labels, so it's the cleanest test. Repeat the winner on **ALIGN-DTW** (also label-free).

| arm | what | question it answers |
|---|---|---|
| **C0** | stage-1 model, no finetune | reference |
| **C1** | stage-1 retrained from scratch on S1 + FT, standard SMT | **"is it just more data?"** — your objection, as a control |
| **F1** | finetune, `L_reveal` only | does the reveal signal alone fix new sequences? |
| **F2** | `L_reveal + L_recon` | does staying readable matter? |
| **F3** | `L_reveal + L_recon + L_twin` | **the full method** |
| F4 | F3 + `L_fam` | only if F3 shows decoder-fooling |

C1 is the important control. The reveal bit is the same information the stage-1 reward head
already gets. If finetuning only matches retraining on more data, the method adds nothing.

## 7. What we measure

1. **Pixel eval on TEST** (primary — it's what the decoder-based loss optimises): F1 by depth,
   split into *seen in S1* vs *novel to both*, plus five swaps.
2. **Onset probe** via the frozen reward head (secondary — it wasn't trained on the new states,
   so read it with that caveat).
3. **Composition vs drift split** — does the belief land right when the swaps finish?
4. **History-leakage table** — did the representation change, or only the outputs?
5. **Decoder-fooling check.** Before finetuning, fit a simple probe for the ball's position on
   stage-1 states. After finetuning, compare its verdict with the frozen decoder's on TEST. If
   the decoder says "ball" and the probe disagrees, states have left the meaningful region —
   turn up `w_twin` or add `L_fam`.
6. **Twin disagreement** over training, as an early warning for the same thing.

## 8. Predictions, written before running

- F3 beats C0 on *novel-to-both* sequences at depth 4, and on five swaps.
- **F3 beats C1.** This is the claim that matters.
- F1 alone improves the FT set more than TEST — it memorises FT sequences.
  F2/F3 transfer better, because the frozen dictionary constrains *how* it learns.

**Kill criteria.**
- **F3 ≈ C1** → it's just data. Stop.
- **F3 improves FT but not novel TEST sequences** → memorising the finetuning set.
- **Decoder-fooling check fails at every `w_twin` that still learns** → the frozen-decoder
  anchor isn't trustworthy enough.

## 9. Risks

- **The signal is thin:** one reveal per episode, and only about a third of reveals show a ball.
  The difference from stage 1 is that it now lands on the chain's *own* states on *new*
  sequences — exactly where they fail.
- **Gradients through ~60 steps** can explode or vanish. Mitigations: gradient clipping
  (already 1.0 in stage 1); if needed, truncate to the last 3 swaps plus the slack.
- **The soft ball detector can be gamed** — the decoder draws yellow without a coherent ball.
  The fooling check catches it; the full-image `L_recon` on show frames is the fix.
- **The frozen teacher limits how far the representation can move.** That's deliberate in v1,
  to keep meaning fixed. Unfreezing it is a later step.

## 10. Implementation

| piece | file |
|---|---|
| FT data config (fresh seed), filtered against TEST | `configs/data/memtest4_ft.yaml` + a filter script |
| embedding cache for FT (needed only for C1) | `pixi run cache-embeddings-prelim` on the new zarr |
| finetune loop: differentiable rollout, both twins, the 3–4 losses, EMA update | `scripts/finetune_twin_phase1.py` |
| knobs (`w_recon`, `w_twin`, `w_fam`, EMA momentum, lr, steps) | `configs/smt_phase1/finetune_twin_phase1.yaml` |
| soft ball score | small function next to `src/info_gathering/ball_detector.py` |
| export for evaluation: write the finetuned updater into a copy of the stage-1 checkpoint, plus a decoder checkpoint pointing at it, so `eval-smt-phase1` runs unchanged | inside the finetune script |

**Smoke tests before any real run:**
1. With zero learning rate, the finetuned-and-exported model reproduces C0's pixel-eval numbers exactly.
2. `ball(D(z))` agrees with the hard detector's verdict on stage-1 imagined frames (≥ 99%).
3. Gradient check: only the online updaters receive gradients; teacher, readout and decoder stay at zero.

## 11. Cost

- FT generation + filtering: minutes.
- C1 (retrain on more data): ~40 min stage 1, plus readout and decoder (~1.5 h) for pixel eval.
- Each finetune arm: ~20–30 min, plus ~5 min eval. No readout/decoder retraining — they're frozen.

## 12. Prior work this borrows from

- **Mean teacher** — Tarvainen & Valpola, NeurIPS 2017: a student kept close to an EMA copy of
  itself on unlabelled data.
- **Test-time adaptation** — e.g. TENT (Wang et al., ICLR 2021), TTT (Sun et al., ICML 2020):
  adapting on new unlabelled data while a fixed part of the network anchors meaning.
- **DMT** (in this repo): rolling the updater on its own states. The difference is the target —
  DMT copies the teacher's history-specific state; here the target is "decodes to the truth".
