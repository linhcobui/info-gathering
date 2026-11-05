# REVERSE — undo the swaps, return to the start

**Goal.** Make history *unrepresentable* in the belief, without labels. The mechanism is one
physical fact about this world: **every swap is its own inverse.** Doing swap(a,b) twice puts
every cup back. So if the belief watches a sequence of swaps and then watches the same swaps
again in reverse order, it must be exactly where it started. A history record cannot pass that
test, because it only ever grows.

Status: proposal. Nothing below has been run. Background: [DELTA.md](DELTA.md),
[DELTA_RESULTS.md](DELTA_RESULTS.md), [ALIGN.md](ALIGN.md).

---

## 1. The problem this targets

From DELTA_RESULTS.md:

| | posterior unseen F1 | posterior 5 swaps | what the posterior state is organised by |
|---|---|---|---|
| D-base (`L_unif`) | 0.593 | 0.542 | partly slot, partly history |
| D-tok-hi (sparse gate) | 0.400 | 0.489 | **start slot + swap count** (counter in tokens 0/2/5/9) |
| ALIGN-LIFT (privileged slot label) | **0.889** | **0.667** | **ball slot only** |

- ALIGN-LIFT works because its label says *which states are equivalent*: every history that
  ends with the ball in slot s is pulled to one point. Once that holds, each swap has to act as
  a fixed map between three points, and composition follows.
- DELTA constrained *how much* the state changes, which a counter satisfies.
- Uniformity in D-base actively pushes histories apart.

We want ALIGN-LIFT's equivalence structure from something other than the label.

## 2. The idea

Group structure rather than a label. The swaps generate the permutation group S3, where each
swap s satisfies `s·s = id`. Any sequence followed by its reverse is the identity:

```
s1 s2 … sn · sn … s2 s1  =  id
```

A belief that respects this returns to its start state after any such palindrome. What that
rules out:

| code | after `w · reverse(w)` | passes? |
|---|---|---|
| counter / history buffer | has recorded 2n events | **no**: it only grows |
| additive event code `z + Σ v(e)` | `z + 2·Σ v(e)` | **no** |
| start slot kept in a frozen block | unchanged | yes (not addressed here; see §4, data) |
| permutation / slot code | back to start | **yes** |

### Why "reverse order", not "reversed video"

A swap played backwards is **not** a valid observation in this world. `choreography.swap_arc`
always sends the right-moving cup along the front arc. Played backwards, the right-moving cup
would ride the back arc, a frame the encoder has never seen.

What *is* valid is doing the same swap **again**, played forward. Verified on the cached
embeddings: every swap of a given slot pair produces **bit-identical** frame embeddings across
episodes (max difference 0 in 5 of 6 pair × split groups; the sixth at 1.5e-3, float noise).
The mean difference between pairs is 0.095–0.103. So:

- undoing swap k = replaying swap k's own 6 cached frames;
- undoing a sequence = replaying its swap clips in reverse order, each played forward;
- a clip cut from any episode is a real observation in any other, so clips can be **spliced**
  into sequences no episode contains, including depths absent from training.

No labels are needed. Clip boundaries are found by matching the cups-at-rest frame embedding
(a single vector: every rest scene renders identically), and a clip is just the frames
between two rest frames.

## 3. The equations

The update is D-base's (full rewrite, no gate). It is the best-performing unlabelled
formulation, and the gate's write-once behaviour is optional (§3.3).

### 3.1 Forward and replay

```
z_0^F   = teacher(window ending at t)                     # a rest anchor, cups down
z_j^F   = Post(z_{j-1}^F, e_{t+j}, a_{t+j})   j = 1..L    # the real frames: n swaps (+ slack)
clips   = [c_1, …, c_n]                                    # each swap's own frames, cut at rest frames
z^R     = z_L^F;  for c in reverse(clips): for frame e in c: z^R = Post(z^R, e, a_hold)
```
Each clip is replayed **forward**. The action is constant through the hold (`[1, 0]`), so
the replay's actions are the real ones.

### 3.2 The loss

Dense: after undoing swap k, the belief must equal the forward belief just before swap k:

```
L_rev = Σ_k  ‖ z^R_(after undoing k) − sg( z^F_(before swap k) ) ‖²  /  n
```

- The **targets are the forward states, stop-gradded.** The replay is pulled back to the forward
  path; the forward path is never pulled toward the replay, which would let both meet in the
  middle by ignoring the swaps.
- `L_rev = 0` is also achieved by a **blind** state that never changes. That is what `L_reward`
  and the live `L_post` are there to prevent: they force the state to change on swaps, and
  `L_rev` decides *how*. Watch `obs_sensitivity`; if it collapses, raise `coef_post` or lower
  `coef_rev`.

Total:
```
L = L_reward + L_post + L_prior + coef_rev · L_rev        (coef_unif → 0: uniformity pushes histories apart)
```

### 3.3 Optional: make the swap an exact involution by construction

A transposition of slots *is* a reflection: swap(a,b) on a one-hot slot code is
`I − (e_a − e_b)(e_a − e_b)ᵀ`, a Householder reflection. So parameterise the event's
effect on each memory token as a gated reflection:

```
û_k = normalize( U_k(e, a) )          g_k = σ( G_k(e, a) )          # event-only, as in DELTA
z_k ← z_k − 2 g_k · û_k (û_kᵀ z_k)                                  # per token k
```

- With `g ∈ {0, 1}` this is **orthogonal and its own inverse**: `A² = I` exactly. It also keeps
  norms fixed, so no rmsnorm is needed.
- `g = 0` on motionless frames gives DELTA's "nothing changes" for free.
- Push `g` to binary with `g(1 − g)`.
- Caveat: a swap spans several frames, and only the *composite* over a clip needs `A² = I`.
  That holds if the gate writes once per swap, which D-tok-hi learned unprompted. `L_rev`
  covers the general case.

### 3.4 Optional: splice longer palindromes than the data contains

Draw n clips at random from the clip bank (any episodes), splice them after a rest anchor,
and apply the same `L_rev`. This gives the posterior depth-3 and depth-4 sequences on
0–1-swap data, with no teacher target needed for the spliced part. Only the reversal is
asked of it.

## 4. Data: lift a random cup

`L_rev` removes path-dependence and counters. It does **not** remove the start slot: a start
code that every swap transforms consistently passes the palindrome test. The fix is in the
data. Today the lifted cup is always slot 0 (`episode_target = [N, 0, 0]`), so "ball in 1" vs
"ball in 2" is never observed and the minimal sufficient state is not the 3-way slot.

- `memtest4.py`: allow `target_slot: random`, drawing it per episode. It is already stored as
  `episode_target`, and the lift is already "whichever cup stands at the target slot".
- Rebuild `memtest4_big_rl`, `memtest4_depth1_rl`, `memtest4_eval_rl` and `memtest4_eval5_rl`,
  then cache embeddings.
- The lift *action* must not name the target: it stays `[0, 1]`. Otherwise the action leaks the
  answer's location before the cup moves.
- **Risk:** the frozen R2-Dreamer encoder never saw cups 1 or 2 lifted. Check the recon ceiling
  on the new data before training; if it drops, retrain the encoder on the new set.

## 5. Implementation

| file | change |
|---|---|
| `src/info_gathering/dataset_reverse.py` (new) | `SmtReversal`: rest anchors with 1–n swaps ahead; returns the forward frames, the clips (cut at rest-frame matches) and the indices of the before-swap states; a split-level clip bank for splicing |
| `src/world_model/models/smt_phase1/losses.py` | `reverse_loss(posterior, z0, e_fwd, a_fwd, clips, before_idx)`; BPTT over ≤ ~40 posterior steps |
| `src/world_model/models/smt_phase1/updater.py` | `arch.update: rewrite \| gate \| reflect` (§3.3); `rewrite` is the default and is bit-identical to D-base |
| `src/world_model/models/smt_phase1/model.py` | `loss.coef_rev`, `loss.rev_max_swaps`, `loss.rev_splice`, `loss.rev_batch`; a second loader for reversal batches, as DMT has |
| `src/info_gathering/memtest4.py`, `configs/data/` | `target_slot: random` (§4) |
| `configs/smt_phase1/model/smt_phase1.yaml` | every new key with an inert default (quote string values like `"off"`) |
| `scripts/delta_extract.py` / `delta_analyze.py` | add the replay residual and per-pair operator transfer to the report |

Smoke tests:
1. `coef_rev: 0`, `update: rewrite` reproduces D-base's loss bit for bit.
2. Clip cutting: every cut clip in memtest4_big matches its scheduled swap's frames (from the
   seed reconstruction), and the same pair's clips are identical across episodes.
3. Gradient contract: `L_rev` reaches the posterior and, through `z0`, the encoder. It never
   reaches the forward targets (stop-grad) or the reward decoder.
4. `update: reflect` with g forced to 1: applying the same event twice returns `z` to 1e-6.

## 6. Experiments

**Stage A: existing data, cheap.** Isolates the loss.

| arm | data | update | L_rev | splice | question |
|---|---|---|---|---|---|
| R-base | memtest4_big | rewrite | — | — | = D-base, no retrain |
| R-rev | memtest4_big | rewrite | 0.1 | no | does reversal alone remove history? |
| R-rev-splice | memtest4_big | rewrite | 0.1 | yes | do synthetic palindromes help composition on unseen? |
| R-reflect | memtest4_big | reflect | 0.1 | yes | does the exact involution add anything? |
| **R-depth1** | memtest4_depth1 | best | 0.1 | **yes** | **the headline: depth 2–5 from 0–1-swap episodes, no labels** |

**Stage B: random-lift data.** Isolates the data.

| arm | data | question |
|---|---|---|
| D-base-rl | memtest4_big_rl | does the data change alone fix the start-slot memory? |
| best R-* -rl | memtest4_big_rl | loss + data together |
| ALIGN-LIFT-rl | memtest4_big_rl | the labelled reference on the same data |

The dominant cost is the reversal BPTT. Estimate about 1.5× a D-base run (~1 h each on a
shared 4090); Stage A is 4 runs on 2 GPUs.

## 7. Diagnostics

All read only the trained model; labels are for diagnostics only.

1. **Onset probe**: teacher / posterior / prior F1, by depth, seen/unseen, and five swaps.
2. **Replay residual on held-out episodes**: `‖z^R − z^F‖` relative to the slot gap. It should
   be ~0 for R-* and large for D-base.
3. **History leakage**: swaps-so-far and start-slot probes within slot (DELTA_RESULTS.md
   table).
4. **t-SNE of posterior rest states** by slot / swaps / start: success looks like ALIGN-LIFT's
   three clouds.
5. **Operator transfer**: fit one linear map per swap pair on states before → after, test on
   unseen sequences. Group check: `‖M_s M_s − I‖`, and whether `M_01 M_12 M_01 ≈ M_12 M_01 M_12`.

## 8. Predictions, written before running

- R-rev: posterior swaps-so-far probe falls toward its majority baseline; start-slot does not
  (§4).
- R-rev-splice: unseen posterior F1 > D-base's 0.593.
- R-depth1 with splicing: posterior F1 at 2–4 swaps clearly above D-depth1-base (0.32–0.50).
  ≥ 0.9 at five swaps would match ALIGN-LIFT's mechanism without its label.
- Stage B: the start-slot probe falls to majority only on random-lift data.

**Kill criteria.**
- `L_rev` reaches ~0 while `obs_sensitivity` collapses: it found the blind state, and the
  reward/post pressure is too weak to stop it.
- `L_rev` stays high: the posterior cannot represent an involutive update at all. Try
  `update: reflect` before concluding.
- Replay residual ~0 but unseen F1 does not improve: reversibility holds on palindromes yet
  different words for the same permutation still map to different states. Then the braid
  relations (`s01 s12 s01 = s12 s01 s12`, via splicing) are the next term. Those need pair
  identity: three clusters of clip embeddings, label-free, but the S3 law is prior knowledge.

## 9. Risks

- **Blind-state collapse** (above). It is the reason for the stop-grad targets and for
  keeping `L_post` live.
- **BPTT cost and instability** over ~40 steps of an 8-layer updater. Mitigations: truncate to
  `rev_max_swaps` ≤ 3; gradient-checkpoint the posterior.
- **The teacher is not constrained directly.** `L_rev` shapes the posterior; the teacher only
  follows through the live `L_post`. If the teacher keeps history, the posterior must fight
  `L_post` to drop it. Fallback: apply `L_rev` to teacher states too, i.e. teacher(window with
  a palindrome inserted) = teacher(window without it), which is also exact via splicing.
- **Start slot**: needs §4. Expect stage A to leave it in.
- **Encoder on new lifts** (§4).
