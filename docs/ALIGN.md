# The alignment loss

What `coef_align` / `coef_sep` actually compute, why they replaced `coef_unif`, and what to read.

Implementation: [losses.py:138-161](../src/world_model/models/smt_phase1/losses.py#L138-L161).
Knobs: [smt_phase1.yaml](../configs/smt_phase1/model/smt_phase1.yaml) under `loss:`.
Results: [SMT_PHASE1_RESULTS.md](SMT_PHASE1_RESULTS.md), section "Alignment arms".

---

## 1. The state it acts on

The memory is `N = 10` tokens of `D_tok = 256`, each rms-normed
([blocks.py](../src/world_model/models/smt_prelim/blocks.py), `normalize_mem`). Flattened,

```
z ∈ R^D,   D = N · D_tok = 2560,   ‖z‖ = sqrt(N · D_tok) = 50.60  (exactly, for every state)
```

So every state lives on a sphere of fixed radius 50.6. Two independent points on that sphere sit
`r·sqrt(2) = 71.6` apart on average — that is the scale against which `align_margin: 20` should be read.
The measured `|z|` column in §7 is 50.60 for every arm, as it must be.

## 2. The eligibility set

Alignment only fires at frames where two states *ought* to be identical:

```python
still    = (e_ctx[:, -1] - e_ctx[:, -2]).abs().amax(-1) < 1e-6   # e_t == e_{t-1}
eligible = still & (batch["meta/phase"] == 1)                    # ... and phase == hold
```

`still` means frame `t` is bit-identical to frame `t-1` in embedding space — nothing moved.
`phase == 1` is the hold: cups down, indistinguishable, ball occluded. At such a frame the
*rendered scene is the same image in every episode*. Whatever distinguishes two eligible states is
therefore carried entirely by the memory, not by the observation.

Measured on `memtest4_big` (50 episodes, 2157 frames): 58.5% of frames are motionless, 53.6% are in
the hold phase, and **21.4% are eligible — 9.2 frames per episode**. Note the pixel-level and
embedding-level stillness tests agree exactly (462 frames either way).

**This is unrelated to `ball_lift_seq`.** That mask marks lift frames among the anchor's 21 *future*
read-outs, and lift frames are `phase == 2` — so every frame the reward head scores is explicitly
*excluded* from alignment. Eligibility is about where the anchor itself sits; the mask is about
where its predictions get scored. An anchor can be both, either, or neither.

Positives are then defined by the *privileged slot label* at the anchor time:

```python
slot_now = batch["slot_dense_seq"][:, 0].long()    # which cup hides the ball at t
S_s      = { i : eligible_i  and  slot_now_i == s }
```

One episode contributes to *different classes at different times*. In the 4-swap episode of §8
(hold 10-45, lift 46-55), the eligible anchors are `t = 10, 14, 20 → class 2`, `t = 26, 32 → class 1`,
`t = 38..45 → class 2`. So `z_26` and `z_32` are a positive pair; `z_20` and `z_38` are a positive
pair spanning a swap out and back; and `z_20` and `z_26`, six steps apart in one episode, are
*negatives*. Across a batch, 98.3% of positive pairs are cross-episode.

**This is supervision, not self-supervision.** Wang & Isola get positives from augmentations of one
datum; we get them from a label the agent cannot see. The honest description of `L_align` is: *a
privileged assertion about which histories are the same state*. §10 says what that buys and costs.

## 3. `L_align` — the pull

```python
align = 0
for s in range(3):
    m = eligible & (slot_now == s)
    if m.sum() > 1:
        mu    = flat[m].mean(0, keepdim=True)
        align = align + (flat[m] - mu).pow(2).mean()
```

`.mean()` runs over **both** the selected rows and the 2560 dimensions, and the three slots are
**summed**, not averaged. So with `n_s = |S_s|` and `μ_s` the batch mean of slot `s`:

```
L_align = Σ_{s : n_s > 1}  (1 / (n_s · D)) · Σ_{i ∈ S_s} ‖z_i − μ_s‖²
```

Three identities worth knowing.

**(a) It is exactly Wang & Isola alignment with α = 2.** Using
`(1/n) Σ_i ‖z_i − μ‖² = (1/2n²) Σ_{i,j} ‖z_i − z_j‖²`,

```
L_align = Σ_s  (1 / 2D) · E_{i,j ∈ S_s} ‖z_i − z_j‖²
```

which is their `L_align(α=2) = E_{(x,y)~p_pos} ‖f(x) − f(y)‖^α`, averaged over all intra-slot pairs,
summed over slots, scaled by `1/2D`. The mean-based form is used because it is O(n) rather than
O(n²) per slot.

**(b) It is exactly center loss with a detached center.** The gradient of the mean through `μ_s`
cancels identically:

```
∂L/∂z_k = (2/n)(z_k − μ) + (1/n) Σ_i 2(z_i − μ)·(−1/n)
        = (2/n)(z_k − μ) − (2/n²) Σ_i (z_i − μ)
        = (2/n)(z_k − μ)                          since Σ_i (z_i − μ) = 0
```

So not detaching `mu` costs nothing and changes nothing. Wen et al.'s moving-average center is an
alternative only if batches are too small to estimate `μ_s` — ours carries ~8 eligible anchors per
slot at `batch_size: 128`, which is thin. That is the main numerical weakness of this term.

**(c) It exerts no force whatsoever on the class means.** `L_align` is invariant to translating a
whole class: replace every `z_i, i ∈ S_s` by `z_i + v` and both `z_i` and `μ_s` shift by `v`. This
matters in §5.

## 4. `L_sep` — the hinge

```python
margin = cfg.get("align_margin", 20.0)
sep = sum(relu(margin - ‖μ_i − μ_j‖)² for i<j) / n_pairs
```

```
L_sep = (1 / |P|) · Σ_{(i,j) ∈ P}  max(0, m − ‖μ_i − μ_j‖)²,    m = 20,  |P| = 3
```

One-sided: zero gradient once the means are `m` apart, quadratic penalty below. This is the
repulsive branch of the Hadsell–Chopra–LeCun contrastive loss, applied between class *means*
instead of between individual pairs — cheaper, and much lower variance with ~8 samples per class.

It replaces uniformity because uniformity has the wrong sign here (§5).

## 5. Why uniformity was removed

SMT's `uniformity_loss` (a direct port of `smt/src/model.py`) is

```
L_unif = log  E_{i≠j}  exp(−t · ‖ũ_i − ũ_j‖²),    t = 2
```

over ℓ2-normalised vectors, on a random 512-subsample. Wang & Isola prove its minimiser is the
uniform distribution on the hypersphere, and that `L_align + L_unif` is the asymptotic limit of
InfoNCE. Two problems in this setting.

**It pushes apart the very states alignment merges.** `L_unif` is taken over *all* pairs, including
two hold-frame states that share a slot. Those are exactly the pairs `L_align` pulls together.
Running both is a tug-of-war with no fixed point at the geometry we want.

**It operates on the wrong unit.** `z.reshape(-1, z.shape[-1])` flattens `(B, N, D_tok)` to
`(B·N, D_tok)`, so each of the 10 memory tokens is a separate sample — *including the 10 tokens of
one state*. Uniformity therefore spends part of its budget pushing apart a single state's own
tokens. `L_align` operates on the flattened 2560-vector, i.e. on states. They are not commensurable.

The measurement (§7) shows the effect directly: within-slot spread is 19–30 with uniformity on,
12.7 with it merely off, and 2.6–2.9 with alignment in its place.

## 6. Scales, and where the equilibrium comes from

`L_align` divides by `D = 2560`; `L_sep` is a raw norm and does not. That is a factor-2560 mismatch,
which is why `coef_align: 0.1` sits next to `coef_sep: 1e-3`. Concretely, per slot,
`L_align ≈ within²/D`:

| | within | `L_align` (×3 slots) | ×`coef_align` |
|---|---|---|---|
| at init | ~30 | 1.05 | 0.105 |
| ALIGN-LIFT converged | 2.94 | 0.010 | 0.001 |

against a reward BCE of ~0.69 at init — so about a sixth of the reward term early, and two orders
of magnitude below it once converged. Alignment does its work in the first few thousand steps and
then stops competing.

**Where the three clusters end up is decided entirely by `L_sep`.** By §3(c), `L_align` cannot move
a class mean. `L_reward` / `L_post` / `L_prior` have no opinion about inter-class distance either.
So `L_sep` is the only term acting on `μ_s`, and being a *one-sided* hinge it pushes until
`‖μ_i − μ_j‖ = m` and then releases. The prediction is that between-class distance parks at exactly
`align_margin`. It does:

```
ALIGN-DENSE   20.71  20.92  20.59      margin 20
ALIGN-LIFT    19.51  20.43  20.99      margin 20
```

Three near-equal sides — a 3-point simplex in the 2560-dim sphere, with within-class variance
driven toward zero. That is neural collapse (Papyan–Han–Donoho: NC1 within-class variance → 0,
NC2 means → simplex ETF), induced deliberately rather than emerging at the terminal phase of
training. It is only a *partial* collapse: a full ETF of three *centred* means on this sphere would put them
`r·sqrt(3) = 87.6` apart, and `L_post` / `L_prior` still require the state to retain enough
non-ball information to be one-step predictable.

The depth-1 arm is the exception — 31.7 / 22.0 / 25.5, above the margin — because with only 0–1
swaps in training there is less history to merge, so the natural separation already exceeds the
hinge and `L_sep` never binds.

## 7. Measured geometry

Memory states at eligible frames, 50 episodes of the arm's own training set. `within` is the RMS
deviation from the slot mean (what `L_align` minimises); `between` is the three pairwise distances
between slot means (what `L_sep` hinges). Script: [scripts/state_geometry.py](../scripts/state_geometry.py) (`ARMS='{"name": "<path>/wm_latest.pt"}' pixi run python scripts/state_geometry.py`).

| arm | within | between (0-1, 0-2, 1-2) | ‖z‖ | between/within |
|---|---|---|---|---|
| baseline (unif, obs decoder) | 30.08 | 10.58  19.65  23.40 | 50.60 | 0.59 |
| NOOBS (unif, binary head) | 19.08 | 14.25  13.63  5.65 | 50.60 | 0.59 |
| slot-dense + unif | 26.73 | 24.35  25.39  26.29 | 50.60 | 0.95 |
| slot-dense, no unif | 12.66 | 12.48  13.85  12.86 | 50.60 | 1.03 |
| **ALIGN-DENSE** | **2.64** | 20.71  20.92  20.59 | 50.60 | **7.87** |
| **ALIGN-LIFT** | **2.94** | 19.51  20.43  20.99 | 50.60 | **6.90** |
| ALIGN-EMBED | 9.04 | 9.22  12.34  12.82 | 50.60 | 1.27 |
| depth-1 baseline | 30.77 | 33.57  36.17  35.41 | 50.60 | 1.14 |
| **depth-1 ALIGN-LIFT** | **1.30** | 31.74  21.98  25.46 | 50.60 | **20.29** |

The ratio column tracks task performance almost monotonically. Arms below ~1.1 fail to extrapolate;
the two above 6.9 reach 0.94–0.99 imagined pixel F1; the one at 20.3 is perfect (§9).

Note ALIGN-EMBED: same `coef_align`, same margin, yet within stays at 9.04 and the means never reach
the margin. Alignment alone cannot compress a state whose only other gradient is reconstruction —
a hidden ball contributes nothing to the frame it is hidden in, so nothing else in that objective
distinguishes the three classes and alignment has to fight the reconstruction term for the whole
budget.

## 8. What the lift head actually supervises

ALIGN-LIFT's reward target is one bit per episode — "is the ball under the cup that gets lifted" —
masked everywhere except the 9 lift frames, where it is constant. With `fut_len: 20` the decoder
emits 21 read-outs `r_t .. r_{t+20}` per anchor, and a read-out is scored only where it lands on a
lift frame. A real 4-swap episode, every anchor (`.` = masked):

```
   t   ball_seq (r_t .. r_t+20)   ball_lift_seq (mask)      supervised
   0   .....................     .....................       0
  ...       (27 anchors with no signal at all)
  26   .....................     .....................       0
  27   ....................0     ....................1       1
  28   ...................00     ...................11       2
  ...
  35   ............000000000     ............111111111       9
  ...
  47   000000000............     111111111............       9
  ...
  54   00...................     11...................       2
```

Right-clamped positions past the episode end are removed by `& seq_mask`, so the final lift frame
is not double-counted. Measured over the training set: **65.3% of anchors carry any signal at all,
and the mean anchor is supervised at 4.15 of its 21 positions.** In that 56-step episode, anchors
`t ≤ 26` get nothing — the lift is more than 20 steps away — so only 50.9% of its anchors are
supervised.

This biases the head by depth. A 0-swap episode is ~30 steps, its lift starts at `t ≈ 21`, and
**96.6% of its anchors are supervised**; the 4-swap episode above gives 50.9%. The reward signal is
therefore concentrated on exactly the episodes that need the least memory — another reason it is
not what drives depth generalisation.

**And for much of the rest the bit is not predictable in principle.** ALIGN-LIFT sets
`decoder_obs: false`, so the decoder's input is `[memory z_t ; a_t .. a_{t+19}]` — no frames. The
action is two-dimensional and encodes the *phase*, nothing else:

```
reveal [0,0]     hold [1,0]     show/lift [0,1]
```

It is constant `[1,0]` for the entire hold, swaps included. So the future actions carry **zero**
information about which swaps happen after `t`, and the decoder cannot see them either. The bit is
a deterministic function of `z_t` only when no swap lands after the anchor. Measured over supervised
anchors:

| swaps landing after `t` | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| share of supervised anchors | **62.4%** | 20.6% | 15.2% | 1.8% |

For the other 37.6% the target is irreducible noise: the remaining swaps are random and unobserved,
so the Bayes-optimal output is the marginal and the BCE floor is that marginal's entropy. Those
anchors cannot teach `z_t` anything, because `z_t` genuinely does not contain the answer.

Net: the head delivers usable gradient on roughly `0.653 × 0.624 ≈ 41%` of anchors, one bit each.
That is the quantitative content of the config's claim that alignment is "the main compression
pressure in this arm" — and it is why the depth-1 extrapolation result (§9) is a statement about
`L_align`, not about the head.

Note the contrast with `decoder_obs: true` (the baseline arm): there the decoder also receives
`e_fut`, the future frame embeddings, which *do* show the swaps happening and eventually show the
ball itself. That decoder can compose the swaps on its own, so the memory is not forced to — which
is what the no-obs arms exist to prevent, and why `reward_acc_first` is tracked separately.

Alignment is unaffected by any of this. `slot_dense_seq[:, 0]` is the slot *at the anchor time*, so
it is well-defined and noiseless at every eligible anchor regardless of what happens later.

## 9. What it bought

Full tables in [SMT_PHASE1_RESULTS.md](SMT_PHASE1_RESULTS.md). The headline, imagined pixel F1 on
`memtest4_eval`:

| arm | all (315) | 4 swaps UNSEEN (34) | 5 swaps (63) |
|---|---|---|---|
| P1-500-dense (uniformity) | 0.960 | 0.741 | — |
| P1-500-nounif | 0.888 | 0.471 | — |
| ALIGN-DENSE | 0.987 | 1.000 | 0.880 |
| ALIGN-LIFT | 0.939 | 0.960 | 0.706 |
| ALIGN-EMBED | 0.277 | 0.125 | 0.222 |

The 5-swap pixel eval was only run for the three alignment arms; for the uniformity arms the
comparable number is the onset probe (prior), 0.667 and 0.533.

And the extrapolation result — ALIGN-LIFT trained on **0–1 swaps only**, evaluated on depths it has
never seen, ball-under-lifted-cup from the posterior chain:

```
memtest4_eval    0 sw  63/63    1 sw  63/63    2 sw  63/63    3 sw  63/63    4 sw  63/63
memtest4_eval5   5 sw  63/63
                 378 episodes, F1 1.000, zero errors
```

The mechanism this suggests: collapsing histories into three tight classes turns "compose a swap"
from a function on a high-dimensional manifold into a permutation of three points, and a permutation
of three points composes for free. The depth-1 model never saw a composed sequence and still
composes five.

## 10. Known weaknesses

- **~8 anchors per slot per batch.** `μ_s` is a noisy estimate. A moving-average center (Wen et al.)
  or a larger batch is the obvious fix; untested.
- **The label is privileged.** `slot_dense_seq` is ground truth the agent has no access to. Nothing
  here is a self-supervised result. The interesting question is what supplies the equivalence
  classes without it — §12's bisimulation references are the place to look.
- **The margin is a free parameter** and the geometry parks exactly on it (§6), so `align_margin`
  directly sets the class separation. 20 was picked once and never swept.
- **Datasets without dense labels fail loudly.** `memtest4.zarr` (250 eps) has x-based mid-swap
  labels; the loader withholds `slot_dense_seq` for it, so alignment arms need
  `memtest4_big` / `memtest4_depth1`.
- **The residual failure mode is drift, not composition.** Every arm composes the swaps correctly and
  then wanders during the motionless slack before the lift. Alignment shrinks the clusters but does
  not pin the state in time; nothing in `L_align` says "and stay there for 12 frames".

## 11. Removing the label: future-trajectory DTW

Every arm above defines positives with `slot_dense_seq`, ground truth the agent cannot see. This
section replaces that with a criterion computed from the cached embeddings alone.

**The criterion.** Two rest states are equivalent if their FUTURES coincide. Distance between
futures is DTW over the embedding trajectory from `t` to the end of the episode, path-length
normalised, on a 64-D PCA of the cached embeddings (which retains 1.0000 of the variance -- the
embeddings live in a <=64-D subspace). Kernel: [dtw.py](../src/world_model/utils/dtw.py), verified
against a naive reference to 1.8e-7 relative error. Matrix: `scripts/build_dtw_matrix.py`,
4672 x 4672 over every rest anchor of `memtest4_big`, 24 s to build, 44 MB.

**Why DTW and not time-aligned L2.** The slack length varies, so two futures with the same swap
sequence are misaligned in time. Over 399k cross-episode pairs:

| | time-aligned L2 | DTW |
|---|---|---|
| exact-match rate | 2.97% | **22.3%** |
| matched pairs joining DIFFERENT swap depths | 0.786 | 0.786 (max gap 4) |
| 1-vs-2 pull, all rest frames | 0.96x | 1.00x |
| 1-vs-2 pull, restricted (below) | 0.39x | **0.24x** |
| same-slot match rate, restricted | 0.003-0.005 | **0.031-0.038** |

DTW matches 7.5x more pairs AND discriminates better. Every DTW match shares the lift answer and
the remaining-swap count, so warping never stretched a k-swap future onto a (k+1)-swap one.

**The restriction that makes it work.** Applied to all rest frames the criterion is useless: it
reconstructs `{0}` vs `{1,2}`, i.e. ALIGN-TARGET, because most eligible frames sit in the
post-swap slack where two different non-target slots have genuinely identical futures. Restricted
to anchors whose future still contains MOTION, all three cross-slot pairs separate:

| slot pair | P(match), all frames | P(match), motion ahead |
|---|---|---|
| 0 vs 0 | 0.4337 (1.00) | 0.0306 (1.00) |
| 1 vs 1 | 0.3971 (0.92) | 0.0342 (1.09) |
| 2 vs 2 | 0.4024 (0.93) | 0.0375 (1.14) |
| 0 vs 1 | 0.0011 (0.00) | 0.0091 (0.30) |
| 0 vs 2 | 0.0012 (0.00) | 0.0103 (0.34) |
| **1 vs 2** | **0.3991 (0.92)** | **0.0088 (0.26)** |

The filter is itself observable -- "does the future contain motion before the lift", read off the
embeddings -- and agrees with the schedule-derived version on 98.5% of anchors. It is a sampling
filter, not information handed to the encoder.

**Attract, never repel.** The obvious formulation, metric regression
`(||z_i - z_j|| - lambda*dtw_ij)^2`, MEASURED TO FAIL:

```
same-slot pairs  mean dtw 9.46  -> target distance 18.9
diff-slot pairs  mean dtw 9.64  -> target distance 19.3
separation in the target: 0.36   (1.9% of its magnitude)
96.8% of same-slot pairs are pushed APART, median target 19.4
```

The signal is in the TAIL (`P(dtw~0)` differs 3.5x), not in the mean. Regressing on the mean
spends 98% of the objective encoding the upcoming SCHEDULE, which `z_t` cannot know. The correct
reading is that the criterion is **sound but incomplete**: `dtw ~ 0` implies equivalence, a large
`dtw` implies nothing, because the futures may have diverged only because the schedules differed.
So it must attract:

```
w_ij = exp(-dtw_ij / tau),  tau = 2
L    = sum_ij w_ij ||z_i - z_j||^2 / D  /  sum_ij w_ij      # over motion-ahead pairs
```

and repulsion has to come from `coef_unif`, restoring Wang & Isola's decomposition in its
original form with positives supplied by future-match rather than by augmentation.
`align_dtw_mode` selects `attract` (default) or `regress` (kept so the negative result is
reproducible).

**Setting the coefficient.** Measured, not guessed: raw encoder-gradient norm per term, times its
coefficient. ALIGN-LIFT's total alignment pressure is align 0.163 + sep 0.918 = 1.08. The regress
form's raw norm is 2333 (it is a squared distance, not a per-dimension mean), so `coef_align`
5e-4; the attract form's is 2.11 and is per-dimension like the slot arms, so `coef_align` 0.2.

**Result, stage 1 (regress arm).** Centroid geometry looks bad and is misleading -- the ball lives
in a subspace, so `within`/`between` over all 2560 dimensions misses it:

| arm | label | within | between (1-2) | kNN purity | linear probe | dims eta^2>0.1 |
|---|---|---|---|---|---|---|
| ALIGN-LIFT | 3-way slot | 2.94 | 20.99 | 0.999 | 0.996 | 2530 |
| ALIGN-TARGET | 1 bit | 6.76 | 1.90 | 0.819 | 0.863 | 1465 |
| **ALIGN-DTW** | **none** | 11.76 | 4.92 | **0.875** | **0.919** | **588** |
| chance | | | | 0.335 | 0.333 | |

**With no privileged label at all, ALIGN-DTW carries more slot information than the arm given the
privileged lift bit**, in a 2.5x more compact subspace. Downstream eval pending.

## 12. Reading

Ranked by how directly they bear on this loss.

1. **Wang & Isola (2020), "Understanding Contrastive Representation Learning through Alignment and
   Uniformity on the Hypersphere", ICML.** [arXiv:2005.10242](https://arxiv.org/abs/2005.10242).
   The source. §3 gives the `L_align(α)` / `L_unif(t)` decomposition, the proof that `L_unif`'s
   minimiser is the uniform distribution on the sphere, and the InfoNCE asymptotic limit. Read §3–4.
   Our `L_align` is their α=2; our `L_sep` is what we swapped in for their `L_unif`.

2. **Hadsell, Chopra & LeCun (2006), "Dimensionality Reduction by Learning an Invariant Mapping",
   CVPR.** The `max(0, m − d)²` hinge, in its original form. `L_align + L_sep` is precisely their
   two-branch contrastive loss, with the repulsive branch lifted from pairs to class means.

3. **Wen, Zhang, Li & Qiao (2016), "A Discriminative Feature Learning Approach for Deep Face
   Recognition", ECCV.** Center loss — `Σ ‖z_i − μ_{y_i}‖²`. §3(b) shows our alignment term *is*
   this. Read for the moving-average center, which is the fix for our thin per-batch class counts.

4. **Khosla et al. (2020), "Supervised Contrastive Learning", NeurIPS.**
   [arXiv:2004.11362](https://arxiv.org/abs/2004.11362). Positives defined by a *label* rather than
   by augmentation — our setting exactly. Their SupCon-out/SupCon-in analysis is the right frame for
   asking whether the mean-based form loses anything against the full pairwise one.

5. **Papyan, Han & Donoho (2020), "Prevalence of Neural Collapse during the terminal phase of deep
   learning training", PNAS 117(40).** [arXiv:2008.08186](https://arxiv.org/abs/2008.08186).
   NC1–NC4. §6's equilateral triangle with vanishing within-class variance is a deliberately induced
   NC1+NC2, which is why the geometry vocabulary here is theirs.

6. **Zhang, McAllister, Calandra, Gal & Levine (2021), "Learning Invariant Representations for
   Reinforcement Learning without Reconstruction" (DBC), ICLR.**
   [arXiv:2006.10742](https://arxiv.org/abs/2006.10742). The RL framing: what we do by fiat with a
   privileged label — declare which histories are the same state — DBC *learns* from a bisimulation
   metric. The most direct answer to §10's "what supplies the equivalence classes without the label".
   Background: **Ferns, Panangaden & Precup (2004), "Metrics for Finite Markov Decision Processes",
   UAI** for bisimulation metrics, and **Li, Walsh & Littman (2006), "Towards a Unified Theory of
   State Abstraction for MDPs", ISAIM** for the abstraction taxonomy (ours is a
   model-irrelevance abstraction, asserted rather than derived).

7. **Kumar & Isola (2026), "Pretraining Recurrent Networks without Recurrence" (SMT).**
   [arXiv:2606.06479](https://arxiv.org/abs/2606.06479), [submodules/smt](../submodules/smt).
   The architecture these losses sit inside, and the origin of the `uniformity_loss` we removed
   (`src/model.py`, `calc_uniformity_loss`). §5 is the argument for why their default is wrong for
   this task, not wrong in general.

8. **Bardes, Ponce & LeCun (2022), "VICReg: Variance-Invariance-Covariance Regularization for
   Self-Supervised Learning", ICLR.** [arXiv:2105.04906](https://arxiv.org/abs/2105.04906).
   An alternative anti-collapse term: a hinge on per-dimension *variance* rather than on
   between-class distance. Worth trying against `L_sep` — it is parameter-free in the way
   `align_margin` is not.

Peripheral but relevant: **van den Oord, Li & Vinyals (2018), CPC**,
[arXiv:1807.03748](https://arxiv.org/abs/1807.03748), for InfoNCE itself;
**Grill et al. (2020), BYOL**, [arXiv:2006.07733](https://arxiv.org/abs/2006.07733), for
collapse avoidance with no repulsive term at all; **Hafner et al. (2023), DreamerV3**,
[arXiv:2301.04104](https://arxiv.org/abs/2301.04104), for the KL-balancing asymmetry that
`detach_prior_target` / `detach_prior_input` reproduce on vectors.
