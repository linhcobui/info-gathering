# OGBar-TENET: the next OGBar world model, built from "On Latent Beliefs"

**Status (2026-10-26): plan only, nothing built.** This doc reviews the note *On Latent Beliefs* (Bui, Nakamura) and lists what has to change in [OGBAR_SMT.md](OGBAR_SMT.md) to follow it. Two user decisions are already fixed:
- **The posterior updater reads only z_t, a_t, o_{t+1}.** The current frame is no longer an input (OGBAR_SMT.md's `updater_obs_input: current` is dropped).
- **There is no prior.** P, the read-out R and everything about imagination go.

| Doc | Covers |
|---|---|
| [OGBAR.md](OGBAR.md) | the environment, the play policy and the dataset |
| [OGBAR_SMT.md](OGBAR_SMT.md) | the previous design (frame-predicting SMT with a stochastic prior) and its four runs |
| memtest4 splice / Align-Walk results (memory note, 2026-10-21/22) | TENET's predecessor on the shell game |

## Symbols

| symbol | meaning |
|---|---|
| t | model step (frameskip 5: one step = 5 raw steps, 200 steps per episode) |
| o_t | the observation at step t: frozen DINOv2 tokens of the side-view frame |
| a_t | action from t to t+1: 5 raw 5-D actions concatenated, 25 numbers |
| r_t | reward at step t (defined in §3.2) |
| h_t | history o_1, a_1, …, a_{t−1}, o_t |
| E | teacher encoder: z_t = E(h_t) |
| f_post | posterior updater: z_{t+1} = f_post(z_t, a_t, o_{t+1}) |
| D | reward decoder: D(z_t, a_t … a_{t+j−1}) = probability that r_{t+j} = 1 |
| c | the bar's hidden centre-of-mass offset (±12.6 cm). Used only for scoring. |
| s | grasp point along the bar, measured from its middle |

## 1. What changes from OGBAR_SMT.md

| part | OGBAR_SMT.md (2026-10-15) | this plan |
|---|---|---|
| posterior | Q(z_t, x_t, a_t, x_{t+1}), Gaussian sample | **f_post(z_t, a_t, o_{t+1})**, deterministic (§3.4) |
| prior P, read-out R | Gaussian NLL prior; R draws frames for imagination | **removed** |
| decoder target | every patch of all 200 future frames | **future reward** r_{t+1} … r_{t+200} |
| decoder input | z\*_t plus the future frames and actions (paired layout) | z\*_t plus **future actions only** (§2.1, decision A) |
| teacher | causal over steps, all 256 patches, one pass per episode | bidirectional over the window, as the note writes (§3.3, decision D) |
| training | one stage | three stages: memorise, find equivalent pairs (TENET), retrain with alignment |
| what z must hold | only what the frame can't show | everything reward needs, the visible scene included, since no updater or decoder sees the current frame |

Flags that disappear: `updater_obs_input`, `z_dist`, `prior_mixture`, `sigma_min`/`sigma_max`, `beta_nll`, `coef_prior`, `decoder_loss`, `decoder_steps_scored` (replaced by a reward-horizon flag). The "z holds only the belief" probe in the evaluation table flips meaning: z now *should* hold bar and gripper pose.

## 2. Review of the note

### 2.1 The main problem for OGBar: future frames show the COM before it affects reward

On the shell game the hidden bit and the reward arrived together: the lift that pays out is the same frame that shows the ball. So a decoder that saw future frames still needed the memory, and the memory note says never to build a decoder without future frames there.

On OGBar the order is different. The COM changes the reward through physics that the camera sees first. A bar gripped off its COM tilts as it leaves the floor, and only seconds later does the slide-in or smash fail. So:
- **A decoder that sees future frames can read the outcome off the tilt** and never needs z. This is the same failure as OGBAR_SMT.md's run 4, where the memory gain was about 0, with reward in place of patches.
- **A TENET walk through episode i's real frames hits the same problem.** If the walk from i−k to i passes the lift, f_post learns the tilt from those frames, so ẑ passes the test whatever z_j believed. Every candidate becomes a positive pair, and beliefs with different c get aligned.

The note already writes the decoder as D(z_t, a_fut), with no future observations, and desiderata (1)–(3) condition on a_fut only. **For OGBar I recommend keeping it that way** (decision A). Then TENET should match it:
- walk f_post through episode i's real frames only while the bar is on the floor;
- from the first frame where the bar is lifted, hand the rest of the test to D as actions only;
- score the pair on D(ẑ, a_rest).

This is the opposite of the memtest4 rule, for an OGBar-specific reason, so it's your call.

### 2.2 The current data leaks the COM and the outcome through the actions

With an action-only decoder, the actions must not encode c or the outcome. The scripted play policy is closed-loop on the hidden state. Measured on `ogbar_v3_400` (2026-10-26):

| segment | count | grasp within 1 cm of the true COM | seated |
|---|---|---|---|
| slide-in | 273 | **249** (it grips at the true COM unless ≥ 2 probes came first) | 170 |
| smash | 266 | 17 (grasp point random, ±16 cm) | 32 |

Smash seat rate by grasp distance from the true COM, |s − c|: 0–2 cm 0.48 (n = 40), 2–4 cm 0.21 (34), 4–6 cm 0.15 (41), above 6 cm **0 of 151**.

What follows:
- **Slide-ins carry almost no COM signal in their reward.** They grip at the COM, so a slide-in fails from noise, not from c (0.67 seated even within 1 cm). Their grasp point also hands D the value of c.
- **Smashes are the only place where reward depends on c given the actions.** That's 32 seats in 400 episodes.
- **More leaks:**
  - the slide-in waits up to 30 steps for the bar to hang level, so the wait length encodes the tilt;
  - it releases "when within 3 mm of seated or stalled";
  - a slide-out runs only if the bar seated;
  - retries follow slips.

  A decoder that is causal over actions (r_{t+j} sees only a_t … a_{t+j−1}) removes the slide-out leak, but not the others.
- **The note's coverage assumption fails:** off-COM slide-ins hardly appear, so their reward outcomes are never in the data.

**This needs new data** (decision C, §4 step 1).

### 2.3 TENET on continuous scenes: "o_j = o_{i−k}" almost never holds

On memtest4 the match was exact: the last w+1 frames and actions equal. On OGBar the bar's pose and the arm's pose are continuous and random, so two episodes almost never share a frame. Both the walk and D(ẑ, a_rest) also need the scene to match, or episode i's actions mean something else in episode j.

Two sources of matches:
- **Within an episode.** A probe puts the bar back where it was, so the frames just before and just after a probe nearly match, with different beliefs: before the probe c is unknown, after it c is narrowed.
- **Across episodes, by design.** The play policy could return the arm to a fixed home pose between segments, and draw bar start poses from a small fixed set. Both are in the v4 data proposal.

Measure before building (§4 step 1): the distribution of nearest-neighbour DINO distance across episodes against within-episode near-duplicates.

### 2.4 One passing test is weak evidence of equivalence

Desideratum (2) asks for equal expected reward under *all* futures. TENET checks one test. With a continuous c this matters: a smash at grasp point s seats whenever |s − c| is under about 5 cm. So beliefs c = 0 and c = 4 cm both pass a test with s = 2 cm, and a pre-probe "c unknown" belief can pass by luck.

Suggested tightening:
- call a pair positive only if it agrees on every matched test available, at least M of them, taken from different episodes;
- call it negative if it disagrees on any.

The memtest4 lessons also carry over:
- use two stage-1 seeds and both must agree (precision 0.98 → 0.999 there);
- pairs on mid-motion frames hurt (Align-splice vs Align-splice-still), so start with frames where the bar rests and the arm is still;
- no pair ever involved an uncertain belief, and calibration on uncertain starts did not improve.

### 2.5 Smaller points on the note

1. **f_prior** is still defined in §2. Delete it per the no-prior decision. The f_posterior line already has the right inputs (z_t, a_t, o_{t+1}).
2. **The letter t means both time and a test.** Rename tests, e.g. q, and index them from the history's end: q = (a_t, o_{t+1}, …, a_{t+N−1}, o_{t+N}).
3. **The test probability.** P(t|h) = P(ht)/P(h) also counts how likely the test's actions are. The PSR definition (Littman et al.) is P(o_{t+1:t+N} | h, a_{t:t+N−1}): the observations given that the actions are taken.
4. **The PSR-to-belief direction is too strong.** P(K|h) fixes every prediction b(s|h) does, but not b itself: states that no test can tell apart are merged. So the PSR is a function of the belief, smaller or equal in size. That is fine, since it is the minimal object you want, but "≅" should say that.
5. **`\label{eq:ztarget}` sits in an `equation*`**, so `\eqref{eq:ztarget}` in §5 prints "??". Also, the displayed line says P(Q|h) where the argument just above it is about K.
6. **Which futures define equivalence.** Desiderata (1)–(3) use open-loop action futures. TENET's walk uses tests that include observations. These are different equivalences. §2.1 argues for the action-only one on OGBar. Make the note and the algorithm use the same one.
7. **TENET's line 1, "D(z_i) = 1",** leaves out D's actions. Say it is a real reward frame of the data where the stage-1 D, fed the real actions, gives p > 0.99.
8. **Line 3** should say where z_j comes from: the stage-1 teacher E(h_j), as in memtest4.
9. **Line 3 matches a single frame.** Matching only o fixes neither velocity nor the arm's motion. Match the last w+1 frames and actions, as memtest4 did.
10. **"Bidirectional E"** in Methodology 1 contradicts OGBAR_SMT.md's causal one-pass teacher. See decision D.
11. **The footnote "SMT induces multiple tree-like structures"** needs a sentence or a figure reference, or it should go.

## 3. Model

### 3.1 Parts

| part | reads | outputs | used |
|---|---|---|---|
| step tokens | o_s (DINO tokens, see decision E) + a_s | tokens per step | in E and f_post |
| teacher E | window of steps ending at t, + 10 registers | z\*_t (10 × 256, rms-normalised per token) | training; warm-up at eval |
| f_post | z_t, a_t, o_{t+1} | z_{t+1} | training (one step); chained at eval |
| reward decoder D | z\*_t, then a_t … a_{t+H−1} as tokens, causal over steps | p(r_{t+j} = 1) for j = 1..H | training; TENET queries |

H = the reward horizon (flag `reward_horizon`, default 200 = the rest of the episode, cut at its end).

### 3.2 Reward

`success` (`bar_in_slot`) is true on 4.8% of raw frames, in 236 of 400 episodes. As a per-frame reward it counts how long the bar sits seated, and that depends on when the policy pushes it out. Recommendation (decision B):
- **r = 1 on the first raw frame of a seat that lasts at least N raw frames after release** (OGBAR.md's strict label), 0 elsewhere;
- at frameskip 5, a model step's r is the maximum over its 5 raw frames.

The return Σ r is then the number of real seats.

### 3.3 Losses (stage 1)

| term | what it is | reaches | weight |
|---|---|---|---|
| reward | binary cross-entropy of D's p(r_{t+j}) against the real r_{t+j}, mean over j = 1..H and the batch | E, D, step tokens | 1.0 |
| post | MSE between f_post(z\*_t, a_t, o_{t+1}) and z\*_{t+1} (live target, as phase 1) | f_post, E | 0.1 |
| uniformity | per memory token across the batch, as OGBAR_SMT.md | E, step tokens | 0.1 |

Uniformity stays because runs 1–3 collapsed without it, and a sparse reward target makes collapse more likely, not less. Cross-entropy is a proper scoring rule, so D's probabilities are trained to be calibrated (desideratum 3).

**Gradient contract** (assert before training, as `scripts/verify_smt_ogbar.py` does):

| term | must reach | must NOT reach |
|---|---|---|
| reward | E, D, step tokens | f_post |
| post | f_post, E | D |
| uniformity | E, step tokens | f_post, D |

### 3.4 Why f_post becomes deterministic

The Gaussian sample was there for the prior: OGBAR_SMT.md already notes that MSE on a sample drives σ_Q to its floor. With no prior, the sample only adds noise. "Don't know" lives in z's mean and shows up as D's probability, as on the shell game (ALL3U.md).

### 3.5 Teacher input and cost (decision D)

The note's E is bidirectional over the window. With all 256 patches and 200 steps that's 51,400 tokens, which is why OGBAR_SMT.md went causal over steps with one pass per episode. Bidirectional means one pass per anchor.

The tilt probe in OGBAR_SMT.md shows a **4×4 average pool keeps the tilt as well as all patches** (R² 0.937 vs 0.936 on side-on frames). With 16 patch tokens + 1 action token per step, 200 steps are 3,400 tokens: full bidirectional attention per anchor fits.

Recommendation: 4×4 pool in E and D. For f_post, either the 4×4 pool or all 256 patches (decision E): it reads one frame, so the cost is small either way.

## 4. Things to do, in order

**Step 0. Settle the decisions** (§5) and correct the note (§2.5).

**Step 1. Data (blocks everything).**
- 1a. **Play policy v4** in `src/ogbar/play.py`, so that reward-relevant actions depend only on the observable history, never on c or on the outcome:
  - slide-in grasp point independent of c: uniform ±12 cm, or the probe estimate (a function of the history, so allowed);
  - a fixed levelling wait instead of "until within 2° of level";
  - a fixed slide length and release time instead of "release when seated or stalled";
  - keep smashes (they already behave this way); possibly raise their rate;
  - arm returns to a fixed home pose between segments;
  - bar start poses drawn from a small fixed set (for cross-episode matches, §2.3).
- 1b. Gate on a pilot (~50 episodes): seat rate against |s − c| for slide-ins should fall off like the smash table above. Check by hand that no action stream depends on a privileged key.
- 1c. Record (400 episodes, as before; the disk has room for one more fp16 cache only after cleanup, or use int8 per-token-scaled at 39.5 GB) and cache DINOv2. Add a pooled 4×4 cache (16 × 384 per frame, about 5 GB) if decision D goes that way.
- 1d. **Match diagnostic:** nearest-neighbour distance (pooled DINO, last w+1 frames + actions) across episodes vs within-episode probe near-duplicates. Pick the match threshold from this, and count how many TENET candidates exist per reward leaf. If almost none exist across episodes, TENET runs on within-episode matches only.
- 1e. A reward-bearing eval set. `ogbar_v3_eval5` is probe-only and has no reward. Use fixed c values × fixed probe plans, each followed by a smash or slide-in at a fixed s, so the "right answer" depends on what the probes revealed.

**Step 2. Code** in `src/world_model/models/smt_ogbar/`:
- remove P, R and the prior loss;
- f_post without x_t;
- reward decoder D on z\*_t + future actions, causal over steps;
- bidirectional teacher on the chosen tokens;
- a loader that adds r per model step at any raw offset.

Verification: the gradient contract; causality of D (perturbing a_{t+j} leaves p(r_{t+i}) for i ≤ j bit-identical); f_post responds to o_{t+1}.

**Step 3. Stage 1, the memoriser.** Train on all training episodes. Check that it really memorises (train reward F1 near 1, calibration) and that the chained f_post stays on the teacher (closure, h = 1..200). Train a second seed for the TENET twin filter.

**Step 4. Stage 2, TENET pairs** (new `scripts/tenet_pairs_ogbar.py`, after `scripts/splice_pairs_phase1.py`):
- leaves: real reward frames where both seeds give p > 0.99 with the real actions;
- step back k (only through frames where the bar is not lifted, §2.1);
- matches z_j by the 1d threshold;
- walk f_post while the bar is on the floor;
- score D(ẑ, a_rest) on both seeds;
- positive if they agree on all matched tests (≥ M); negative if p < 0.01 on any test.

Score the pairs with privileged c only. A positive is correct if both episodes give the same seat outcome under the test's grasp point (|s − c| below about 4.5 cm for both, or above it for both). Report precision and coverage, as on memtest4.

**Step 5. Stage 3, retrain from scratch with alignment.** Same losses + pair alignment, plus uniformity on negatives (`smt_phase1/losses.py` `_pair_align` with `align_neg: unif`, the Align-Walk recipe), coef 0.1 / 1e-3 as there. Ablations:
- the pair sampler with coef 0 (the "is it just the data?" control);
- still-frame pairs only.

**Step 6. Evaluation** (privileged c only for scoring), on held-out episodes and the 1e set:

| question | measure |
|---|---|
| does the memory pay its way? | memory gain: D's reward cross-entropy with a blank / other-episode z\* minus with the real one, at steps before a grasp. Must be > 0 (it was ≈ 0 in all four OGBAR_SMT runs) |
| does the belief hold c? | linear probe for c on chained f_post states, split by probes seen so far |
| is it calibrated? | reliability of D's p(seat) before a grasp, binned, against the real seat rate; before vs after the first probe |
| does compression generalise? | stage 3 vs stage 1 vs the coef-0 control on held-out episodes and the generalisation axes of OGBAR_SMT.md (longer gaps, more turns, held-out c) |
| no-memory baseline | DINO-WM (history 3) trained on the v4 data with a reward head |

## 5. Decisions needed

| | question | recommendation |
|---|---|---|
| A | Does D see future frames? | No, actions only, as the note writes (§2.1). Reverses the memtest4 rule for an OGBar-specific reason. |
| B | Reward definition | First frame of a seat held ≥ N raw frames after release (§3.2). |
| C | New data (play policy v4) | Yes. The current data leaks c through slide-in actions, and only 32 smashes carry COM-dependent reward (§2.2). |
| D | Teacher tokens | 4×4 pool, bidirectional over the window (§3.5). |
| E | f_post's observation tokens | 4×4 pool to match E; all 256 patches is affordable if wanted. |
| F | Equivalence used by TENET | Action-only tests, agreement on ≥ M matched tests (§2.4, §2.5 item 6). |
