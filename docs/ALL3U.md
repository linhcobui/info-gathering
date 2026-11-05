# ALL3-U: can SMT say "I don't know" when the start is never shown?

Started 2026-10-14; first round finished 17:36. Status: **does not pass yet**. Hand-crafted keeps the start-shown result perfect (1.00 to 8 swaps), but the uncertain answers (1/3, 1/2) are not honest enough. See [Results](#results).

## Why

The ALL3 Hand-crafted-backtracking run is perfect up to 8 swaps when the start is shown. When it isn't, it does not say "don't know":
- **Two re-renderings of `memtest4_eval`**, with only the pre-lift pixels changed:
  - the cups start up but no ball is drawn;
  - the cups are already down from frame 0.
- **The decoder's answer is still a logit of ±25 on every frame**, never between 0.1 and 0.9.
- **It behaves as if the ball started under the middle cup** and follows it through the real swaps. That rule explains 315/315 answers, and the accuracy at the lift is 0.57.
- **The state at the lift sits inside a real "ball in slot k" cluster**, not between clusters.
- Figure: `figures/noreveal_posterior.png`.

**Why that happens:**
- Every training episode showed the start, so the honest answer was always exactly 0 or 1. The decoder's loss (BCE) would reward 1/3 if any training situation called for it; none did.
- The pull-only alignment has no "unknown" class, so it snaps every state into one of three.

## What changes

**Data.** `data/memtest4_all3u.zarr` is built by `scripts/build_uncertain_data.py train`. It holds 768 training episodes and 24 held out:
- the 384 ALL3 training episodes, copied unchanged (start shown);
- **384 closed-start episodes**: the cups stand over the ball from frame 0, so the start slot is never shown. There are 96 per depth (0–3 swaps), with every swap order present.
  - **Half of them have one peek.** After k swaps (k drawn uniformly from 0..n), the cup standing in the lifted slot rises, shows the ball or an empty spot, and comes back down. Then the remaining swaps run.
  - Actions and timings are the same as in a shown episode. A peek adds 10 frames: 6 with the lift action, 4 with the hold action.
- **Held out:** 24 episodes, 8 of each kind.

The generator changes (`closed_start`, `peek_after` in `memtest4.sample_episode`) never draw from the episode's random number generator. So every episode, old or new, still replays from its seed (checked).

**The honest answer is the belief:** the set of slots the ball can be in, given everything seen *so far*.

| what was seen | belief | P(ball under the lifted cup) |
|---|---|---|
| nothing | {0, α, β} | 1/3 |
| peek: ball | {0} | 1 |
| peek: empty cup | {α, β}, "the other two" | 0 |
| ... then a swap of slot 0 with α | {0, β} | 1/2 |

Slots are named 0 (the lifted cup's slot), α and β. Swaps move the set around; a peek collapses it.

**Alignment goes backwards in time; labels must not.** ALL3's backtracking walk labelled earlier frames from later sure moments. That is fine when the start was shown, because the earlier frames really knew. In a closed-start episode it would hand every frame an answer only revealed later, at the peek or at the lift. So the labels are the belief *at that frame*, filtered forward, never smoothed back. They form 7 classes, each pulled together with weight 0.1, pull only:
- `{0} {α} {β} {0,α,β} {α,β} {0,α} {0,β}`.

Labels come from `scripts/belief_labels_phase1.py`:
- **privileged:** true slots, true swaps, true peek outcome.
- **hand-crafted:** no hidden labels; everything comes from the frozen Naive run:
  - **Was the start shown?** Yes if the decoder is sure (p < 0.1 or > 0.9) at the first still frame. A closed start should give about 1/3 there.
  - **Start shown:** the ALL3 backtracking walk from sure "ball here" moments, in both directions.
  - **Start closed:** forward only, starting from {0, α, β}.
    - Swaps are named by the latent-action codes (A/B/C by flip rate on shown episodes, as in ALL3).
    - A peek uses the reward the environment gives at the peek's lift frames (ball revealed or not). That is the model's own training signal.
    - The walk stops at any segment it can't name.

**Runs.** WINDOW60, 20,000 steps, batch 128, as ALL3; only the alignment term differs. Run directories are under `logdir/world_model/smt_phase1_all3u/`.

| run | alignment | config |
|---|---|---|
| Naive | none | `train_all3u_window60_naive_phase1.yaml` |
| Hand-crafted | 7-class belief, hand-crafted labels | `train_all3u_window60_handcrafted_phase1.yaml` |
| Align-privileged | 7-class belief, true labels | `train_all3u_window60_alignpriv_phase1.yaml` |

## Test and pass criterion

**Closed-start test set.** `data/memtest4_evalu.zarr` is built by `scripts/build_uncertain_data.py eval`:
- 0–8 swaps, 45 episodes per depth: 15 without a peek and 30 with one;
- slack 0–20, as in the other test sets.

**Score.** `scripts/belief_eval_phase1.py` gives the frozen reward decoder's p at the final lift onset. This is the posterior chained over every real frame (`posterior_r1`, memory only), compared with the exact Bayes answer p\*.

**Pass:** |p − p\*| ≤ 0.1. It must hold in each category (1/3, 1/2, the other two, known slot), on 0–3 and on 4–8 swaps. The start-shown sets (`memtest4_eval{,5,6,7,8}`) must stay at ALL3's 1.00.

**Before: the ALL3 Hand-crafted-backtracking run on the closed-start test set** (pass rate · mean p; ᵉ = mean |p − p\*|):

| honest answer | swaps | n | ALL3 Hand-crafted-backtracking |
|---|---|---|---|
| 1/3: nothing seen | 0–3 | 60 | 0.00 · 0.25 |
| 1/3: nothing seen | 4–8 | 75 | 0.00 · 0.40 |
| 1/2: one of two, the lifted cup is one | 0–3 | 32 | 0.00 · 0.53 |
| 1/2: one of two, the lifted cup is one | 4–8 | 56 | 0.00 · 0.23 |
| 0: one of the other two (empty peek) | 0–3 | 44 | 0.84 · 0.16 |
| 0: one of the other two (empty peek) | 4–8 | 38 | 0.76 · 0.24 |
| 0/1: known slot | 0–3 | 44 | 0.55 · 0.46ᵉ |
| 0/1: known slot | 4–8 | 56 | 0.88 · 0.11ᵉ |

**Caveat on data.** The training store is twice ALL3's size. The new episodes are a new condition, not more of the same, and all three runs share the store, so the comparison between them is fair. The comparison with ALL3 is not.

## Results

### Closed-start test set (`belief_table.md`)

Pass rate · mean p. ᵉ = mean |p − p\*| for the known-slot rows. Posterior chained to the final lift onset.

| honest answer | swaps | n | Naive | Hand-crafted | Align-privileged |
|---|---|---|---|---|---|
| 1/3: nothing seen | 0–3 | 60 | 0.75 · 0.35 | 0.60 · 0.38 | 0.73 · 0.34 |
| 1/3: nothing seen | 4–8 | 75 | 0.16 · 0.15 | 0.52 · 0.25 | 0.59 · 0.32 |
| 1/2: one of two, the lifted cup is one | 0–3 | 32 | 0.34 · 0.35 | 0.31 · 0.35 | 0.38 · 0.44 |
| 1/2: one of two, the lifted cup is one | 4–8 | 56 | 0.12 · 0.31 | 0.11 · 0.16 | 0.20 · 0.45 |
| 0: one of the other two (empty peek) | 0–3 | 44 | 0.95 · 0.01 | 0.91 · 0.04 | 0.98 · 0.02 |
| 0: one of the other two (empty peek) | 4–8 | 38 | 0.87 · 0.04 | 0.76 · 0.08 | 0.82 · 0.11 |
| 0/1: known slot | 0–3 | 44 | 0.68 · 0.26ᵉ | **1.00** · 0.00ᵉ | **1.00** · 0.00ᵉ |
| 0/1: known slot | 4–8 | 56 | 0.70 · 0.27ᵉ | **1.00** · 0.00ᵉ | **1.00** · 0.00ᵉ |

### Start shown (`memtest4_eval{,5..8}`), posterior F1 at the lift onset (`depth_table_shown.md`)

| swaps | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Naive | 0.80 | 0.94 | 0.85 | 0.62 | 0.49 | 0.27 | 0.09 | 0.14 | 0.10 |
| **Hand-crafted** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Align-privileged | 1.00 | 1.00 | 1.00 | 0.98 | 0.94 | 0.93 | 0.86 | 0.88 | 0.87 |

ALL3 references: Naive 0.95 / 0.98 / 0.87 / 0.85 / 0.49 / 0.38 / 0.47 / 0.46 / 0.39; Hand-crafted-backtracking and Align-privileged 1.00 everywhere.

### What the numbers say

- **Uncertainty now exists.** The ALL3 model scored 0.00 on both uncertain rows and was never between 0.1 and 0.9. Now most "nothing seen" answers land between 0.2 and 0.45.
- **Nothing hidden leaks.** The correlation between p and the true outcome is about 0 in every uncertain row, for every run.
- **Belief alignment keeps the certain cases perfect,** and Hand-crafted keeps ALL3's start-shown result exactly. Align-privileged with 7 classes is slightly worse on shown starts at 6–8 swaps (0.86–0.88).
- **"Nothing seen" decays with depth.** On 4–8 swaps, 21 of 75 Hand-crafted episodes (14 of 75 Align-privileged) snap to p ≈ 0: the "don't know" state drifts into a confident "empty". The teacher, which is not chained, is no better (0.15 pass on 4–8), so this is not drift along the chain. The state itself does not keep "don't know" stable under many swaps.
- **The 1/2 case is barely learned.** Only **40 of 768** training episodes reach the lift in a 1/2 state ({0,α} 17, {0,β} 23). Those states are 3.7% of labelled frames. The decoder has almost no examples from which to learn a 0.5 answer, and its answers are spread from 0 to 1.

### Next

- **Rebalance the peeks, keeping 384 closed-start episodes (not more data).**
  - Place the peek **early**, after 0 or 1 swaps rather than uniformly in 0..n, so that swaps involving slot 0 follow an empty peek more often.
  - Bias the swap orders after an empty peek towards the lifted cup.
  - That turns many {α,β} endings into {0,α} / {0,β}.
- **Make "don't know" closed under swaps.** A swap maps {0,α,β} to itself, but the learned state drifts at 4–8 swaps. That's beyond the trained 0–3, even though the pull already acts at the resting frame between every pair of swaps. Shown starts with the same alignment generalise perfectly to 8, so the difference is how the "don't know" state is anchored. It has no sure moment of its own, unlike "ball here". Worth testing:
  - check whether the drift starts after swaps 1–3 or only past 3, using the per-frame "lift now?" trajectories on the closed-start test set;
  - then decide between more closed-start episodes at 3 swaps (same total) and a stronger pull weight on that class.
- **Re-run Hand-crafted and Align-privileged only.** The labels are already right (100% on labelled frames, closed starts detected 384/384).
