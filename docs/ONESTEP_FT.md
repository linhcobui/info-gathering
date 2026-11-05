# ONESTEP_FT: finetune the posterior one step at a time, against a frozen reward decoder

Status: plan. A draft script exists (`scripts/finetune_onestep_phase1.py`, written 2026-10-06, not
run). It will be brought in line with this plan before anything runs.

## Purpose

An **ablation, not a method**: it trains on test data. The question is whether the posterior
*can* produce states that the frozen WINDOW60 reward decoder reads correctly on 4-swap episodes,
when each update gets credit for one step only, with no backpropagation through the chain. If it
can't fit even the episodes it trains on, the problem is the decoder or the state space, not
credit assignment.

## Model

Start from the WINDOW60 stage-1 checkpoint (`logdir/world_model/smt_phase1_memtest4/window60_phase1`).

| part | role | trained? |
|---|---|---|
| teacher | z_0 from the window ending at frame 0 | frozen |
| **posterior** | z_t = posterior(z_{t-1}, x_t, a_{t-1}) | **trained** |
| reward decoder | reads z_t + the future (frame, action) tokens, paired layout, window 60 | frozen (the judge) |
| prior, readout, pixel decoder | not used | — |

## Data

The test set's 4-swap episodes (`data/memtest4_eval.zarr`, 63 episodes), seen and unseen swap
orders together. Frames come from the cached embeddings, normalised with the model's own
statistics, exactly what `observe()` feeds the posterior.

## One training step

All episodes are rolled out **together in one batch**, in lockstep from t = 0 to t = T. Episodes
have different lengths, so they're padded to the longest. Each episode's own T and onset come from
its own data, and masks switch off what doesn't apply to it.

```
z_0 = teacher(window ending at frame 0)                       all episodes at once, no gradient

for t = 1 .. T:                                                 T = latest lift onset in the batch
    z_t  = posterior( detach(z_{t-1}), x_t, a_{t-1} )          (B, 10, 256): one step, current weights
    r    = decoder( z_t, (x_t,a_t) … (x_{t+59},a_{t+59}) )     (B, 61): r_t .. r_{t+60}
    L_t  = BCE( r, ball revealed ) on the read-outs that are
              • lift offsets 1-9 of that episode,
              • inside that episode (not padding), and
              • from states with t ≤ that episode's onset (after it, the ball is in view)

loss = Σ_t Σ_episodes L_t / (number of scored read-outs)
backward → only the posterior gets gradients;  clip 1.0;  one optimiser step
```

`detach(z_{t-1})` is the point. The loss at t can only change the single posterior step that
produced z_t. It never flows back through z_{t-1} to earlier steps: truncated BPTT of length 1.
The rollout still runs the whole episode with the current weights, so each step's *input* is the
chain's own state, not the teacher's.

- **One optimiser step per rollout, not per t.** Every z_t in a rollout comes from the same weights,
  and the detached inputs make the summed gradient exactly Σ_t (one-step gradient at t). That's
  the one-step objective, with no BPTT. Stepping after every t would change the weights
  mid-episode and make highly correlated, noisy updates, so it's not equivalent. In
  loss-landscape terms, each batch is one descent step on a fixed objective J(θ) = Σ_t L_t, with
  the chain's own states held as inputs. It's a **semi-gradient**, as in TD learning: the states
  depend on θ but are treated as constants, and rolling out again after each update keeps the
  training states on the model's own distribution (like DAgger).
- **Implementation:** call `backward()` on each L_t as it's computed (gradients accumulate exactly
  as if summed, and each step's graph is freed, so memory doesn't grow with T). Then one
  `optimizer.step()` per rollout. The number of scored read-outs is known from the masks before
  the rollout, so each L_t is divided by that total up front.
- **T is the lift onset**, the last state whose read-outs still need memory. States after it see
  the ball in their own frames.
- **Batching:** one batch of all 63 episodes (or minibatches of 16). The time loop runs to the
  batch's latest onset. Padded frames repeat the last real frame and are masked out of the loss.

## Settings

| | value | why |
|---|---|---|
| optimiser | AdamW, lr 1e-5, no weight decay, grad clip 1.0 | one-step gradients don't explode like full BPTT; start low and raise if nothing moves |
| steps | 500 | 63 episodes; evaluate every 100 |
| precision | bf16 autocast, as in stage 1 | |

## Checks before any real run

1. **Reproduces the earlier numbers at step 0.** The script's evaluation must match what we already
   measured: decoding from frame 13 gives 0.38 seen / 0.64 unseen (4 swaps) and 0.60 (5 swaps); at
   the onset, 0.56 on unseen 4 and 0.60 on 5, from the onset probe.
2. **Gradient contract:** after the first backward, only `posterior.*` has gradients.
3. **One-step credit is real:** for one t, the posterior's gradient from L_t alone, computed in the
   full rollout, must equal the gradient from recomputing just that step from the stored z_{t-1}.
   If they match, nothing reached earlier steps.
4. **Batched equals unbatched:** for 2 episodes, the batched rollout's states and losses match
   rolling each episode alone.

## What we measure (before vs after)

Decoder accuracy at the first scored lift frame, read from the posterior **at the onset** (the
eval's question) and **from frame 13** (before any swap):

| set | role |
|---|---|
| 4-swap test, the 63 trained on | can it fit them at all? |
| …split into seen / unseen orders | does fitting differ by order? |
| 5-swap test | never trained on: does anything transfer? |
| 0–3-swap test | did it break what worked? |

Plus the training loss and gradient norm over steps.

## How to read it

- **Can't fit the 63 episodes:** with a frozen judge and one-step credit, the posterior can't reach
  states this decoder reads as right. The decoder (which memorised its training episodes and keys
  on the lift position) is the obstacle. Fix the decoder first (slack augmentation), then rerun.
- **Fits them, but 5 swaps and 0–3 don't improve or get worse:** it learned per-episode states,
  i.e. memorised test. One-step credit isn't enough by itself to make composition.
- **Fits them and 5 swaps improve:** one-step credit against a frozen judge transfers, and is the
  basis for the real finetuning method on non-test data.

**Caveat:** this trains on test episodes. Its 4-swap numbers are in-sample; only the 5-swap and
0–3 numbers say anything about generalisation.
