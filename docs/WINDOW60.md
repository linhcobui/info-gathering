# WINDOW60: teacher window 60, reward decoder window 60, decoder sees frames

Status: **implemented 2026-10-06; stage 1 running** (`logdir/world_model/smt_phase1_memtest4/window60_phase1`,
chain log `logdir/twin_ft/chain_baseline_window60.log`). Results are written automatically to
[WINDOW60_RESULTS.md](WINDOW60_RESULTS.md) when the evals finish (`scripts/window60_report.sh`).

Smoke tests passed:
- Perturbing frame x_{t+m} first moves r_{t+m+1} (m = 0, 1, 5, 30, 58), and action a_{t+10} first
  moves r_{t+11}.
- P1-250 reproduces exactly under the default layout.
- All `verify-smt-phase1` gates pass, and every anchor now reaches a scored lift frame (1.000, vs
  0.682 at window 20).
- Training runs at about 9 it/s.

Control numbers, P1-250, from the new tools:
- **Pixel F1 score:** unseen 3 = 0.26 and unseen 4 = 0.31, against 0.67 and 0.51 for always saying
  "ball". At 5 swaps: 0.37, against 0.52.
- **Mid-hold probe, teacher slot accuracy by swaps done:** 1.00 / 0.85 / 0.68 / 0.42 / 0.44.
- **History:** start slot 1.00 (the current slot alone predicts 0.47); swaps so far 0.86 (slot
  alone 0.26).

## Idea

Today the reward decoder looks only 20 steps ahead. In a typical 4-swap episode the reward
reaches back to about swap 3, so states earlier in the hold are never asked anything about the
ball. They're shaped only by being one-step reachable, and the cheapest one-step-reachable
state is the history.

WINDOW60 makes every state in the hold answer for the reveal:

- **The window covers the whole hold.** The earliest state that needs the ball's position is
  frame 13: cups just lowered, swap 1 starts at frame 14. From there the decoder must see as far as
  the first scored lift frame (onset + 1), which takes onset + 1 − 13 steps. That's 38 for the
  longest training episode (onset 50) and 46 for the longest test one (onset 58). 60 covers both.
- **The decoder sees the future frames.** The swaps still to come are in its input, so the reveal
  is predictable from any anchor, provided z_t holds where the ball is *now*.
- **The target is unchanged.** The reward is "ball revealed" on lifted frames only (onset +1 to +9),
  masked everywhere before. That's the non-privileged bit.

The decoder then has to compute *reveal = (ball position in z_t), followed by (the swaps it sees)*.
One bit can't do that; the 3-way position can.

## The model at anchor t

```
teacher   z_t = enc( (x_{t-59}, a_{t-60}) … (x_t, a_{t-1}) )      window 60: unchanged

decoder   one token per time step: that step's frame + the action taken at it   (PAIRED layout)

  pos:   0 … 9     10           11                 12                 …   69
       [ z_t ×10 | (x_t, a_t) | (x_{t+1}, a_{t+1}) | (x_{t+2}, a_{t+2}) | … | (x_{t+59}, a_{t+59}) ]   causal
            ↓         ↓              ↓                   ↓                          ↓
           r_t     r_{t+1}        r_{t+2}             r_{t+3}            …       r_{t+60}

          r_t at pos 9 (the last memory token: the first position that has seen all of z_t);
          r_{t+k} at pos 9+k. Read-outs are contiguous; the stack is 10 + 60 = 70 tokens.

loss      BCE(r_{t+k}, ball revealed at frame t+k)  ONLY where frame t+k is lift offset 1-9
          + L_post + L_prior + 1e-3 · L_unif         (unchanged)
```

**What each read-out can see** (● visible; causal mask = everything to its left):

```
                 z_t   (x_t,a_t)  (x_t+1,a_t+1)  (x_t+2,a_t+2)  (x_t+3,a_t+3)
 r_t    (pos 9)   ●
 r_t+1  (pos 10)  ●       ●
 r_t+2  (pos 11)  ●       ●            ●
 r_t+3  (pos 12)  ●       ●            ●              ●
 r_t+4  (pos 13)  ●       ●            ●              ●              ●

 newest frame seen when predicting frame t+k:  x_{t+k-1}     (1-frame lag: predict the next frame's ball)
```

**The pairing must be "frame + the action taken AT it", (x_{t+j}, a_{t+j}).** The teacher pairs each
frame with the action that *produced* it, (x_j, a_{j-1}). Using that here would put the lift action
in the same token as x_{onset+1}, which already shows the ball, so r_{onset+1} could copy it.

**At the lift:** r_{onset+1} reads at the token (x_onset, a_onset = lift). The cup is still down and
the lift has just been triggered, so it must come from memory. From offset 2 on, a read-out sees the
offset-1 frame with the ball visible, which is harmless.

**Building the token.** `Linear([x; a])` is mathematically `W_x·x + W_a·a`, the same as two linears
added (the comment in `updater.py` claiming otherwise is wrong). That's fine, because the
transformer layers mix frame and action afterwards. If the token itself should model frame × action
interactions, use a small MLP: `Linear → SiLU → Linear`.

## Data

`data/memtest4.zarr`: 250 episodes, 0–4 swaps, 225 trained and 25 held out. Same as P1-250, so
P1-250 is the direct control:

| run | teacher window | decoder window | decoder input |
|---|---|---|---|
| P1-250 (done) | 60 | 20 | memory + actions + frames, 2 frames behind |
| **WINDOW60** | 60 | **60** | **memory + one (frame, action) token per step, 1 frame behind** |

No no-obs control. A reward decoder without frames can't predict the reveal from a mid-hold state:
actions carry only the phase, never which cups swap. So its extra targets are noise, and the
comparison says nothing. The frames are the point of the design, not an ingredient to ablate.

## Code changes

1. **`RewardDecoder`: the paired input.** One token per step, `token_j = f([x_{t+j}; a_{t+j}])`
   (a linear or a small MLP), read-outs at `out[:, K-1:]`. Frames `e_seq[:, :-1]` (x_t … x_{t+F-1})
   and actions `a_seq[:, :-1]` (a_t … a_{t+F-1}) are exactly what the dataset already provides.
   Exposed as `arch.decoder_layout: paired` in `model/smt_phase1.yaml`, with the current layout as
   the default so existing checkpoints load and behave unchanged.
2. **`onset_probe_phase1.py` → `read_reward`.** Build its frames the same way.
3. **Config `train_memtest4_window60_phase1.yaml`.** Copied from `train_memtest4_phase1.yaml` with
   `fut_len: 60`, `decoder_layout: paired`, and a fixed logdir.
4. **`scripts/midhold_probe_phase1.py`** (new, diagnostic only). For teacher and posterior states
   at every hold frame of `memtest4_eval`, fit a linear probe for the ball's current slot on half
   the episodes and test on the other half, by **swaps still to come** (0–4). Run it on the same
   states for start slot and swaps so far (history leakage).
5. **`scripts/smt_f1_table.py`.** Split "unseen" into unseen-3 and unseen-4, each with the
   always-"ball" baseline. The pooled number hid that every run is at chance on unseen 4.

The dataset needs nothing: `SmtAnchors` already builds `fut_len + 1` future steps, clamped and
masked at the episode end. Padded frames after the end only feed read-outs that are never scored.

(The `decoder_obs_timing` switch from 2026-10-05 has been removed; `decoder_layout` replaces it.)

## Smoke tests before the real run

1. **Timing:** perturb frame x_{t+m}; the first read-out that moves must be r_{t+m+1}.
2. **No leak at the lift:** the offset-1 read-out sees frames only through the onset. Already
   checked on the data: the onset frame shows 0 ball pixels in every episode of all four stores, and
   offset 1 shows ≥ 15 when the ball is there.
3. **Old checkpoints unchanged:** with the default layout, P1-250's onset probe reproduces exactly.
4. **`verify-smt-phase1` passes** with `decoder_layout=paired`.
5. **Shape and speed:** the stack is 70 tokens. Check 50 steps at batch 128 for memory and it/s on
   the shared GPUs.

## Run and evaluate

```bash
scripts/twin_ft_chain.sh baseline window60 cuda:N    # stage 1 → readout → pixel decoder → evals
pixi run python scripts/midhold_probe_phase1.py      # WINDOW60 vs P1-250
pixi run python scripts/smt_f1_table.py "P1-250=<pd>" "WINDOW60=<pd>"
```

(The chain script needs a 3-line change to take a config name instead of the `s3`/`s3ft` tag.)
Cost: about 1 h stage 1 plus 2 h readout and decoder, per run.

## What counts as a result

| question | measured by | WINDOW60 succeeds if |
|---|---|---|
| does the state carry the position mid-hold? | mid-hold probe, with 1–4 swaps still to come | teacher clearly above P1-250, near 1.0 |
| did history get dropped? | start-slot / swaps-so-far probes among same-slot states | lower than P1-250 (0.9–1.0 today) |
| does it compose? | pixel F1 score, **unseen 4 swaps** and **5 swaps** | clearly above always-"ball" (~0.55) and P1-250 |

**Kill criteria:**
- The probe doesn't move: the decoder found another way, or the frames aren't being used.
- The probe moves but unseen 4 / 5 swaps don't: the position is carried *alongside* the history.
  The posterior still imitates history-specific states, and the next step is to merge states by
  what this decoder needs.

**Caveat:** n is small (unseen 4 ≈ 55 episodes at 250, 5 swaps = 63). Treat differences under
about 0.1 as noise, and use a paired test on the same episodes.

## After stage 1: finetuning

The WINDOW60 stage-1 model is the base for a finetuning stage. The method is **new and still to be
proposed**: not DMT (rolling the updater on its own states and copying the teacher's
history-specific state), and not the twin finetune in TWIN_FINETUNE.md. It will get its own plan
document once the stage-1 results are in.
