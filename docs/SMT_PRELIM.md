# smt_prelim — SMT-supervised memory over frozen Dreamer embeddings

**Status: preliminary, active.** Everything is new and namespaced `smt_prelim` / `*_prelim`.
No pre-existing file was modified except four added `pixi.toml` tasks and two `pypi-dependencies`.

---

## 1. The question

R2-Dreamer fails on **memtest4** (3 cups, 0–4 swaps, ball hidden under one): imagined-lift
**F1 0.714**, dropping to 0.667 / 0.500 / 0.510 at 2 / 3 / 4 swaps, while `recon` stays
0.897–1.000. The readout is fine; the **prior** is wrong. memtest5 (same episode structure,
2 cups) scores 0.986, so the failure is specifically about tracking *which pair moved*.

Suspected bottleneck: the RSSM's `deter` is trained only by BPTT through a ~60-step unroll
with no per-step target. **SMT** (`submodules/smt`, Kumar & Isola, "Pretraining Recurrent
Networks without Recurrence") replaces recurrent credit assignment with supervised one-step
memory labels `(m_t, x_{t+1}) -> m_{t+1}`, where the labels come from a transformer encoder
trained on a predictive-state objective. This experiment ports that idea onto the Dreamer's
own encoder and cell.

---

## 2. Architecture

The frozen encoder is **the trained ConvEncoder from the memtest4 R2-Dreamer run** — never a
random init:

```
logdir/world_model/r2dreamer_memtest4/2026-09-14_18-57-20_memtest4/checkpoints/wm_latest.pt
  state["model"]["agent"], keys prefixed "encoder."  ->  networks.MultiEncoder, strict=True
```
That run had no holdout split (`eval_every: 1e9`), so `wm_best.pt` was never written and
`wm_latest.pt` (step 15000) is the only checkpoint — and is the one behind the 0.714 baseline,
which keeps the comparison apples-to-apples. Embeddings are E = 1024, one per frame, cached.

Three trained modules:

| module | what it is | in -> out |
|---|---|---|
| **Teacher encoder** | x_transformers `Encoder` (bidirectional, rotary, RMSNorm), depth 8, + K=10 learned registers | `e[t-C+1..t], prev_a[...]` -> `deter (2048)`, `logit (32,16)` |
| **Teacher decoder** | x_transformers `Decoder` (causal), depth 4 | `state + (e[t..t+F-1], a[t..t+F-1])` -> `ê[t+1..t+F]` |
| **One-step predictor** | the native `RSSM` cell (`Deter` block-GRU + `_obs_net`) | `(stoch_t, deter_t, a_t, e_{t+1})` -> `m_{t+1}` |

`K * d_model = 10 * 256 = 2560 = deter 2048 + stoch 32*16`, which is exactly `rssm.feat_size`
and satisfies SMT's own `d_model * mem_tokens == mem_size` constraint.

**The target `m_{t+1}`** is the *same encoder, same weights*, window slid forward by one —
SMT's `x_ctx_next = x[:, 1:Tc+1]`. Two encoder calls per anchor. The causal decoder never
produces the dynamics target; it defines *what `m` contains*.

**Deviations from SMT, and only these:**
1. tokens are continuous embeddings, so token embedding is `Linear`, not `nn.Embedding`;
2. every token carries the action at its step (SMT models unconditioned sequences);
3. the memory is RSSM-shaped, so the dynamics term is MSE on `deter` + KL on the stoch logits
   rather than one MSE. **This deviation is the source of finding 6.4 — read it.**
4. the decoder is shifted one token from SMT's `[K-1:-1]` slice, because predicting `e_{t+1}`
   needs `a_t` and a memory token has nowhere to put it.

`Deter.forward` is a convex mix of `tanh` candidates, so the cell's `deter` lives in (-1,1).
**The teacher's deter head must end in `tanh`** or the MSE target is structurally unreachable
and the only symptom is `deter_r2` stuck near 0.

---

## 3. Files

```
src/world_model/models/smt_prelim/
  bridge.py         sys.path shim -> native rssm/distributions (via the r2dreamer bridge)
  embed_cache.py    build_encoder, cache_embeddings, read_embed_attrs, exact_fp32
  blocks.py         sliding_windows, window_mask, uniformity_loss, normalize_mem,
                    masked_mean, state_feat
  teacher.py        TeacherEncoder, TeacherDecoder (+ decode_step, decode_without_memory)
  predictor.py      OneStepPredictor (RSSM cell; excludes the unused _img_net from parameters())
  losses.py         smt_losses: dec / dyn_deter / kl_dyn / kl_rep / unif + all diagnostics
  pixel_decoder.py  EmbedPixelDecoder: reads ONLY latents["embed"]
  model.py          SmtPrelimWM (the WorldModel ABC)

src/info_gathering/dataset_smt_prelim.py   EmbeddedEpisodes, SmtAnchors, smt_prelim_datasets
scripts/cache_embeddings_prelim.py         stage 1 + gate A checks
scripts/train_smt_prelim.py                stage 1b
scripts/closure_smt_prelim.py              gate D (state-chaining error vs the teacher)
scripts/slack_split_prelim.py              the F1 split that IS the result
configs/smt_prelim/                        cache_embeddings, train_memtest4_prelim,
                                           pixel_decoder_prelim, eval_memtest4_prelim, closure,
                                           model/smt_prelim.yaml, data/{memtest4_windows,
                                           memtest4_anchors}.yaml
```

Note the layering: the caching lives in the **model** package, not `info_gathering`, because
`info_gathering` is the lower layer and never imports `world_model`. `dataset_smt_prelim.py`
is a pure zarr reader.

**Stages 2 and 3 need no new script.** `train_decoder.py` and `eval_memtest1.py` touch the
model only through the ABC and run against this backend via `--config-dir`.

---

## 4. Running it

```bash
pixi run cache-embeddings-prelim                  # writes embed_r2d_memtest4_wm15000 into both zarrs
pixi run train-smt-prelim                         # stage 1b, 20k steps, ~25 min
pixi run train-pixel-decoder-prelim wm_run=<run>  # embed -> pixel head, 10k steps
pixi run eval-smt-prelim decoder_run=<dec run>    # yellow-pixel F1
pixi run closure-smt-prelim wm_run=<run>          # gate D
pixi run python scripts/slack_split_prelim.py <per_episode.csv> data/memtest4_eval.zarr <ctx_len>
```
Useful overrides: `model.ctx_len=60`, `model.loss.reweight=nomem`, `model.observe_mode=window`,
`trainer.steps=500`, `wandb.mode=disabled`.

---

## 5. Results

All numbers are **imagined-lift F1 from the yellow-pixel detector on decoded frames**
(`ball_pixels`, threshold 0.15, >= 3 px). Never report latent-probe numbers. Eval set is
`data/memtest4_eval.zarr`, 315 episodes, positive rate 0.359 (so "always empty" = F1 0,
accuracy 0.667; base-rate guessing ~= F1 0.33).

| run | overall | 0 sw | 1 sw | 2 sw | 3 sw | 4 sw |
|---|---|---|---|---|---|---|
| **R2-Dreamer baseline** | 0.714 | 0.976 | 0.926 | 0.667 | 0.500 | 0.510 |
| smt ctx10, `observe_mode=window` | 0.197 | 0.516 | 0.194 | 0.000 | 0.069 | 0.074 |
| smt ctx10, `observe_mode=recurrent` | 0.366 | 0.400 | 0.333 | 0.207 | 0.392 | 0.444 |
| **smt ctx60, recurrent** | **0.740** | **1.000** | **1.000** | 0.774 | 0.524 | 0.375 |

ctx60 by ball moves: 1.000 / 0.886 / 0.583 / 0.303 / 0.000 for 0/1/2/3/4.
ctx60 by hold: 1.000 (4–11), 0.957 (12–19), 0.842 (20–27), 0.478 (28–35) — memory horizon ~25 steps.

Gates: **A** exact re-encode + provenance + a linear read-out of "ball on screen" at **1.0000**.
**B** `recon` F1 **1.000** — the frozen 1024-d embedding plus an autoencoder head caps nothing.
`gt` is 1.000 in every run, so the eval set and detector are clean.

**In progress:** `ctx60 + reweight=nomem` and `ctx10 + reweight=nomem` (section 7).

---

## 6. Findings

**6.1 TF32 silently corrupts the embedding cache.** The same frame embedded 1.7e-3 apart at
block 512 vs block 64, because TF32 keeps ~10 mantissa bits and cuDNN picks a different kernel
per batch shape. `exact_fp32()` disables it; re-encoding is then bit-identical. `observe()`
uses the same context, or eval inputs drift off the training manifold by that margin.

**6.2 `imagine` was nondeterministic.** `rssm.obs_step` samples the posterior with
`gumbel_softmax` internally. Carrying that *sampled* `stoch` forward makes the rollout
stochastic even with `eval_stoch: mode`. Carry `_stoch(logit)` (the mode) forward instead.

**6.3 `observe` must be recurrent.** SMT's inference (`read_with_teacher_rnn`) encodes a
*prefix* with the teacher and then **unrolls the RNN** over subsequent observations; the MNIST
notebook encodes 256 tokens once and then generates 1568 with `rnn_update` every step, never
re-encoding. Re-encoding a fresh window per step caps memory at `ctx_len` and never exercises
the cell. Fixed; `observe_mode: recurrent` is the default.

**6.4 The RSSM's only observation channel is the categorical `stoch`, and it collapses.**
`deter_{t+1} = f(stoch_t, deter_t, a_t)` — observations never reach `deter` directly.
Measured: two wildly different embeddings into `predictor.step` with identical
`(stoch, deter, action)` change `deter` by **exactly 0.0**; perturbing a frame mid-hold changes
the chained `deter` by **exactly 0.0** at every later step. The categorical is inert —
teacher entropy **88.55 of a maximum 88.72**, max class prob 0.073 vs 0.0625 uniform. Nothing
forces information through the discrete bottleneck because the teacher's `deter` head is free.
SMT's own `rnn_update` is a transformer over `[memory ; observation] -> memory`, a wide path;
the RSSM's is a straw. **This is the main open design issue.**

**6.5 A sliding-window teacher trains the cell to forget.** The target `m_{t+1} = enc(window
ending at t+1)` knows only the last `ctx_len` frames, so the cell is supervised to *discard*
older information. Unrolling a cell whose fixed point is "the last C frames" cannot carry more,
under any inference scheme. This is why ctx10 fails and ctx60 works, and it is structural,
not an optimisation failure.

**6.6 The objective barely ever asks about the ball.** Loss by phase at the one-step position:
reveal 0.0000, **show 0.0001**, hold 0.09–0.20 — the phases where the ball is visible contribute
nothing. Over all 90,740 (anchor, future-position) pairs in training:
- **96.2%** have the same ball-visibility in input and target (answerable by copying)
- **1.0%** (910 pairs, ~3.6 per episode) have the ball *appearing* — the only ones needing memory
- ball-appearance events are **1.00% of total squared error**; a cup-arc frame is worth 2.5x a
  ball-reveal frame in MSE, and there are 40x more of them.

SMT's tasks (pixel streams, text) have *dense* predictive signal. Cups-and-ball is a
**sparse-event** task: 96% of the video is deterministic and the whole question resolves in
~3 frames. A predictive-state objective allocates gradient by prediction difficulty, which here
is almost entirely cup kinematics.

**6.7 61% of frame transitions are bit-identical** (reveal 99.9% static, hold 41%, show 70%),
and the rest are large. The decoder is *deliberately* allowed to copy its input: the memory then
only has to carry what the decoder's inputs do not reveal. At a lift, copying is wrong and only
the state can answer — the task supplies the memory pressure by itself, so no anti-shortcut
machinery is used (no dropout, no masking beyond padding).

---

## 7. The reweighting fix (in progress)

`model.loss.reweight: nomem` weights each prediction by **what the memory buys**, not by how
large the error is:

```
w_j = stopgrad( || decode_without_memory(...)_j - y_j ||^2 )     # same decoder, z zeroed
L   = masked_mean( w_j * ||ŷ_j - y_j||^2 ) ,  w normalised to mean 1
```
Cup arcs are predictable without memory -> low weight. Ball reveals are not -> high weight.
Self-calibrating, uses **no labels** (weighting by the ball detector would hand the model the
eval target). `reweight_warmup: 2000` keeps weights uniform until the no-memory path is worth
listening to. Measured at the switch: **w_moving 1.94 vs w_static 0.17**, a 3–12x ratio.
Watch `train/weight_moving`, `train/weight_static`.

---

## 8. Gotchas

- **`swaps` in `per_episode.csv` is NOT the cup-swap count.** It counts `ball_slot` transitions
  (`eval_memtest1.py:209`), i.e. how many times the *ball* moved. With 3 cups a random pair
  leaves the ball alone ~1/3 of the time. For cup swaps, join on `episode_swapped` in the zarr.
- **Aggregate F1 is uninformative.** Always report the `slack = hold - hold_base(n_swaps)` split
  (`scripts/slack_split_prelim.py`) and the by-swap breakdown. 0.72 could be 0.95/0.45.
- **Eval episodes are padded to 68 > ctx_len 60**, so ctx60 takes the *recurrent* path at eval,
  not the window path. Every lift onset is at `t <= 58`, so the onset state is still a full-prefix
  teacher encoding.
- **Gate B before anything else.** If `recon` F1 drops below ~0.95 the ceiling is broken and every
  imagined number is uninterpretable.
- **Verify gradient paths after touching `losses.py`.** Backward each term alone and check which
  modules get grads: `dec` -> {enc, dec}; `dyn_deter` -> {enc, deter_cell}; `kl_dyn` ->
  {enc, deter_cell, obs_net}; `kl_rep` -> {enc}; nothing may reach `_img_net`.
- `x-transformers==2.18.5` and `einops==0.8.1` (SMT's pins) are now in `pixi.toml`. einops was
  bumped from 0.3.0; nothing else in the repo uses it.
- Training holds out 25 of 250 episodes; the Dreamer baseline used all 250. State this when
  quoting the comparison, or run with `data_anchors.holdout_episodes=0`.

---

## 9. Open questions

1. **6.4 is the blocker.** Three options: (A) memory = categorical only, forcing everything
   through the discrete bottleneck; (B) memory = continuous, replacing the RSSM cell with SMT's
   own transformer `rnn_update` — closest to SMT, but no longer a cell-for-cell comparison with
   the Dreamer; (C) an information/entropy term on the teacher's logits — not in SMT.
2. Does `reweight=nomem` close the ctx60 gap at 3–4 swaps? Does it help ctx10 at all (it should
   not — 6.5 is a different problem)?
3. A sliding-window teacher bounds memory at `ctx_len` (6.5). Beating the baseline at long holds
   with a *short* window requires the one-step target to stop being a windowed encoding.
4. DMT (SMT's second stage, `submodules/smt/src/train_dmt.py`) is not implemented here.
