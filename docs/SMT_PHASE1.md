# smt_phase1 — vector memory, prior/posterior updaters, reward-only decoder

**Status: specification, not yet implemented.** Background for everything referenced here is in
[SMT_PRELIM.md](SMT_PRELIM.md) (the experiment, results, findings) and
[SMT_ARCHITECTURE.md](SMT_ARCHITECTURE.md) (wiring diagrams). Read SMT_PRELIM.md §6 first — the
design below is a direct response to those findings.

---

## 1. Why

Four measurements from `smt_prelim` motivate the redesign:

1. **The RSSM state was dead weight.** In `state_mode: tokens` the `deter`/`logit` heads received no
   gradient at all (6.55M idle params). In `rssm` mode the categorical collapsed to uniform (entropy
   88.55 of a maximum 88.72), so the carried state was provably blind to observations — feeding two
   wildly different embeddings into `predictor.step` changed `deter` by **exactly 0.0**.
2. **The latent-prediction loss spends its gradient on the wrong thing.** Cup kinematics are 96% of
   (anchor, position) pairs and 99% of the squared error, and need no memory. Ball events are ~1% of
   both. Reweighting toward them did not help (F1 0.740 → 0.720).
3. **The state does not hold the answer past two swaps.** Read at the eval anchor, the auxiliary
   reward head gives F1 **1.000 / 1.000 / 0.828 / 0.378 / 0.444** for 0/1/2/3/4 cup swaps, and the
   pixel eval tracks it almost cell-for-cell (0.930 / 0.945 / 0.828 / 0.421 / 0.444). The rollout
   faithfully renders what the state contains; the state is what is missing.
4. **The belief updater was never actually tested.** With `ctx_len=60` the lift onset sits at
   `t ≤ 58` for **100%** of eval episodes, so the rollout always began from a teacher full-prefix
   encoding, and the updater only ever ran on its own predicted embeddings.

Phase 1 keeps what earned its place and fixes (4) directly: memory is a plain vector, the decoder
predicts only the fact the task is about, and the eval is a genuine recurrent rollout that never
re-encodes.

**Reused unchanged:** the frozen R2-Dreamer encoder and its cached embeddings
(`embed_r2d_memtest4_wm15000`), `scripts/cache_embeddings_prelim.py`, and the anchor dataset
`src/info_gathering/dataset_smt_prelim.py` — it already emits one item per timestep, which is what
"sweep across all timesteps" means here.

---

## 2. Architecture

```
memory   m_t = z : (K=10, d=256) plain vectors, rms-normed per token
                   2560 floats. No deter, no logits, no categorical, no sampling.
```

| module | signature | role |
|---|---|---|
| `TeacherEncoder` | `(e[t-59..t], a[t-59..t]) → z_t` | produces one-step targets; `ctx_len = 60` |
| `PosteriorUpdater` | `f_post(z_t, e_{t+1}, a_t) → z_{t+1}` | runs while **observing** |
| `PriorUpdater` | `f_prior(z_t, a_t) → z_{t+1}` | runs while **imagining** |
| `RewardDecoder` | `(z_t, e_{t..t+19}, a_{t..t+19}) → 21 logits` | **training only**; `fut_len = 20` |
| `StateReadout` | `z → ê (1024)` | **stage 2**, frozen states, feeds the pixel decoder |

All attention stacks follow SMT's own MNIST configuration (`submodules/smt/README.md`:
`d_model=256`, `n_heads=8`, `encoder_depth=8`, `rnn_depth=8`, `decoder_depth=4`), with rotary
position embeddings and RMSNorm throughout:

| module | x_transformers class | depth | heads | width | head dim | attention |
|---|---|---|---|---|---|---|
| `TeacherEncoder` | `Encoder` | 8 | 8 | 256 | 32 | bidirectional, 60 frames + 10 registers |
| `PosteriorUpdater` | `Encoder` | 8 | 8 | 256 | 32 | bidirectional, 10 memory + 1 obs token |
| `PriorUpdater` | `Encoder` | 8 | 8 | 256 | 32 | bidirectional, 10 memory + 1 action token |
| `RewardDecoder` | `Decoder` | 4 | 8 | 256 | 32 | **causal**, 10 memory + 40 interleaved tokens |
| `StateReadout` | `Encoder` | 4 | 8 | 256 | 32 | bidirectional, 10 memory + 1 query token |

An 8-layer trunk at this width is ~8.4M params (measured in `smt_prelim`); depth 4 is ~4.2M.

The teacher encoder is `smt_prelim`'s `TeacherEncoder.tokens(...)` path — a bidirectional
x_transformers `Encoder` over `[frame+action tokens ; K learned registers]`, returning the
rms-normed register outputs. Drop its `to_deter` / `to_logit` heads entirely.

### 2.1 The two updaters

Both are **one-step functions**, applied once per anchor during training and never unrolled. They
become recurrent only at eval — that is SMT's whole thesis ("Pretraining Recurrent Networks without
Recurrence").

```python
class PosteriorUpdater(nn.Module):          # observing: the observation is available
    def __init__(self, embed_dim=1024, act_dim=2, d_model=256, depth=8, n_heads=8, n_registers=10):
        super().__init__()
        self.n_registers = n_registers
        self.token_in = nn.Linear(embed_dim + act_dim, d_model)     # pairwise concat, not summed
        self.rnn = Encoder(dim=d_model, depth=depth, heads=n_heads,
                           rotary_pos_emb=True, use_rmsnorm=True)   # bidirectional, K+1 tokens

    def step(self, z, embed, action):                               # z (B,K,d)
        token = self.token_in(torch.cat([embed, action], -1)).unsqueeze(1)   # (B,1,d)
        out = self.rnn(torch.cat([z, token], dim=1))                          # (B,K+1,d)
        return normalize_mem(out[:, :self.n_registers])                       # (B,K,d)


class PriorUpdater(nn.Module):              # imagining: action only, no observation
    def __init__(self, act_dim=2, d_model=256, depth=8, n_heads=8, n_registers=10):
        super().__init__()
        self.n_registers = n_registers
        self.act_in = nn.Linear(act_dim, d_model)
        self.rnn = Encoder(dim=d_model, depth=depth, heads=n_heads,
                           rotary_pos_emb=True, use_rmsnorm=True)

    def step(self, z, action):
        token = self.act_in(action).unsqueeze(1)
        out = self.rnn(torch.cat([z, token], dim=1))
        return normalize_mem(out[:, :self.n_registers])
```

`normalize_mem` is `smt_prelim/blocks.py` (`F.rms_norm` over the last dim) — SMT's `normalize_mem`
with `rms_norm=True`, `fsq` off.

> **Sizing note.** `z_r2` was **0.9999** for the equivalent module in `smt_prelim`, so 8.4M params
> per updater is almost certainly oversized — an MLP on the concatenated `(z, e, a)` would likely
> match it. Kept as a transformer here only so the comparison to `smt_prelim` is clean. If compute
> matters, shrink this first; it is not the bottleneck.

### 2.2 The reward decoder — training only

**SMT's rule, verified from source** (`submodules/smt/src/model.py:148-159`). The sequence is
`[z_0..z_{K-1}, x_fut[0..Tf-1]]`, the slice is `[K-1:-1]`, and since `y_fut[j] == x_fut[j]`
(`get_smt_batch` + `y[t] = x[t+1]`), the alignment is:

```
position:   K-1        K            K+1          ...
holds:      z_{K-1}    x_fut[0]     x_fut[1]     ...
              │          │            │
              v          v            v
predicts:   x_fut[0]   x_fut[1]     x_fut[2]     ...
```

Each token predicts **the next** token; the last memory token predicts the first future one; the
target is never visible to the position predicting it.

Phase 1 applies that rule to an **interleaved action/observation sequence**:

```
[ z_0 … z_9 │ a_t │ e_t │ a_{t+1} │ e_{t+1} │ … │ a_{t+19} │ e_{t+19} ]
     │         │             │
     v         v             v
    r_t     r_{t+1}       r_{t+2}    …     reward read at the memory, then at every ACTION token
```

Exact causal visibility (memory tokens are at the front, so **every** read-out sees all ten):

| read-out | at position | sees in totality |
|---|---|---|
| `r_t` | `z_9` (K−1) | `z_0…z_9` |
| `r_{t+1}` | `a_t` (K) | `z_0…z_9, a_t` |
| `r_{t+2}` | `a_{t+1}` (K+2) | `z_0…z_9, a_t, e_t, a_{t+1}` |
| `r_{t+k}` | `a_{t+k-1}` | `z_0…z_9, a_t…a_{t+k-1}, e_t…e_{t+k-2}` |

Sequence length `K + 2F = 50`; `F + 1 = 21` read-outs (`r_t … r_{t+20}`).

**The property this buys:** `e_{t+j}` first becomes visible when predicting `r_{t+j+2}`, so **the
frame that reveals the ball can never be used to predict its own reward.** No self-leak at any
position. (The earlier summed `(e,a)` layout did leak: the token predicting `r_{t+j+1}` held
`e_{t+j}`, and at the lift that frame already shows the ball.)

```python
class RewardDecoder(nn.Module):
    """Causal decoder over [memory ; a_t, e_t, a_{t+1}, e_{t+1}, ...] -> reward logits."""

    def __init__(self, embed_dim=1024, act_dim=2, d_model=256, depth=4, n_heads=8, n_registers=10):
        super().__init__()
        self.n_registers = n_registers
        self.act_in   = nn.Linear(act_dim, d_model)
        self.embed_in = nn.Linear(embed_dim, d_model)
        self.kind = nn.Embedding(2, d_model)         # 0 = action token, 1 = observation token
        self.decoder = Decoder(dim=d_model, depth=depth, heads=n_heads,
                               rotary_pos_emb=True, use_rmsnorm=True)   # CAUSAL
        self.to_reward = nn.Linear(d_model, 1)       # the ONLY head -- no embedding prediction

    def forward(self, z, e_fut, a_fut):              # (B,K,d), (B,F,E), (B,F,A)
        B, F = a_fut.shape[:2]
        acts = self.act_in(a_fut)   + self.kind.weight[0]               # (B,F,d)
        obs  = self.embed_in(e_fut) + self.kind.weight[1]               # (B,F,d)
        inter = torch.stack([acts, obs], dim=2).reshape(B, 2 * F, -1)   # a_t, e_t, a_{t+1}, ...
        out = self.decoder(torch.cat([z, inter], dim=1))                # (B, K+2F, d)
        heads = torch.cat([out[:, self.n_registers - 1: self.n_registers],   # r_t   at z_{K-1}
                           out[:, self.n_registers::2]], dim=1)              # r_t+1.. at a-tokens
        return self.to_reward(heads).squeeze(-1)                        # (B, F+1)
```

`kind` distinguishes an action token from an observation token: they now share one sequence and
rotary encodes only position, not type.

**Note the deliberate redundancy:** `e_t` appears both inside `z_t` (the encoder window ends at `t`)
and again as a decoder token. This matches SMT, where the memory also encodes `x_t` while `x_fut`
re-supplies tokens. The consequence for interpretation: `r_t` and `r_{t+1}` are the read-outs that
genuinely test the memory, and later ones progressively less so, since they accumulate direct
observation tokens.

### 2.3 Reward target: mask / 0 / 1

Supervised **only at the final lift, once the cup has begun to rise**:

```
              hold (cups down, swaps)                show = THE LIFT
           .............................   |------------------------------------|
offset:                                     0     1     2     3    …    9
cup_z_max: 0.05 .................... 0.05  0.05  0.081 0.139 0.170 …  0.170
target:    mask .................... mask  mask  0/1   0/1   0/1   …  0/1
```

```python
lift  = (phase == SHOW) & (cup_z_max > cup_rest_z)      # show offsets 1..9
target = ball_under_target                              # 0 / 1; masked elsewhere
```

**Offset 0 must be excluded.** The lift action lands one frame before the cup moves, so the ball is
visible in **0%** of episodes there; supervising it would teach "predict no ball" on every episode,
positives included.

All three source arrays are already per-frame in the zarr: `phase`, `cup_z_max`,
`ball_under_target`. 9 supervised positions per episode. With `fut_len = 20`, **68.4%** of anchors
see at least one (measured; up from 44.1% at `fut_len = 10`).

`dataset_smt_prelim.py` already produces `ball_seq` and `ball_lift_seq` with exactly this rule —
reuse it. Only `fut_len` changes, via config. Its `meta_keys` **must** include `cup_z_max`, or
`load_embedded_episodes` raises (it does so deliberately; a silent all-False mask trains happily to
nothing).

---

## 3. Losses

```python
z_t      = encoder(e_ctx,      a_ctx,      ctx_mask)        # window ending at t
z_target = encoder(e_ctx_next, a_ctx_next, ctx_mask_next)   # SAME weights, window slid one step

z_post  = posterior.step(z_t, e_next, a_next)               # e_{t+1}, a_t
z_prior = prior.step(z_t, a_next)                           # a_t only

logits  = reward_decoder(z_t, e_fut, a_fut)                 # (B, F+1) -> r_t .. r_{t+F}

losses = {
  "reward": masked_mean(BCEWithLogits(logits, y_reward), reward_mask),
  "post":   (z_post  - z_target).pow(2).mean(),
  "prior":  (z_prior - z_target.detach()).pow(2).mean(),    # detached -- see below
  "unif":   uniformity_loss(z_t),
}
total = 1.0 * reward + 1.0 * post + 1.0 * prior + 1e-3 * unif
```

`masked_mean` and `uniformity_loss` are `smt_prelim/blocks.py`.

### Why the prior's target is detached and the posterior's is not

`L_post` with a **live** target shapes the encoder to produce states that are one-step reachable —
SMT's `detach_rnn=False`, and wanted.

`L_prior` with a live target would instead push the encoder to make states predictable *without
seeing the observation*, i.e. to discard exactly the observation-specific information we need.
Detaching makes the prior chase the encoder and never the reverse. This is the asymmetry Dreamer
gets from KL balancing (`dyn` 1.0 / `rep` 0.1), expressed on vectors instead of categoricals.

### Why `.pow(2)`, and when to abandon it

MSE is SMT's own choice (`calc_mse_loss`, model.py:31-38), and on this geometry it is equivalent to
cosine: every token is rms-normed, so `‖a−b‖² = 2d − 2⟨a,b⟩`, monotone in cosine similarity.

It is **not neutral**. MSE weights all 2560 dimensions equally, and the ball occupies almost none of
that norm — measured in `smt_prelim`, states with *different* ball answers sit **45.6** apart versus
**44.1** for matching ones, on vectors of norm 50.6. That is how `z_r2` reached 0.9999 while the
answer still failed at 3–4 swaps: the loss was 99.99% satisfied by bulk geometry, with nothing left
for the direction that mattered.

**If phase 1 reproduces that pattern — high `z_r2`, low reward accuracy at 3–4 swaps — change the
dynamics metric first:** whitened/Mahalanobis MSE, or a contrastive term scoring discriminability
rather than L2.

### Gradient propagation

```
                       ┌───────────── reward  (BCE, lift frames only)
                       │                   │
  encoder ─── z_t ─────┼─── reward_decoder ┘
     ▲                 │
     │                 ├─── posterior ──► z_post ──┐
     │                 │                           ├─ L_post = MSE(z_post, z_target)
     │   z_target ─────┼───────────────────────────┘     target LIVE → reaches encoder
     │      ▲          │
     │      │          └─── prior ──────► z_prior ─┐
     │      └── stopgrad ───────────────────────── ┴─ L_prior = MSE(z_prior, sg(z_target))
     │                                                   prior only
     └── unif
```

**Assert this with a per-term gradient attribution test before training:**

| term | must reach | must NOT reach |
|---|---|---|
| `reward` | encoder, reward_decoder | posterior, prior |
| `post` | encoder, posterior | reward_decoder, prior |
| `prior` | prior | encoder, posterior, reward_decoder |
| `unif` | encoder | everything else |

A stray `.detach()` silently removes a path and looks healthy right up to the final eval. This has
already happened twice in this project — once a missing `meta_key` left the auxiliary mask all-False
with zero gradient, once an expectation in the test itself was wrong.

---

## 4. Eval — the point of the redesign

**The decoder is not used at eval, at all.** The updaters do every step.

```python
z = encoder(window at t=0)                       # warm-up: ONE step (config `eval_warmup`)
for t in range(1, onset + 1):                    # OBSERVING: posterior on real embeddings
    z = posterior.step(z, embed[t], prev_action[t])
for t in range(onset + 1, end):                  # IMAGINING: prior, no observations at all
    z = prior.step(z, prev_action[t])
    frames.append(pixel_decoder(readout(z)))     # stage-2 modules, frozen
```

The state is maintained **recurrently across ~40 real steps and then imagined through the lift**,
never re-encoded. This is exactly what `ctx_len=60` made untestable in `smt_prelim`, and it is the
claim phase 1 exists to test.

`eval_warmup = 1` is the strict setting: the teacher encodes only the window at `t = 0` (which, with
left-padding, sees one frame), and the updater carries everything after. Raising it makes the test
easier; `eval_warmup = ctx_len` reduces to the untestable `smt_prelim` configuration.

**`StateReadout`** is a bidirectional transformer over the K memory tokens → a 1024-d embedding,
trained in **stage 2 on frozen stage-1 states** (MSE against the true embedding):

```python
class StateReadout(nn.Module):
    """z (B,K,d) -> reconstructed embedding (B,1024). Stage 2 only; stage-1 weights frozen."""

    def __init__(self, embed_dim=1024, d_model=256, depth=4, n_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)   # learned read-out token
        self.net = Encoder(dim=d_model, depth=depth, heads=n_heads,
                           rotary_pos_emb=True, use_rmsnorm=True)     # bidirectional
        self.to_embed = nn.Linear(d_model, embed_dim)

    def forward(self, z):
        q = self.query.expand(z.shape[0], -1, -1)
        out = self.net(torch.cat([z, q], dim=1))        # (B, K+1, d)
        return self.to_embed(out[:, -1])                # read the query token
```

Depth 4 matches the decoder: it is a read-out, not a representation learner. Its output feeds
the existing `EmbedPixelDecoder` (`smt_prelim/pixel_decoder.py`), so the yellow-pixel detector and
`scripts/eval_memtest1.py` work unchanged and the headline number stays pixel-based and comparable
to earlier runs. `recon` — real embeddings straight through the pixel decoder — remains the ceiling
and measured **F1 1.000**.

Stage 2 never shapes the representation: stage-1 weights are frozen throughout.

---

## 5. Files

**New package `src/world_model/models/smt_phase1/`:**
`updater.py` (both variants), `decoder.py` (`RewardDecoder`), `readout.py` (`StateReadout`),
`losses.py`, `model.py` (`SmtPhase1WM`).

**Reused from `smt_prelim` unchanged** — import, do not copy:
- `bridge.py` — the `sys.path` shim to `submodules/r2dreamer`. At most one such backend per process.
- `blocks.py` — `normalize_mem`, `masked_mean`, `uniformity_loss`, `sliding_windows`, `window_mask`
- `embed_cache.py` — `build_encoder`, `cache_embeddings`, `read_embed_attrs`, `exact_fp32`
- `pixel_decoder.py` — `EmbedPixelDecoder`, reads only `latents["embed"]`
- `teacher.py` — `TeacherEncoder` (use `.tokens(...)`; the `to_deter`/`to_logit` heads are unused)

**Reused as-is:** `src/info_gathering/dataset_smt_prelim.py`. Only `fut_len` changes (10 → 20) via
config, and `meta_keys` must include `cup_z_max`.

**Configs** under `configs/smt_phase1/` mirroring `configs/smt_prelim/`. **Add every knob to the
YAML, not just the Python signature** — a Hydra override on a key absent from the struct fails.
This has bitten twice; both times a launched run died at startup.

**Scripts:** `scripts/train_smt_phase1.py`, `scripts/train_readout_phase1.py`.
`scripts/train_decoder.py` and `scripts/eval_memtest1.py` are reused via `--config-dir`, because
they touch the model only through the `WorldModel` / `Decoder` ABCs in
`src/world_model/models/base.py`.

The stage-1 checkpoint must carry `config.data` pointing at the **standard windowed** dataset
(`info_gathering.dataset_memtest2.memtest2_datasets`), so `load_frozen_models` rebuilds
`EpisodeWindows` with images for stages 2 and 3, while training reads the anchor dataset from a
second config group.

---

## 6. Verification, in order

1. **Dataset** — `fut_len 20` gives exactly 9 supervised positions per episode at show offsets 1–9;
   positive rate ≈ 0.36; no supervised position in the pad; 68.4% anchor coverage.
2. **Gradient attribution** — the four-term table in §3, asserted.
3. **Shapes and causality** — the reward decoder is causal (perturbing future position `j` leaves
   read-outs before `j` bit-identical); the posterior updater responds to its embedding input
   (`obs_sensitivity ≫ 0`); the prior must be **insensitive** to embeddings, since it never sees one.
4. **Closure, before rendering any pixels** — chain the posterior over real observations and compare
   against the teacher's encoding at each step: `‖z_chain_h − z_enc(t+h)‖` and R² for `h = 1..40`.
   **If the chain diverges over ~40 steps the eval cannot work**, and nothing downstream is
   interpretable. Do this first.
5. **Stage-2 ceiling** — `recon` F1 must stay ≈ 1.000, and `readout(z_teacher) → pixels` should also
   be high. That is the ceiling the imagined rollout is measured against.
6. **Eval** — yellow-pixel F1 by cup-swap count, reported alongside **balanced accuracy and MCC**.
   F1 alone is unsafe at this 0.359 positive rate: a near-constant predictor once scored F1 0.544
   while its accuracy fell to 0.410, below every baseline.

**Reference numbers on `data/memtest4_eval.zarr`** (315 episodes, positive rate 0.359; "always
empty" = F1 0.000 at accuracy 0.641; "always ball" = F1 0.528 at accuracy 0.359):

| run | F1 | acc | MCC | 0 sw | 1 sw | 2 sw | 3 sw | 4 sw |
|---|---|---|---|---|---|---|---|---|
| R2-Dreamer baseline | 0.714 | 0.794 | 0.552 | 0.976 | 0.926 | 0.667 | 0.500 | 0.510 |
| smt_prelim ctx60, SMT cell | **0.757** | 0.835 | 0.635 | 1.000 | 1.000 | 0.774 | 0.634 | 0.318 |
| smt_prelim ctx60 + ball head | 0.724 | 0.816 | 0.591 | 0.930 | 0.945 | 0.828 | 0.421 | 0.444 |

---

## 7. What this can and cannot show

It **can** show whether a recurrently maintained vector belief carries the ball through ~40
observed steps and a 10-step imagination — the thing no previous run tested, because `ctx_len=60`
meant the teacher re-encoded the whole prefix at every step.

It **cannot** be compared like-for-like to the R2-Dreamer 0.714: the reward head trains on the
quantity the eval measures. Report it as its own aux-supervised condition.

**Falsifiable expectation.** `smt_prelim` showed the *encoder* resolves 0–2 swaps and fails at 3–4
even under direct supervision. Phase 1 changes the state and the objective but **not the encoder**.
So if 3–4 swaps stay near chance, the constraint is the encoder's representation of permutation
structure, and the next move is the encoder — not another objective and not another cell.

A useful secondary reading: compare `r_t` / `r_{t+1}` accuracy (memory-only read-outs) against the
later ones (which accumulate direct observation tokens). A large gap means the decoder is leaning on
observations rather than memory.
