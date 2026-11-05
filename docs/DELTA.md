# DELTA — a sparse, event-gated updater

**Goal.** One latent that breaks *itself* into objects, with no slots imposed, no labels and no
more data. The mechanism is to constrain *how the belief is allowed to change*: each event may
touch only a few parts of the latent, and always the same parts for the same kind of event.

Status: run 2026-10-02 — all arms in §6 trained and evaluated; kill criteria 2 (tracking collapses where the mask binds) and 3 (history doesn't fall) were met.
Results: [DELTA_RESULTS.md](DELTA_RESULTS.md).

---

## 1. The problem this targets

Three measurements from SMT phase 1 motivate it.

**The teacher makes a new latent for every history, even when the belief is the same.** Among
teacher states that agree on the ball's slot, history is still readable:

| arm | swaps-so-far probe (majority) | start-slot probe (majority) | within-slot variance from depth |
|---|---|---|---|
| BASELINE (unaligned) | 0.906 (0.273) | 1.000 (0.515) | 44.2% |
| ALIGN-LIFT | 0.842 (0.273) | 0.966 (0.515) | 19.3% |
| ALIGN-DTW | 0.929 (0.273) | 1.000 (0.515) | 31.9% |

**Unaligned models fail at composition, not drift.** BASELINE has 0.59 composition error at
4-unseen and 5 swaps. Its belief is already wrong when the swaps finish.

**The updater can't fix this however hard it's trained.** A full history is a perfectly updatable
state: "append the next event" is a valid Markov transition with zero one-step error. That's why
`L_post` sits at ~6e-4 while the teacher memorises. *Updatability and minimality are different
properties*, and the SMT objective asks only for the first.

## 2. The idea

Gestalt *common fate*: what changes together is one thing. Plus *minimal change*: revise as
little as possible. Causal representation learning formalises the pair as **mechanism sparsity**
(Lachapelle et al. 2022). If each kind of event may affect only a few latent dimensions, and the
same few every time, the latent factorises into the world's actual components — identifiably,
without being told what they are.

Applied to this world, where every event is local (a swap moves two of three cups):

| code | what a swap changes | same dims every time? | verdict |
|---|---|---|---|
| history buffer (what the teacher does now) | writes the event into fresh dims | **no** — the location depends on how many events came before | rejected |
| global "ball is in slot k" | rewrites the whole vector | yes | not sparse |
| **per-place blocks, ball stored inside its block** | exchanges blocks *x* and *y*; the third is untouched | **yes** | sparsest stationary code |

So the objective selects a factorised code, kills history encoding (it's non-stationary), and
makes composition free (each event type becomes a fixed operator on fixed blocks).

## 3. The equations

### Posterior (observing)

```
z_hat_{t+1} = normalize_mem( Updater(z_t, e_{t+1}, a_t) )     # current updater = proposal
m_{t+1}     = sigmoid( MaskNet(e_{t+1}, a_t) )                 # in [0,1], one per region
z_{t+1}     = (1 − m_{t+1}) ⊙ z_t  +  m_{t+1} ⊙ z_hat_{t+1}
```

### Prior (imagining, no observation)

```
z_{t+1} = (1 − m_p(a_t)) ⊙ z_t  +  m_p(a_t) ⊙ normalize_mem( PriorUpdater(z_t, a_t) )
```

### Loss

```
L = L_reward + L_post + L_prior + λ_t · ( mean(m) + mean(m_p) )
```

### What each piece is for

- **The convex form `(1−m)⊙z + m⊙ẑ` replaces `z + m⊙Δ`.** It is the same idea, but it keeps
  untouched entries *exactly* unchanged even with SMT's per-token rms-norm. If you add first and
  normalise after, the renorm moves every entry of a token whenever any of them moved.
- **The mask sees only the event — the observation and the action.** It never sees `z`, time or
  history. The same event therefore always opens the same regions, so stationarity is structural
  rather than learned. This is a GRU update gate with two changes: the gate is blind to the
  state, and it is penalised for opening.
- **The action** is an event (an intervention), not history, so it's safe in the mask. In
  memtest4 it carries phase only (`[0,0]` reveal / `[1,0]` hold / `[0,1]` show), and it is
  constant through swaps. Swap identity must come from the observation, which is enough: the R2D
  embedding names the swap pair at **1.000** on moving frames. At the lift onset the cup hasn't
  moved yet, so `m_p([0,1])` is the only signal to open the region being lifted.
- **λ_t** is warmed up from 0, and the mask bias is initialised open (sigmoid ≈ 0.95). Otherwise
  the degenerate solution — a mask that stays shut forever — can win before tracking is learned.

### How it shapes the teacher

`L_post`'s target is live, so its gradient reaches the teacher. A masked updater cannot express
"append to history" or "rewrite everything", so a teacher encoding either can't be matched.
Training then pushes the teacher toward states that change sparsely and in fixed places. This is
"the updater pushes the teacher" with a constraint that removes the drifting and memorising
equilibria, which plain `coef_post` scaling never could (both of those also reach zero
`L_post`).

## 4. Mask granularity

| setting | regions | for | against |
|---|---|---|---|
| `token` (start here) | 10 memory tokens | aligned with per-token rms-norm; a 10-value mask is interpretable at a glance | assumes object boundaries fall on token boundaries |
| `dim` | 2,560 dimensions | the purest "one latent": no structure assumed at all | noisier, harder to read |

The memory tokens are generic and interchangeable, with no competition or assignment imposed. A
token mask only lets the model *choose* to use them as regions.

## 5. Implementation

| file | change |
|---|---|
| `src/world_model/models/smt_phase1/updater.py` | `MaskNet` (MLP on `[e, a]` → 10 or 2560, sigmoid). `PosteriorUpdater.step` / `PriorUpdater.step` return `(z, m)` via a new `step_with_mask`; `step` keeps its signature for existing callers |
| `src/world_model/models/smt_phase1/losses.py` | call `step_with_mask`; add `delta_l1 = mean(m)` and `delta_l1_prior`; log mask mean and fraction open |
| `src/world_model/models/smt_phase1/model.py` | `observe()` / `imagine()` use the gated step, so evaluation runs the same equation as training |
| `src/world_model/models/smt_phase1/dmt.py` | rollouts through the gated step |
| `configs/smt_phase1/model/smt_phase1.yaml` | `arch.delta_mask: off \| token \| dim`, `loss.coef_delta`, `loss.delta_warmup_steps`, `arch.delta_mask_init: 3.0`. Every key present with an inert default — the Hydra struct gotcha |

Smoke tests before any run:
1. `delta_mask: off` reproduces the baseline loss on a fixed batch, bit for bit.
2. A mask of all zeros returns `z_t` exactly; a mask of all ones returns `ẑ`.
3. Gradient contract unchanged: `delta_l1` reaches only the mask network.

## 6. Experiments

All on the existing data, probes and seeds. No alignment unless stated.

| arm | train | mask | λ | question |
|---|---|---|---|---|
| D-base | 0–4 | off | — | control (existing BASELINE-noobs) |
| D-tok-lo | 0–4 | token | 0.01 | does gentle sparsity change anything? |
| D-tok-hi | 0–4 | token | 0.1 | does strong sparsity produce blocks and kill history? |
| D-dim | 0–4 | dim | best of above | does it work with no structure assumed? |
| **D-depth1** | **0–1** | best | best | **the headline: label-free closure from 12 sequences** |

**Why D-depth1 is the decisive arm.** The 0–1-swap data already contains all nine
(slot × swap) transitions the world has. ALIGN-LIFT trained on it was perfect to depth 5;
unaligned, it failed, because after one swap its state was a latent the updater had never seen as
an input. If sparse, stationary updates force *code reuse*, the post-swap state lands back on a
code that has been an input before, and depth 2–5 should come free — with no labels.

## 7. Diagnostics

Training never sees the hidden slot. The diagnostics may, since they only read the trained model.

1. **Mask by event type** — a heatmap of mean `m`, regions × {static, reveal, descent, swap(0,1),
   swap(0,2), swap(1,2), lift}. Expect: static ≈ 0; each swap pair opens a small subset that
   differs by pair.
2. **Blocks** — cluster regions by mask co-activation. Probe the ball's slot from each region
   alone: does the ball live *inside* the place regions rather than in a global code?
3. **History leakage** — re-run the §1 table. The success criterion is that swaps-so-far and
   start-slot probes fall toward their majority baselines.
4. **Composition/drift split** — the error decomposition already built.
5. **Swap-operator transfer R²** — does each swap become one shared map? (ALIGN-LIFT 0.994.)
6. **Onset probe and pixel eval** — posterior F1 by depth, seen/unseen, and five swaps.

## 8. Predictions, written before running

- **Static-frame mask → ~0.** Drift disappears as a side effect.
- **History probes fall substantially** relative to BASELINE at λ = 0.1.
- **Composition error at 4-unseen falls** from BASELINE's 0.59.
- **D-depth1 reaching ≥ 0.9 at five swaps** would be the strongest label-free result of the project.

**Kill criteria.**
- The mask is fully open at every λ where tracking still works. Then the sparsity pressure never
  binds and this is just the baseline.
- The mask shuts and tracking collapses at every λ where it binds.
- History leakage doesn't fall at any working λ. Then minimal change does not imply minimal
  state here.

## 9. Risks

- **Mask collapse to zero.** Mitigated by warm-up and an open initialisation; detected via the
  fraction of the mask that is open.
- **Basis dependence.** The effect lives in the latent's own axes. That's what makes it
  identifiable, but it means rotated probes won't show it, so read diagnostics in the native
  basis.
- **Only three event types.** Identifiability theory wants variety, and three swap pairs is small.
  That's acceptable for a first test and an argument for richer worlds later.
- **Interaction with rms-norm.** A partially masked token is a convex mix of two unit-rms vectors,
  so it isn't exactly unit-rms. The teacher target is unit-rms. Expect a small `L_post` floor;
  renormalise touched tokens only if it matters.
- **The teacher is unconstrained by itself.** All the pressure comes through live `L_post`. If
  that's too weak, the fallback is the split-and-compose consistency check on the teacher
  directly.

## 10. Relation to earlier proposals

DELTA subsumes three of them:
- the **gated updater** (static frames can't move the state),
- the **swap operator `W_s`** (each event type becomes a fixed operator on fixed regions),
- the **minimality bottleneck** (history-appending is ruled out by stationarity rather than by
  capacity).

Unlike ALIGN-* it needs no labels, and unlike object files it imposes no slots.

## 11. References

- Lachapelle et al., *Disentanglement via Mechanism Sparsity Regularization: A New Principle for
  Nonlinear ICA*, CLeaR 2022.
- Moran et al., *Identifiable Deep Generative Models via Sparse Decoding*, TMLR 2022.
- Zheng, Ng & Zhang, *On the Identifiability of Nonlinear ICA: Sparsity and Beyond*, NeurIPS 2022.
- Schölkopf et al., *Toward Causal Representation Learning*, Proc. IEEE 2021.
- Spelke, *Principles of Object Perception*, Cognitive Science 1990 (common fate).
- Alchourrón, Gärdenfors & Makinson, *On the Logic of Theory Change*, J. Symbolic Logic 1985
  (minimal change).
- Cho et al., 2014 (the GRU update gate, which DELTA's gate generalises).
- Liu et al., *Transformers Learn Shortcuts to Automata*, ICLR 2023.
