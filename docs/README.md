# Docs

Design notes, plans and results, newest project last. Each doc gives its own status and date at the top.

## Start here

| Doc | What |
|---|---|
| [NEW_MACHINE.md](NEW_MACHINE.md) | Setup checklist for the new GPU machine (handoff, 2026-11-04) |

## Conventions

| Doc | What |
|---|---|
| [FIGURES.md](FIGURES.md) | Figure style for every graphic: the ten colours, Palatino (P052), captions that define every number |

## Shell game (memtest): SMT memory on the cups-and-ball task

| Doc | What |
|---|---|
| [SMT_PRELIM.md](SMT_PRELIM.md) | smt_prelim: SMT-supervised memory over frozen Dreamer embeddings |
| [SMT_ARCHITECTURE.md](SMT_ARCHITECTURE.md) | smt_prelim architecture |
| [SMT_PHASE1.md](SMT_PHASE1.md) | Phase 1 spec: vector memory, prior/posterior updaters, reward-only decoder |
| [SMT_PHASE1_RESULTS.md](SMT_PHASE1_RESULTS.md) | Phase 1 results |
| [ALIGN.md](ALIGN.md) | The alignment loss |
| [SYMBOLIC_ABLATIONS.md](SYMBOLIC_ABLATIONS.md) | Symbolic-observation ablations |
| [DELTA.md](DELTA.md), [DELTA_RESULTS.md](DELTA_RESULTS.md) | DELTA: a sparse, event-gated updater; results |
| [REVERSE.md](REVERSE.md) | REVERSE: undo the swaps, return to the start (proposal) |
| [TWIN_FINETUNE.md](TWIN_FINETUNE.md), [TWIN_FINETUNE_RESULTS.md](TWIN_FINETUNE_RESULTS.md) | Teaching the updater new situations without labels; results |
| [ONESTEP_FT.md](ONESTEP_FT.md) | Finetuning the posterior one step at a time (plan) |
| [WINDOW60.md](WINDOW60.md), [WINDOW60_RESULTS.md](WINDOW60_RESULTS.md), [WINDOW60_ALIGNPOS.md](WINDOW60_ALIGNPOS.md) | Teacher and decoder window 60; results; with alignment on the lift bit |
| [ALL3_RESULTS.md](ALL3_RESULTS.md) | ALL3: Naive vs Align-self vs Hand-crafted vs Align-privileged on every 0–3-swap order |
| [ALL3U.md](ALL3U.md) | ALL3-U: can SMT say "I don't know" when the start is never shown? |
| [SELF_SIGNATURE.md](SELF_SIGNATURE.md) | Belief alignment from the model's own predictions (paused) |

## OGBar: a bar with a hidden centre of mass

| Doc | What |
|---|---|
| [OGBAR.md](OGBAR.md) | The environment, the play policy, the datasets |
| [OGBAR_SMT.md](OGBAR_SMT.md) | A stochastic SMT world model on frozen DINO tokens |
| [OGBAR_TENET.md](OGBAR_TENET.md) | The next OGBar world model, from "On Latent Beliefs" (plan) |

## Error-spike segmentation around the frozen DINO-WM (`src/seg/`)

| Doc | What |
|---|---|
| [recon.md](recon.md) | Where the DINO-WM, its embeddings and the data live; how to call it |
| [assumptions.md](assumptions.md) | Every assumption and decision in the pipeline |
| [surprise_detectors.md](surprise_detectors.md) | What was tried and found: one-step vs open-loop errors, the detector search, the privileged-data audit |
