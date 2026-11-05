#!/usr/bin/env bash
# ALIGN-SELF-PREDICTIONS: train (20000 steps) -> onset probes (0-4 swaps, 5 swaps) -> F1 table vs Naive /
# Align-self / Align-privileged -> mid-hold probe. Waits for free RAM before each stage (machine dies > 125 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
M=logdir/world_model/smt_phase1_memtest4; B=logdir/world_model/smt_phase1_memtest4_big
RUN=$M/window60_alignselfpred_phase1
wait_ram() { while [ "$(free -g | awk '/Mem:/{print $7}')" -lt "${1:-20}" ]; do echo "waiting for RAM..."; sleep 60; done; }
wait_ram 20; echo "== train $(date)"
pixi run train-smt-phase1-window60-alignselfpred
wait_ram 20; echo "== onset probes $(date)"
pixi run onset-probe-phase1 wm_run=$RUN outdir=$RUN/onset_probe_EVAL zarr_path=data/memtest4_eval.zarr
pixi run onset-probe-phase1 wm_run=$RUN outdir=$RUN/onset_probe_EVAL5 zarr_path=data/memtest4_eval5.zarr
echo "== F1 table $(date)"
pixi run python scripts/onset_f1_compare.py "Naive=$M/window60_phase1:data/memtest4.zarr:225" \
  "Align-self=$M/window60_alignself_phase1:data/memtest4.zarr:225" \
  "Align-self-predictions=$RUN:data/memtest4.zarr:225" \
  "Align-privileged=$B/2026-09-22_21-12-04_memtest4_big_alignlift_phase1:data/memtest4_big.zarr:475" \
  --md $RUN/f1_compare.md
wait_ram 20; echo "== mid-hold probe $(date)"
pixi run midhold-probe-phase1 "Align-self-predictions=$RUN" "Align-self=$M/window60_alignself_phase1" "Naive=$M/window60_phase1" \
  --device cuda:0 --json $RUN/midhold_probe.json
echo "== done $(date)"
