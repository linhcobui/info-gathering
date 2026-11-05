#!/usr/bin/env bash
# ALL3 side-by-side: Naive / Align-self-predictions / Align-privileged, all WINDOW60, trained on EVERY swap order
# of 0-3 swaps (data/memtest4_all3.zarr), tested on 0-3 and 4-8 swaps.
#   1. Naive (cuda:0) and Align-privileged (cuda:1) in parallel
#   2. latent actions on the frozen Naive -> its own 3-class labels (label-free choice of the two swap actions)
#   3. Align-self-predictions (cuda:0); onset probes of the first two meanwhile (cuda:1)
#   4. onset probes of Align-self-predictions, then the by-depth table
# Waits for free RAM before each stage (the shared machine dies above 125 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
A=logdir/world_model/smt_phase1_all3
NAIVE=$A/naive; PRIV=$A/alignpriv; ASP=$A/alignselfpred
wait_ram() { while [ "$(free -g | awk '/Mem:/{print $7}')" -lt "${1:-25}" ]; do echo "waiting for RAM..."; sleep 60; done; }
probes() {   # run, device
  for k in "" 5 6 7 8; do
    pixi run onset-probe-phase1 wm_run=$1 outdir=$1/onset_probe_EVAL$k zarr_path=data/memtest4_eval$k.zarr device=$2
  done
}

wait_ram 30; echo "== 1. Naive + Align-privileged $(date)"
pixi run python scripts/train_smt_phase1.py --config-name train_all3_window60_naive_phase1 device=cuda:0 > $A/naive_train.out 2>&1 &
P1=$!
pixi run python scripts/train_smt_phase1.py --config-name train_all3_window60_alignpriv_phase1 device=cuda:1 > $A/alignpriv_train.out 2>&1 &
P2=$!
wait $P1; wait $P2

wait_ram 25; echo "== 2. latent actions + labels on the frozen Naive $(date)"
pixi run python scripts/train_latent_action_prior_phase1.py --run $NAIVE/checkpoints/wm_latest.pt \
  --train_zarr data/memtest4_all3.zarr --n_train 384 --out $NAIVE/latent_action_prior
pixi run python scripts/psr_labels_phase1.py --run $NAIVE/checkpoints/wm_latest.pt \
  --lap $NAIVE/latent_action_prior/latent_action_prior.pt --actions auto --lap_eval $NAIVE/latent_action_prior/eval.json \
  --train_zarr data/memtest4_all3.zarr --n_train 384 --out $NAIVE/latent_action_prior/psr_labels_train.npz
pixi run python scripts/build_psr_align_labels.py --src $NAIVE/latent_action_prior/psr_labels_train.npz \
  --out data/labels_psr_self_all3naive.npz

wait_ram 25; echo "== 3. Align-self-predictions + probes of the first two $(date)"
pixi run python scripts/train_smt_phase1.py --config-name train_all3_window60_alignselfpred_phase1 device=cuda:0 > $A/alignselfpred_train.out 2>&1 &
P3=$!
probes $NAIVE cuda:1
probes $PRIV cuda:1
wait $P3

wait_ram 20; echo "== 4. probes of Align-self-predictions + table $(date)"
probes $ASP cuda:0
pixi run python scripts/onset_depth_table.py "Naive=$NAIVE" "Align-self-predictions=$ASP" "Align-privileged=$PRIV" \
  --trained 0-3 --md $A/depth_table.md
echo "== done $(date)"
