#!/usr/bin/env bash
# ALL3-U (docs/ALL3U.md): Naive / Hand-crafted / Align-privileged on data/memtest4_all3u.zarr (ALL3 + closed-start
# episodes, half with one peek), tested on closed-start episodes 0-8 swaps (belief eval) and on the usual
# start-shown sets 0-8 (onset probes).
#   1. Naive + Align-privileged (true-belief labels, data/labels_belief_all3u_privileged.npz) in parallel
#   2. latent actions on the frozen Naive (start-shown episodes) -> hand-crafted belief labels
#   3. Hand-crafted
#   4. belief eval + onset probes for all three
# Waits for free RAM and a GPU with room before each stage (shared machine).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=logdir/world_model/smt_phase1_all3u
NAIVE=$A/naive; PRIV=$A/alignpriv; HC=$A/handcrafted
mkdir -p $A
wait_ram() { while [ "$(free -g | awk '/Mem:/{print $7}')" -lt "${1:-25}" ]; do echo "waiting for RAM..."; sleep 60; done; }
wait_gpu() {   # sets DEV to a GPU with >= $1 MiB free
  while true; do
    DEV=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
          | awk -F', ' -v need="$1" '$2 - $3 >= need {print "cuda:" $1; exit}')
    [ -n "$DEV" ] && { echo "using $DEV"; return; }
    echo "waiting for a GPU with ${1} MiB free... $(date +%H:%M)"; sleep 120
  done
}
train() {   # config name, log name
  pixi run python scripts/train_smt_phase1.py --config-name $1 device=$DEV > $A/$2_train.out 2>&1
}
evals() {   # run dir
  wait_gpu 4000
  pixi run python scripts/belief_eval_phase1.py --run $1 --device $DEV
  for k in "" 5 6 7 8; do
    pixi run onset-probe-phase1 wm_run=$1 outdir=$1/onset_probe_EVAL$k zarr_path=data/memtest4_eval$k.zarr device=$DEV
  done
}

if [ ! -f $NAIVE/checkpoints/wm_latest.pt ] || [ ! -f $PRIV/checkpoints/wm_latest.pt ]; then
  wait_ram 30; echo "== 1. Naive + Align-privileged $(date)"
  wait_gpu 6000; train train_all3u_window60_naive_phase1 naive & P1=$!
  sleep 240
  wait_gpu 6000; train train_all3u_window60_alignpriv_phase1 alignpriv & P2=$!
  wait $P1; wait $P2
fi

wait_ram 25; wait_gpu 5000; echo "== 2. latent actions + hand-crafted belief labels $(date)"
if [ ! -f $NAIVE/latent_action_prior/latent_action_prior.pt ]; then
  pixi run python scripts/train_latent_action_prior_phase1.py --run $NAIVE/checkpoints/wm_latest.pt \
    --train_zarr data/memtest4_all3u.zarr --n_train 384 --out $NAIVE/latent_action_prior --device $DEV
fi
pixi run python scripts/belief_labels_phase1.py --mode handcrafted --run $NAIVE/checkpoints/wm_latest.pt \
  --lap $NAIVE/latent_action_prior/latent_action_prior.pt --device $DEV | tee $A/handcrafted_labels.log

wait_ram 25; echo "== 3. Hand-crafted + evals of the first two $(date)"
wait_gpu 6000; train train_all3u_window60_handcrafted_phase1 handcrafted & P3=$!
evals $NAIVE
evals $PRIV
wait $P3

wait_ram 20; echo "== 4. evals of Hand-crafted + tables $(date)"
evals $HC
pixi run python scripts/belief_eval_phase1.py --table "Naive=$NAIVE" "Hand-crafted=$HC" "Align-privileged=$PRIV" \
  --md $A/belief_table.md
pixi run python scripts/onset_depth_table.py "Naive=$NAIVE" "Hand-crafted=$HC" "Align-privileged=$PRIV" \
  --trained 0-3 --md $A/depth_table_shown.md
echo "== done $(date)"
