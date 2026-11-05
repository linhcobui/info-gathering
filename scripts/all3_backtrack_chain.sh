#!/usr/bin/env bash
# ALL3 Hand-crafted-backtracking: train (cuda:0) -> onset probes 0-8 swaps -> by-depth table with the other three runs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # fits next to other users' GPU jobs (~5.2 GB peak)
A=logdir/world_model/smt_phase1_all3; RUN=$A/handcrafted_backtracking
wait_ram() { while [ "$(free -g | awk '/Mem:/{print $7}')" -lt "${1:-25}" ]; do echo "waiting for RAM..."; sleep 60; done; }
# other users' jobs share the GPUs: wait until one has >= $1 MiB free and use it
wait_gpu() {
  while true; do
    DEV=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
          | awk -F', ' -v need="$1" '$2 - $3 >= need {print "cuda:" $1; exit}')
    [ -n "$DEV" ] && { echo "using $DEV"; return; }
    echo "waiting for a GPU with ${1} MiB free... $(date +%H:%M)"; sleep 120
  done
}
wait_ram 25; wait_gpu 5500; echo "== train $(date)"
pixi run python scripts/train_smt_phase1.py --config-name train_all3_window60_handcrafted_backtracking_phase1 device=$DEV > $A/handcrafted_backtracking_train.out 2>&1
wait_ram 20; wait_gpu 4000; echo "== probes $(date)"
for k in "" 5 6 7 8; do
  pixi run onset-probe-phase1 wm_run=$RUN outdir=$RUN/onset_probe_EVAL$k zarr_path=data/memtest4_eval$k.zarr device=$DEV
done
pixi run python scripts/onset_depth_table.py "Naive=$A/naive" "Align-self-predictions=$A/alignselfpred" \
  "Hand-crafted-backtracking=$RUN" "Align-privileged=$A/alignpriv" --trained 0-3 --md $A/depth_table_handcrafted_backtracking.md
echo "== done $(date)"
