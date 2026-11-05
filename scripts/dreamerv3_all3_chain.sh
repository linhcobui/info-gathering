#!/usr/bin/env bash
# ALL3 baseline: native DreamerV3 on every 0-3 swap order -> pixel decoder -> imagined-lift eval on 0-8 swaps.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=logdir/world_model/dreamerv3_memtest4_all3/main; DEC=$RUN/decoder/main
wait_ram() { while [ "$(free -g | awk '/Mem:/{print $7}')" -lt "${1:-25}" ]; do echo "waiting for RAM..."; sleep 60; done; }
wait_gpu() {
  while true; do
    DEV=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
          | awk -F', ' -v need="$1" '$2 - $3 >= need {print "cuda:" $1; exit}')
    [ -n "$DEV" ] && { echo "using $DEV"; return; }
    echo "waiting for a GPU with ${1} MiB free... $(date +%H:%M)"; sleep 120
  done
}
mkdir -p $RUN
wait_ram 25; wait_gpu 6000; echo "== world model $(date)"
pixi run python scripts/train_world_model.py --config-name train_memtest4_all3_dreamerv3 logdir=$RUN device=$DEV > $RUN/train.out 2>&1
wait_ram 25; wait_gpu 6000; echo "== pixel decoder $(date)"
pixi run train-decoder wm_run=$RUN wm_checkpoint=$RUN/checkpoints/wm_latest.pt logdir=$DEC device=$DEV > $RUN/decoder.out 2>&1
wait_ram 20; wait_gpu 4000; echo "== imagined-lift eval $(date)"
for k in "" 5 6 7 8; do
  pixi run python scripts/eval_memtest1.py --config-name eval_memtest4 decoder_run=$DEC \
    zarr_path=data/memtest4_eval$k.zarr outdir=$DEC/eval_EVAL$k device=$DEV > $RUN/eval_EVAL$k.out 2>&1
done
pixi run python scripts/lift_eval_depth_table.py $DEC --md $RUN/depth_table.md
echo "== done $(date)"
