#!/usr/bin/env bash
# Align-Walk (smt_phase1_all3u/alignsplice_unifneg) + DMT with only the label-free terms (dmt_all3u_phase1.yaml,
# coef_slot 0), then onset probes on the shown-start sets 0-8 and an F1 table: SMT vs SMT+DMT.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=logdir/world_model/smt_phase1_all3u; W=$A/alignsplice_unifneg; D=$W/dmt/main
wait_gpu() {
  while true; do
    DEV=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
          | awk -F', ' -v need="$1" '$2 - $3 >= need {print "cuda:" $1; exit}')
    [ -n "$DEV" ] && { echo "using $DEV"; return; }
    echo "waiting for a GPU with ${1} MiB free... $(date +%H:%M)"; sleep 120
  done
}
wait_gpu 8000; echo "== DMT $(date)"
pixi run python scripts/train_dmt_phase1.py --config-name dmt_all3u_phase1 wm_run=$W logdir=$D device=$DEV trainer.resume=auto \
  >> $A/alignwalk_dmt_train.out 2>&1
grep -oE "^\[6000\] .*" $A/alignwalk_dmt_train.out | grep -oE "train/(loss/[a-z]+|dmt/r2_end)=[-0-9.e]+" | paste -sd' '
wait_gpu 4000; echo "== onset probes $(date)"
for k in "" 5 6 7 8; do
  pixi run onset-probe-phase1 wm_run=$D outdir=$D/onset_probe_EVAL$k zarr_path=data/memtest4_eval$k.zarr device=$DEV > /dev/null 2>&1
done
pixi run python scripts/onset_depth_table.py "Naive=$A/naive" "Align-Walk SMT=$W" "Align-Walk SMT+DMT=$D" \
  "Align-privileged=$A/alignpriv" --trained 0-3 --md $A/depth_table_alignwalk_dmt.md 2>&1 | grep "^|\|^\*\*"
echo "== done $(date)"
