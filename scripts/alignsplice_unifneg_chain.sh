#!/usr/bin/env bash
# Negative pairs pushed apart with the UNIFORMITY loss restricted to negative pairs (align_neg: unif; never zero),
# weight 1e-3 as Naive's uniformity. A = all pairs (Align-splice-unifneg), B = still-frame pairs + still negatives
# (Align-splice-still-unifneg). Smoke test -> train A and B -> evals -> tables.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=logdir/world_model/smt_phase1_all3u; CFG=train_all3u_window60_alignsplice_neg_phase1
wait_gpu() {
  while true; do
    DEV=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
          | awk -F', ' -v need="$1" '$2 - $3 >= need {print "cuda:" $1; exit}')
    [ -n "$DEV" ] && { echo "using $DEV"; return; }
    echo "waiting for a GPU with ${1} MiB free... $(date +%H:%M)"; sleep 120
  done
}
OV="+model.loss.align_neg=unif model.loss.coef_sep=1.0e-3"
run() {   # name, pair file
  wait_gpu 6000; echo "== train $1 $(date)"
  pixi run python scripts/train_smt_phase1.py --config-name $CFG device=$DEV logdir=$A/$1 $OV trainer.resume=auto \
    loader.pair_file=$2 model.loss.align_pair_path=$2 "wandb.tags=[phase1,smt,all3u,window60,alignsplice,unifneg]" \
    >> $A/$1_train.out 2>&1
  grep -oE "pair sampler.*" $A/$1_train.out
  grep -oE "train/loss/sep=[-0-9.e]+" $A/$1_train.out | awk -F= '{s+=$2; n++; if ($2==0) z++} END {print "push logged " n " times, zero " z+0 " times, mean " s/n}'
  wait_gpu 4000; echo "== evals $1 $(date)"
  pixi run python scripts/belief_eval_phase1.py --run $A/$1 --device $DEV > /dev/null 2>&1
  for k in "" 5 6 7 8; do
    pixi run onset-probe-phase1 wm_run=$A/$1 outdir=$A/$1/onset_probe_EVAL$k zarr_path=data/memtest4_eval$k.zarr device=$DEV > /dev/null 2>&1
  done
}
run alignsplice_unifneg data/pairs_splice_all3u_neg.npz
run alignsplice_still_unifneg data/pairs_splice_all3u_neg_still.npz
L="Naive=$A/naive Naive_seed1=$A/naive_twin Align-splice=$A/alignsplice Align-splice-unifneg=$A/alignsplice_unifneg Align-splice-still=$A/alignsplice_still Align-splice-still-unifneg=$A/alignsplice_still_unifneg Hand-crafted=$A/handcrafted Align-privileged=$A/alignpriv"
pixi run python scripts/onset_depth_table.py $L --trained 0-3 --md $A/depth_table_shown_splice_unifneg.md 2>&1 | grep "^|\|^\*\*"
pixi run python scripts/belief_eval_phase1.py --table $L --md $A/belief_table_splice_unifneg.md 2>&1 | grep "^|"
echo "== done $(date)"
