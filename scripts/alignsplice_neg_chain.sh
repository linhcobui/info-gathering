#!/usr/bin/env bash
# ALIGN-SPLICE-NEG: Align-splice (all pairs, twin + start filter, w = 2) + a margin push on NEGATIVE pairs from the
# same splice test (every model and lift timing says p < 0.01 after the splice). Pairs -> smoke test -> train -> evals.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=logdir/world_model/smt_phase1_all3u; NAIVE=$A/naive; R=$A/alignsplice_neg
wait_gpu() {   # sets DEV to a GPU with >= $1 MiB free (shared machine)
  while true; do
    DEV=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
          | awk -F', ' -v need="$1" '$2 - $3 >= need {print "cuda:" $1; exit}')
    [ -n "$DEV" ] && { echo "using $DEV"; return; }
    echo "waiting for a GPU with ${1} MiB free... $(date +%H:%M)"; sleep 120
  done
}
wait_gpu 5000; echo "== 1. pairs + negatives $(date)"
pixi run python scripts/splice_pairs_phase1.py --run $NAIVE/checkpoints/wm_latest.pt --twin $A/naive_twin/checkpoints/wm_latest.pt \
  --sure_start --p_thr 0.99 --neg_thr 0.01 --wait 1 4 --windows 2 --device $DEV --out $NAIVE/splice_pairs_twin_sure_neg
cp $NAIVE/splice_pairs_twin_sure_neg/pairs_w2.npz data/pairs_splice_all3u_neg.npz
wait_gpu 6000; echo "== 2. smoke test $(date)"
S=/tmp/claude-1011/-home-linhco-info-gathering/0407b08f-4f7e-4906-b323-595aa1ab5ba5/scratchpad/smoke_alignsplice_neg; rm -rf $S
pixi run python scripts/train_smt_phase1.py --config-name train_all3u_window60_alignsplice_neg_phase1 device=$DEV logdir=$S \
  trainer.steps=20 trainer.log_every=10 wandb.mode=disabled tensorboard=false 2>&1 | grep -E "pair sampler|^\[20\]"
wait_gpu 6000; echo "== 3. train $(date)"
pixi run python scripts/train_smt_phase1.py --config-name train_all3u_window60_alignsplice_neg_phase1 device=$DEV > $A/alignsplice_neg_train.out 2>&1
grep -oE "^\[20000\] .*" $A/alignsplice_neg_train.out | grep -oE "train/loss/(reward|align|sep)=[0-9.e-]+" | paste -sd' '
wait_gpu 4000; echo "== 4. evals $(date)"
pixi run python scripts/belief_eval_phase1.py --run $R --device $DEV > /dev/null 2>&1
for k in "" 5 6 7 8; do
  pixi run onset-probe-phase1 wm_run=$R outdir=$R/onset_probe_EVAL$k zarr_path=data/memtest4_eval$k.zarr device=$DEV > /dev/null 2>&1
done
L="Naive=$NAIVE Naive_seed1=$A/naive_twin Align-splice=$A/alignsplice Align-splice-neg=$R Align-splice-still=$A/alignsplice_still Hand-crafted=$A/handcrafted Align-privileged=$A/alignpriv"
pixi run python scripts/onset_depth_table.py $L --trained 0-3 --md $A/depth_table_shown_splice_neg.md 2>&1 | grep -A26 "^\*\*F1"
pixi run python scripts/belief_eval_phase1.py --table $L --md $A/belief_table_splice_neg.md 2>&1 | grep "^|"
echo "== done $(date)"
