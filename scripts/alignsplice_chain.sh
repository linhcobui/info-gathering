#!/usr/bin/env bash
# ALIGN-SPLICE on ALL3-U: positive pairs with NO privileged information, then an aligned run and its evals.
#   1. pairs (scripts/splice_pairs_phase1.py) on the frozen Naive-U + its seed-1 twin, p > 0.99, both lift
#      timings (1 and 4 frames), windows w = 0 and 2, three filter settings reported side by side:
#        twin            both models must agree
#        sure            decoder-only "was the start shown?" filter (both models)
#        twin + sure     both
#   2. the pair set used for training is fixed IN ADVANCE (not picked by its scored precision): the strictest one,
#      twin + sure, w = 2 -> data/pairs_splice_all3u.npz
#   3. train Align-splice (train_all3u_window60_alignsplice_phase1.yaml)
#   4. belief eval (closed starts) + onset probes (shown starts 0-8), tables next to Naive / Hand-crafted / Align-privileged
# Waits for free RAM and a GPU with room before each stage (shared machine).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=logdir/world_model/smt_phase1_all3u
NAIVE=$A/naive; TWIN=$A/naive_twin; PRIV=$A/alignpriv; HC=$A/handcrafted; SPL=$A/alignsplice
wait_ram() { while [ "$(free -g | awk '/Mem:/{print $7}')" -lt "${1:-25}" ]; do echo "waiting for RAM..."; sleep 60; done; }
wait_gpu() {   # sets DEV to a GPU with >= $1 MiB free
  while true; do
    DEV=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
          | awk -F', ' -v need="$1" '$2 - $3 >= need {print "cuda:" $1; exit}')
    [ -n "$DEV" ] && { echo "using $DEV"; return; }
    echo "waiting for a GPU with ${1} MiB free... $(date +%H:%M)"; sleep 120
  done
}
pairs() {   # out name, extra args
  local out=$1; shift
  pixi run python scripts/splice_pairs_phase1.py --run $NAIVE/checkpoints/wm_latest.pt --p_thr 0.99 --wait 1 4 \
    --windows 0 2 --device $DEV --out $NAIVE/$out "$@"
}

wait_ram 20; wait_gpu 6000; echo "== 1. pairs $(date)"
pairs splice_pairs_twin_sure --twin $TWIN/checkpoints/wm_latest.pt --sure_start
pairs splice_pairs_twin      --twin $TWIN/checkpoints/wm_latest.pt
[ -f $NAIVE/splice_pairs_w14_sure/pairs_w2.npz ] || pairs splice_pairs_w14_sure --sure_start

echo "== 2. training pair set: twin + sure, w = 2 (fixed in advance)"
cp $NAIVE/splice_pairs_twin_sure/pairs_w2.npz data/pairs_splice_all3u.npz

wait_ram 25; wait_gpu 6000; echo "== 3. train Align-splice $(date)"
if [ ! -f $SPL/checkpoints/wm_000020000.pt ] && ! grep -q "^\[20000\]" $A/alignsplice_train.out 2>/dev/null; then
  pixi run python scripts/train_smt_phase1.py --config-name train_all3u_window60_alignsplice_phase1 device=$DEV \
    > $A/alignsplice_train.out 2>&1
fi

wait_ram 20; wait_gpu 4000; echo "== 4. evals + tables $(date)"
pixi run python scripts/belief_eval_phase1.py --run $SPL --device $DEV
for k in "" 5 6 7 8; do
  pixi run onset-probe-phase1 wm_run=$SPL outdir=$SPL/onset_probe_EVAL$k zarr_path=data/memtest4_eval$k.zarr device=$DEV
done
pixi run python scripts/belief_eval_phase1.py --table "Naive=$NAIVE" "Hand-crafted=$HC" "Align-privileged=$PRIV" \
  "Align-splice=$SPL" --md $A/belief_table_splice.md
pixi run python scripts/onset_depth_table.py "Naive=$NAIVE" "Hand-crafted=$HC" "Align-privileged=$PRIV" \
  "Align-splice=$SPL" --trained 0-3 --md $A/depth_table_shown_splice.md
echo "== done $(date)"
