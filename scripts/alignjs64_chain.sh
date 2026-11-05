#!/usr/bin/env bash
# ALIGN-JS64 on ALL3-U: pairs judged by K = 64 borrowed futures (Naive-U only, JS rule) -> smoke test -> train ->
# label-free DMT -> onset probes (shown starts 0-8) + closed-start belief eval -> F1 and average-p tables.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=logdir/world_model/smt_phase1_all3u; R=$A/alignjs64; D=$R/dmt/main; W=$A/alignsplice_unifneg
wait_gpu() {
  while true; do
    DEV=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
          | awk -F', ' -v need="$1" '$2 - $3 >= need {print "cuda:" $1; exit}')
    [ -n "$DEV" ] && { echo "using $DEV"; return; }
    echo "waiting for a GPU with ${1} MiB free... $(date +%H:%M)"; sleep 120
  done
}
wait_gpu 8000; echo "== 1. soft pairs $(date)"
[ -f data/pairs_js64_all3u.npz ] || pixi run python scripts/soft_pairs_phase1.py --runs $A/naive/checkpoints/wm_latest.pt --rule js --K 64 --n_pairs 30000 --out data/pairs_js64_all3u.npz --device $DEV 2>&1 \
  | grep -v -i "warn\|╭\|╰\|│\|·\|Encoder"
wait_gpu 6000; echo "== 2. smoke test $(date)"
S=/tmp/claude-1011/-home-linhco-info-gathering/0407b08f-4f7e-4906-b323-595aa1ab5ba5/scratchpad/smoke_alignjs64; rm -rf $S
pixi run python scripts/train_smt_phase1.py --config-name train_all3u_window60_alignjs64_phase1 device=$DEV logdir=$S \
  trainer.steps=20 trainer.log_every=10 wandb.mode=disabled tensorboard=false 2>&1 | grep -oE "pair sampler.*|^\[[0-9]+\]|train/loss/(align|sep)=[-0-9.e]+" | paste -sd' '
wait_gpu 6000; echo "== 3. train $(date)"
pixi run python scripts/train_smt_phase1.py --config-name train_all3u_window60_alignjs64_phase1 device=$DEV trainer.resume=auto >> $A/alignjs64_train.out 2>&1
grep -oE "train/loss/sep=[-0-9.e]+" $A/alignjs64_train.out | awk -F= '{s+=$2; n++; if ($2==0) z++} END {print "push logged " n " times, zero " z+0 " times, mean " s/n}'
wait_gpu 8000; echo "== 4. DMT $(date)"
pixi run python scripts/train_dmt_phase1.py --config-name dmt_all3u_phase1 wm_run=$R logdir=$D device=$DEV trainer.resume=auto >> $A/alignjs64_dmt_train.out 2>&1
wait_gpu 4000; echo "== 5. evals $(date)"
for run in $R $D; do
  pixi run python scripts/belief_eval_phase1.py --run $run --device $DEV > /dev/null 2>&1
  for k in "" 5 6 7 8; do
    pixi run onset-probe-phase1 wm_run=$run outdir=$run/onset_probe_EVAL$k zarr_path=data/memtest4_eval$k.zarr device=$DEV > /dev/null 2>&1
  done
done
L="Naive=$A/naive Align-Walk=$W Align-Walk+DMT=$W/dmt/main Align-Soft+DMT=$A/alignsoft/dmt/main Align-JS64=$R Align-JS64+DMT=$D Align-privileged=$A/alignpriv"
pixi run python scripts/onset_depth_table.py $L --trained 0-3 --md $A/depth_table_alignjs64.md 2>&1 | grep "^|\|^\*\*"
python3 scripts/belief_mean_p_table.py $L --md $A/belief_mean_p_alignjs64.md
echo "== done $(date)"
