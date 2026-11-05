#!/usr/bin/env bash
# ALIGN-JSW-PUSH on ALL3-U: same pairs as ALIGN-JSW (data/pairs_jsw_all3u.npz); pull with w = 1 - max JS, push with max JS,
# no blind uniformity -> smoke test -> train -> label-free DMT -> onset probes + belief eval -> tables -> t-SNE.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=logdir/world_model/smt_phase1_all3u; R=$A/alignjsw_push; D=$R/dmt/main; W=$A/alignsplice_unifneg
wait_gpu() {
  while true; do
    DEV=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
          | sort -t, -k3 -n | awk -F', ' -v need="$1" '$2 - $3 >= need {print "cuda:" $1; exit}')
    [ -n "$DEV" ] && { echo "using $DEV"; return; }
    echo "waiting for a GPU with ${1} MiB free... $(date +%H:%M)"; sleep 120
  done
}
wait_gpu 8000; echo "== 1. soft pairs $(date)"
[ -f data/pairs_jsw_all3u.npz ] || pixi run python scripts/soft_pairs_phase1.py --runs $A/naive/checkpoints/wm_latest.pt --rule jsweight --tests lifts --K 64 --max_len 60 --n_pairs 30000 --out data/pairs_jsw_all3u.npz --device $DEV 2>&1 \
  | grep -v -i "warn\|╭\|╰\|│\|·\|Encoder"
wait_gpu 6000; echo "== 2. smoke test $(date)"
S=/tmp/claude-1011/-home-linhco-info-gathering/0407b08f-4f7e-4906-b323-595aa1ab5ba5/scratchpad/smoke_alignjsw_push; rm -rf $S
pixi run python scripts/train_smt_phase1.py --config-name train_all3u_window60_alignjsw_push_phase1 device=$DEV logdir=$S \
  trainer.steps=20 trainer.log_every=10 wandb.mode=disabled tensorboard=false 2>&1 | grep -oE "pair sampler.*|^\[[0-9]+\]|train/loss/(align|sep)=[-0-9.e]+" | paste -sd' '
wait_gpu 6000; echo "== 3. train $(date)"
pixi run python scripts/train_smt_phase1.py --config-name train_all3u_window60_alignjsw_push_phase1 device=$DEV trainer.resume=auto >> $A/alignjsw_push_train.out 2>&1
grep -oE "train/loss/sep=[-0-9.e]+" $A/alignjsw_push_train.out | awk -F= '{s+=$2; n++; if ($2==0) z++} END {print "push logged " n " times, zero " z+0 " times, mean " s/n}'
wait_gpu 8000; echo "== 4. DMT $(date)"
pixi run python scripts/train_dmt_phase1.py --config-name dmt_all3u_phase1 wm_run=$R logdir=$D device=$DEV trainer.resume=auto >> $A/alignjsw_push_dmt_train.out 2>&1
wait_gpu 4000; echo "== 5. evals $(date)"
for run in $R $D; do
  pixi run python scripts/belief_eval_phase1.py --run $run --device $DEV > /dev/null 2>&1
  for k in "" 5 6 7 8; do
    pixi run onset-probe-phase1 wm_run=$run outdir=$run/onset_probe_EVAL$k zarr_path=data/memtest4_eval$k.zarr device=$DEV > /dev/null 2>&1
  done
done
L="Naive=$A/naive Align-Walk=$W Align-Walk+DMT=$W/dmt/main Align-JSW+DMT=$A/alignjsw/dmt/main Align-JSW-push=$R Align-JSW-push+DMT=$D Align-privileged=$A/alignpriv"
pixi run python scripts/onset_depth_table.py $L --trained 0-3 --md $A/depth_table_alignjsw_push.md 2>&1 | grep "^|\|^\*\*"
python3 scripts/belief_mean_p_table.py $L --md $A/belief_mean_p_alignjsw_push.md
echo "== done $(date)"
wait_gpu 4000; echo "== 6. t-SNE $(date)"
O=figures/tsne_alignjsw_push; mkdir -p $O; T="data/memtest4_all3u.zarr:768"
for z in "" 5 6 7 8; do
  pixi run python scripts/rest_states_extract.py --zarr data/memtest4_eval$z.zarr --out $O/states_eval$z.npz --device $DEV \
    "Naive=$A/naive:$T" "Align-Walk SMT+DMT=$W/dmt/main:$T" "Align-JSW-push SMT=$R:$T" "Align-JSW-push SMT+DMT=$D:$T" \
    "Align-privileged=$A/alignpriv:$T" 2>&1 | grep -iE "error|traceback" || true
done
python3 scripts/tsne_ball_slot.py $O/tsne_ball_slot.png "0–4 swaps (memtest4_eval)=$O/states_eval.npz" \
  "5–8 swaps (out of distribution)=$O/states_eval5.npz,$O/states_eval6.npz,$O/states_eval7.npz,$O/states_eval8.npz" 2>&1 | grep -v -i warn
echo "== t-SNE done $(date)"
