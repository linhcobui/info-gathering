#!/usr/bin/env bash
# docs/TWIN_FINETUNE.md, end to end. Two entry points, one per GPU queue:
#
#   scripts/twin_ft_chain.sh baseline <s3|s3ft> <cuda:N>
#   CONFIG=<config> RUN=<run dir> NO_REPORT=1 scripts/twin_ft_chain.sh baseline <tag> <cuda:N>   (any stage-1 config)
#       stage 1 -> readout -> pixel decoder -> pixel eval (0-4, 5) -> onset probes (0-4, 5)
#       s3 = C0 (trained on 0-3 swaps; its readout+decoder are the frozen dictionary)
#       s3ft = C1 (retrained from scratch on S1 + the finetune pool: "is it just more data?")
#
#   scripts/twin_ft_chain.sh finetune <arm> <cuda:N> [hydra overrides...]
#       waits for C0's decoder, finetunes the updaters (scripts/finetune_twin_phase1.py),
#       exports, then the same pixel evals and onset probes on the export
#
# Every stage runs in a FIXED directory with trainer.resume=auto and is retried, so a crash
# (e.g. an OOM from another user's jobs on the shared GPUs) resumes instead of starting over.
# Completion is checked from the checkpoint's step, never from a file merely existing: stage 1
# writes wm_latest.pt at step 0 before it trains anything.
set -u
cd "$(dirname "$0")/.."
ROOT=logdir/world_model/smt_phase1_twinft
LOGS=logdir/twin_ft
mkdir -p "$LOGS"

MODE=$1; TAG=$2; DEV=$3; shift 3
LOG=$LOGS/chain_${MODE}_${TAG}.log
say(){ echo "[$(date '+%m-%d %H:%M:%S') $MODE/$TAG $DEV] $*" | tee -a "$LOG"; }

wait_mem(){   # block until the GPU has $1 MiB free (other users' jobs come and go)
  local idx=${DEV#cuda:} need=$1 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" | tr -d ' ')
    [ "$free" -ge "$need" ] && return 0
    say "waiting for ${need} MiB free on $DEV (have $free)"; sleep 120
  done
}

wait_ram(){   # block until the machine has $1 GB of RAM available (it dies above 125 GB, shared)
  local need=$1 avail
  while true; do
    avail=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
    [ "$avail" -ge "$need" ] && return 0
    say "waiting for ${need} GB of RAM available (have $avail)"; sleep 120
  done
}

retry(){      # retry <name> <cmd...>: up to 4 attempts, 3 min apart
  local name=$1; shift
  for attempt in 1 2 3 4; do
    wait_mem 5000
    wait_ram "${MIN_RAM_GB:-20}"
    say "$name: attempt $attempt"
    if "$@" >>"$LOG" 2>&1; then say "$name: ok"; return 0; fi
    say "$name: FAILED (attempt $attempt), tail:"; tail -5 "$LOG" | sed 's/^/    /' | tee -a "$LOG" >/dev/null
    sleep 180
  done
  say "$name: GAVE UP"; return 1
}

ckpt_step(){  # ckpt_step <path> <key>: the step recorded in a checkpoint
  pixi run python -c "import torch,sys; s=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(int(s[sys.argv[2]]))" "$1" "$2" 2>/dev/null | tail -1
}

evals(){      # evals <decoder_run> <wm_run> : pixel eval + onset probe, on 0-4 and on 5 swaps
  local pd=$1 wm=$2
  retry "pixel eval 0-4" pixi run eval-smt-phase1 decoder_run="$pd" device="$DEV" outdir="$pd/eval_EVAL"
  retry "pixel eval 5" pixi run eval-smt-phase1 decoder_run="$pd" device="$DEV" \
    zarr_path=data/memtest4_eval5.zarr outdir="$pd/eval_EVAL5"
  retry "onset probe 0-4" pixi run onset-probe-phase1 wm_run="$wm" device="$DEV" outdir="$wm/onset_probe_EVAL" \
    zarr_path=data/memtest4_eval.zarr
  retry "onset probe 5" pixi run onset-probe-phase1 wm_run="$wm" device="$DEV" outdir="$wm/onset_probe_EVAL5" \
    zarr_path=data/memtest4_eval5.zarr
}

report(){ pixi run python scripts/twin_ft_report.py >>"$LOG" 2>&1 && say "report updated" || say "report FAILED"; }

if [ "$MODE" = baseline ]; then
  # any stage-1 config can use this chain: CONFIG=<config name> RUN=<fixed run dir> NO_REPORT=1
  CONFIG=${CONFIG:-train_memtest4_${TAG}_noobs_phase1}
  RUN=${RUN:-$ROOT/${TAG}_noobs_phase1}
  RO=$RUN/readout/main
  PD=$RO/pixel_decoder_phase1/main
  say "stage 1 -> $RUN"
  retry "stage 1" pixi run python scripts/train_smt_phase1.py --config-name "$CONFIG" \
    logdir="$RUN" device="$DEV" trainer.resume=auto || exit 1
  [ "$(ckpt_step "$RUN/checkpoints/wm_latest.pt" step)" = 20000 ] || { say "stage 1 did not reach step 20000"; exit 1; }
  retry "readout" pixi run train-readout-phase1 wm_run="$RUN" logdir="$RO" device="$DEV" trainer.resume=auto || exit 1
  [ "$(ckpt_step "$RO/checkpoints/wm_latest.pt" readout_step)" = 8000 ] || { say "readout did not reach 8000"; exit 1; }
  retry "pixel decoder" pixi run train-pixel-decoder-phase1 wm_run="$RO" logdir="$PD" device="$DEV" \
    trainer.resume=auto || exit 1
  [ "$(ckpt_step "$PD/checkpoints/decoder_latest.pt" step)" = 10000 ] || { say "decoder did not reach 10000"; exit 1; }
  touch "$PD/DECODER_DONE"
  say "dictionary ready: $PD"
  evals "$PD" "$RUN"
  touch "$PD/EVALS_DONE"
  [ -n "${NO_REPORT:-}" ] || report
  say "BASELINE $TAG DONE"
elif [ "$MODE" = finetune ]; then
  ARM=$TAG
  PD0=$ROOT/s3_noobs_phase1/readout/main/pixel_decoder_phase1/main
  while [ ! -e "$PD0/DECODER_DONE" ]; do sleep 300; done
  OUT=$ROOT/s3_noobs_phase1/twin_ft/$ARM
  say "finetune $ARM -> $OUT  overrides: $*"
  retry "finetune" pixi run python scripts/finetune_twin_phase1.py decoder_run="$PD0" name="$ARM" \
    logdir="$OUT" device="$DEV" resume=auto "$@" || exit 1
  [ -e "$OUT/export/pixel_decoder/checkpoints/decoder_latest.pt" ] || { say "no export"; exit 1; }
  evals "$OUT/export/pixel_decoder" "$OUT/export"
  if [ -e "$OUT/export_twin/pixel_decoder/checkpoints/decoder_latest.pt" ]; then   # the EMA twin, pixel only
    retry "twin pixel eval 0-4" pixi run eval-smt-phase1 decoder_run="$OUT/export_twin/pixel_decoder" \
      device="$DEV" outdir="$OUT/export_twin/pixel_decoder/eval_EVAL"
    retry "twin pixel eval 5" pixi run eval-smt-phase1 decoder_run="$OUT/export_twin/pixel_decoder" \
      device="$DEV" zarr_path=data/memtest4_eval5.zarr outdir="$OUT/export_twin/pixel_decoder/eval_EVAL5"
  fi
  touch "$OUT/EVALS_DONE"
  report
  say "FINETUNE $ARM DONE"
else
  echo "usage: $0 baseline <s3|s3ft> <cuda:N> | finetune <arm> <cuda:N> [overrides]"; exit 2
fi
