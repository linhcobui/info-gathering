#!/usr/bin/env bash
# After the WINDOW60 chain: mid-hold probe + F1 table, WINDOW60 vs P1-250 -> docs/WINDOW60_RESULTS.md
set -u
cd "$(dirname "$0")/.."
W=logdir/world_model/smt_phase1_memtest4/window60_phase1
P=logdir/world_model/smt_phase1_memtest4/2026-09-20_13-58-25_memtest4_phase1
WPD=$W/readout/main/pixel_decoder_phase1/main
PPD=$P/readout/2026-09-20_14-32-18/pixel_decoder_phase1/2026-09-20_15-09-31
DEV=${DEV:-cuda:1}
while [ ! -e "$WPD/EVALS_DONE" ]; do sleep 120; done
OUT=docs/WINDOW60_RESULTS.md
{
  echo "# WINDOW60 — results"
  echo
  echo "Auto-written by \`scripts/window60_report.sh\` on $(date '+%Y-%m-%d %H:%M'). Plan: [WINDOW60.md](WINDOW60.md)."
  echo "WINDOW60 vs its control P1-250: same 250-episode data, model and steps; WINDOW60 has a 60-step"
  echo "reward-decoder window with one (frame, action) token per step, P1-250 a 20-step window."
  echo
  echo "## Imagined-lift F1 score (pixel eval)"
  echo
  pixi run python scripts/smt_f1_table.py "P1-250=$PPD" "WINDOW60=$WPD" 2>/dev/null | grep "^|"
  echo
  echo "## Mid-hold probe (does the state know where the ball is now? does it keep history?)"
  echo
  echo '```'
  pixi run midhold-probe-phase1 "P1-250=$P" "WINDOW60=$W" --device "$DEV" --json logdir/twin_ft/midhold_window60.json 2>/dev/null \
    | grep -v "Pixi task\|^ *$\|dense slot"
  echo '```'
  echo
  echo "## Training (reward accuracy on the first scored lift frame, the one memory-only read-out)"
  echo
  echo '```'
  pixi run python - "$W/metrics.jsonl" <<'PY' 2>/dev/null
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for key in ("holdout/reward_acc_first", "holdout/loss/reward", "holdout/loss/post"):
    vals = [(r["step"], r[key]) for r in rows if key in r]
    if vals:
        print(f"{key:<28}" + "  ".join(f"{s}:{v:.3f}" for s, v in vals[::4] + [vals[-1]]))
PY
  echo '```'
} > "$OUT"
echo "wrote $OUT"
