#!/usr/bin/env bash
# RAM guard: the machine dies above 125 GB of RAM, and it is shared with other users' jobs.
# Every INTERVAL seconds: log available RAM. If MemAvailable < MIN_AVAIL_GB, kill the largest
# process owned by THIS user (never anyone else's) and log what was killed.
#   nohup setsid scripts/ram_guard.sh > /dev/null 2>&1 &        (log: logdir/ram_guard.log)
set -u
cd "$(dirname "$0")/.."
MIN_AVAIL_GB=${MIN_AVAIL_GB:-8}
INTERVAL=${INTERVAL:-10}
LOG=logdir/ram_guard.log
ME=$(id -un)
mkdir -p logdir
echo "[$(date '+%m-%d %H:%M:%S')] guard start: kill my largest process if MemAvailable < ${MIN_AVAIL_GB} GB" >> "$LOG"
n=0
while true; do
  avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  avail_gb=$((avail_kb / 1048576))
  if [ "$avail_gb" -lt "$MIN_AVAIL_GB" ]; then
    read -r pid rss cmd < <(ps -u "$ME" -o pid=,rss=,cmd= --sort=-rss | grep -v "ram_guard" | head -1)
    if [ -n "${pid:-}" ]; then
      kill "$pid" 2>/dev/null
      echo "[$(date '+%m-%d %H:%M:%S')] LOW RAM: ${avail_gb} GB available -> killed my pid $pid ($((rss / 1048576)) GB): ${cmd:0:120}" >> "$LOG"
    else
      echo "[$(date '+%m-%d %H:%M:%S')] LOW RAM: ${avail_gb} GB available, none of it mine" >> "$LOG"
    fi
  fi
  n=$((n + 1))
  if [ $((n % 30)) -eq 0 ]; then                    # every ~5 min: a heartbeat line
    mine_kb=$(ps -u "$ME" -o rss= | awk '{s+=$1} END{print s+0}')
    echo "[$(date '+%m-%d %H:%M:%S')] available ${avail_gb} GB | mine ~$((mine_kb / 1048576)) GB (RSS, overcounts shared)" >> "$LOG"
  fi
  sleep "$INTERVAL"
done
