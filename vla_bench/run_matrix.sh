#!/usr/bin/env bash
# Run a list of benchmark jobs strictly one at a time, each inside its own memory-capped
# systemd scope.  Usage: ./run_matrix.sh <phase-file>
# Line format:   <name> | <command> [| mem=16G]
#
# Guards (this 30 GB machine was pushed into OOM/swap-thrash three times before these existed):
#   * MemoryMax per job (default 12G, per-line override), MemorySwapMax=0: a job that outgrows
#     its cap is killed inside its own cgroup and cannot push the desktop into swap.
#   * oom_score_adj=1000 (choom): if the kernel ever needs a victim, it is the job.
#   * start gate: a job only starts once MemAvailable >= cap + 3 GB and the GPU is idle
#     (< GPU_BUSY_MIB used), so it never stacks on top of another session's workload.
#   * watchdog: if system MemAvailable drops below PANIC_MIB while a job runs, the job is killed.
#   * compile jobs (ninja / torch extensions) are forced to one process via shim/ninja.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PHASE="$1"
DEFAULT_MEM="${MEMMAX:-12G}"
PANIC_MIB="${PANIC_MIB:-3000}"
GPU_BUSY_MIB="${GPU_BUSY_MIB:-6000}"
avail_mib() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0; }

trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }

while IFS= read -r line <&3; do
  case "$line" in ''|\#*) continue;; esac
  case "$line" in *'|'*) ;; *) continue;; esac
  name="$(trim "${line%%|*}")"; rest="${line#*|}"
  mem="$DEFAULT_MEM"
  # optional trailing "| mem=16G" (the command itself may contain || and pipes)
  if [[ "$rest" =~ ^(.*)\|[[:space:]]*mem=([0-9]+G)[[:space:]]*$ ]]; then rest="${BASH_REMATCH[1]}"; mem="${BASH_REMATCH[2]}"; fi
  cmd="$(trim "$rest")"
  [ -z "$name" ] && continue
  if [ -f "results/$name.json" ] && [ "${FORCE:-0}" != "1" ]; then echo "[skip] $name (results/$name.json exists)"; continue; fi
  # a job of the form "[ -d X ] || build" whose product already exists is a no-op: do not make it wait at the gate
  if [[ "$cmd" =~ ^\[\ -[df]\ ([^\ ]+)\ \]\ \|\|\  ]] && [ -e "${BASH_REMATCH[1]}" ]; then echo "[skip] $name (${BASH_REMATCH[1]} exists)"; continue; fi
  need=$(( ${mem%G} * 1024 + 3072 ))
  waited=0
  while [ "$(avail_mib)" -lt "$need" ] || [ "$(gpu_used)" -gt "$GPU_BUSY_MIB" ]; do
    [ $((waited % 600)) -eq 0 ] && echo "[wait] $name: need ${need} MiB free RAM (have $(avail_mib)), GPU used $(gpu_used) MiB" | tee -a logs/matrix.log
    sleep 30; waited=$((waited + 30))
  done
  echo "[run ] $name (cap $mem) :: $cmd" | tee -a logs/matrix.log
  unit="vlab-$(echo "$name" | tr -c 'a-zA-Z0-9\n' '-')-$$-$RANDOM"
  t0=$(date +%s)
  ( PATH="$HERE/shim:$PATH" MAX_JOBS=1 choom -n 1000 -- \
      systemd-run --user --scope -q --unit="$unit" -p "MemoryMax=$mem" -p "MemorySwapMax=0" \
      bash -c "$cmd" > "logs/run_$name.log" 2>&1 < /dev/null ) &
  jobpid=$!
  peak=0; note=""
  while kill -0 $jobpid 2>/dev/null; do
    p=$(cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/*/"$unit.scope"/memory.peak 2>/dev/null | head -1)
    [ -n "${p:-}" ] && peak=$(( p / 1048576 ))
    if [ "$(avail_mib)" -lt "$PANIC_MIB" ]; then
      note=" WATCHDOG-KILLED(system MemAvailable<$PANIC_MIB MiB)"
      systemctl --user kill --signal=KILL "$unit.scope" 2>/dev/null
    fi
    sleep 2
  done
  wait $jobpid; rc=$?
  # nothing may outlive its job: reap any process still in the scope (e.g. an engine worker)
  systemctl --user kill --signal=KILL "$unit.scope" 2>/dev/null || true
  echo "[done] $name rc=$rc in $(( $(date +%s) - t0 ))s peak_host_ram=${peak}MiB$note" | tee -a logs/matrix.log
done 3< "$PHASE"
