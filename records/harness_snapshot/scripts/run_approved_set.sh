#!/usr/bin/env bash
# The set approved on 2026-09-02: 1e (MP at IP's pool, the missing cell for
# "is the penalty IP-specific?"), then the two cheap c=200 baselines that make
# 1d's and 1e's c=200 points interpretable.
#
# Waits for the 1d run to exit first -- one vLLM server at a time on this box.
# Each step is skipped if its warm JSON already exists, so this is re-runnable,
# and a failed arm does not cancel the cheap baselines behind it.
set -uo pipefail
cd /home/bo/vast_profiling_problem || exit 1
R=results/phase1
WAIT_PID="${WAIT_PID:-949970}"

step() {  # step <name> <marker> <cmd...>
  local name="$1" marker="$2"; shift 2
  if [ -s "$marker" ]; then echo "=== $name already done, skipping ==="; return 0; fi
  echo "=== $name start $(date +%H:%M:%S) ==="
  "$@"; local rc=$?
  echo "=== $name end $(date +%H:%M:%S) rc=$rc ==="
  sleep 60
  return 0
}

echo "=== waiting for 1d (pid $WAIT_PID) $(date +%H:%M:%S) ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo "=== 1d finished $(date +%H:%M:%S); 60s for HBM release ==="
sleep 60

step "1e MP@13.7M c=1000,200" "$R/1e_mp_nohybrid/c200_warm.json" \
  env CONC="1000 200" ./scripts/phase1e_mp_nohybrid.sh
step "1a c=200" "$R/1a_rerun/c200_warm.json" env CONC=200 ./scripts/phase1_control_1a.sh
step "1c c=200" "$R/1c_rerun/c200_warm.json" env CONC=200 ./scripts/phase1_control.sh
echo "=== approved set complete $(date +%H:%M:%S) ==="
