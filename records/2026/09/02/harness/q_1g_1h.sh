#!/usr/bin/env bash
# Approved 2026-09-02 17:07: 1g then 1h, in that order, after 1f finishes.
# One vLLM server at a time on this shared box.
# A failed arm must NOT cancel the one behind it -- each step returns 0.
set -uo pipefail
cd /home/bo/vast_profiling_problem || exit 1
WAIT_PID="${WAIT_PID:-1164107}"

echo "=== waiting for 1f (pid $WAIT_PID) $(date +%H:%M:%S) ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo "=== 1f finished $(date +%H:%M:%S); 60s for HBM release ==="
sleep 60

step() {  # step <name> <marker> <cmd...>
  local name="$1" marker="$2"; shift 2
  if [ -s "$marker" ]; then echo "=== $name already done, skipping ==="; return 0; fi
  echo "=== $name start $(date +%H:%M:%S) ==="
  "$@"; local rc=$?
  echo "=== $name end $(date +%H:%M:%S) rc=$rc ==="
  sleep 60
  return 0
}

step "1g IP backoff=0" results/phase1/1g_ip_nobackoff/c1000_cold.json \
  ./scripts/phase1g_ip_nobackoff.sh
step "1h MP py-spy"    results/phase1/1h_mp_profile/c1000_cold.json \
  ./scripts/phase1h_mp_profile.sh
echo "=== queue complete $(date +%H:%M:%S) ==="
