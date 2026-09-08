#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The full ladder: every arm at every concurrency, one fresh set of processes
# and one freshly wiped L2 per point.
#
#   ./run_ladder.sh                                  # 3 arms x 6 points
#   ARMS="ip_l2 mp_l2_fs" CONC="100 200" ./run_ladder.sh
#
# ORDER is concurrency-major, arm-minor, so the two arms of each comparison run
# back to back under the same machine conditions.
#
# ONE PROCESS SET PER POINT, because:
#   - each point writes its own corpus (~8.5 GB per request), and they cannot
#     accumulate;
#   - wiping L2 under a running lmcache server would leave its in-memory index
#     pointing at files that no longer exist;
#   - it removes any vLLM prefix-cache or LMCache in-memory carryover between
#     points.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"

OUT="${OUT:-$PWD/out}"
CONC="${CONC:-100 200 300 400 500 600}"
ARMS="${ARMS:-ip_l2 mp_l2_fs mp_ceiling}"
mkdir -p "$OUT"

echo "[ladder] start $(date +%F' '%H:%M:%S)  conc='$CONC'  arms='$ARMS'  out=$OUT"

for c in $CONC; do
  for arm in $ARMS; do
    log="$OUT/ladder_$arm.log"
    need=$(( c * 9 + 300 ))
    echo "[ladder] ===== $arm c=$c  (needs ~${need}GB) $(date +%H:%M:%S) ====="
    # Free the previous point's corpus before this point's disk gate runs.
    rm -rf "${L2_DIR:?}/ip" "${L2_DIR:?}/mp"
    mkdir -p "$L2_DIR/ip" "$L2_DIR/mp"
    free_gb=$(df -BG --output=avail "$L2_DIR" | tail -1 | tr -dc '0-9')
    if (( free_gb < need )); then
      echo "[ladder] SKIP $arm c=$c: ${free_gb}GB free, need ${need}GB" | tee -a "$OUT/SKIPPED"
      continue
    fi
    OUT="$OUT/c$c" ARM="$arm" CONC="$c" "$HERE/run_point.sh" >> "$log" 2>&1
    echo "[ladder] ----- $arm c=$c exit=$? $(date +%H:%M:%S)"
  done
done

echo "[ladder] ALL DONE $(date +%F' '%H:%M:%S)"
echo "[ladder] table it with:  $HERE/collect.py $OUT"
