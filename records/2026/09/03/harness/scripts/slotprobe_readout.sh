#!/usr/bin/env bash
# Readout for chain26.  The split alone is not enough to judge the fix: if the
# LMCache allocator is full, from_gpu never runs, so store() is cheap for a
# reason that has nothing to do with where the drain landed.
D=/home/bo/vast_profiling_problem/results/slotprobe/tp4_slotprobe
clean() { sed 's/\x1b\[[0-9;]*m//g' "$D/server.log"; }

echo "=== SLOTPROBE, last report per worker ==="
clean | grep -o "SLOTPROBE pid=.*" | tail -4

echo
echo "=== STEPPROBE (record 2 ipstoreprobe: loop=143.64 exec=140.47 cpu=140.25) ==="
clean | grep -o "STEPPROBE pid=.*" | tail -4

echo
echo "=== is the store path actually live? ==="
printf "  'Stored ... tokens' lines : %s\n" "$(clean | grep -c 'Stored [0-9]* out of total' || echo 0)"
printf "  allocation failures       : %s\n" "$(clean | grep -ci 'failed to allocate\|allocation failed\|no memory' || echo 0)"
echo "  a sample of the offload timings:"
clean | grep -o "Stored [0-9]* out of total.*" | tail -3
echo "  distinct offload_time values (last 200 stores):"
clean | grep -o "offload_time: [0-9.]*" | tail -200 | awk '{s+=$2; n++} END{if(n)printf "    n=%d mean_offload_ms=%.3f\n", n, s/n; else print "    none"}'

echo
echo "=== hook timers ==="
/home/bo/vast_profiling_problem/.venv/bin/python /home/bo/vast_profiling_problem/scripts/timer_report.py "$D/timers" 2>&1 | head -25

echo
echo "=== end-to-end (record 2 ipstoreprobe: 326.0 s) ==="
/home/bo/vast_profiling_problem/.venv/bin/python - "$D/cold.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(f"  dur={d.get('duration',0):.1f}s  tok/s={d.get('total_token_throughput',0):.0f}")
except Exception as e:
    print("  MISSING:", e)
PY
