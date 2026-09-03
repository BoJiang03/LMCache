#!/usr/bin/env bash
# chain26 -- split loss #2's 33.7 ms copy into "drain" and "DMA".
#
#   `slot_mapping.to(self.device)` in IP's wait_for_save profiled at 33.7 ms
#   per call for a 480 KB int64 tensor.  A pageable H2D copy is BOTH
#   host-blocking and stream-ordered, so that number is
#
#       (a) draining the forward-pass kernels already queued on the stream, and
#       (b) the 480 KB DMA, which should be ~50 us.
#
#   The fix differs completely between the two:
#     -> copy dominates : pinned staging + non_blocking, plus a
#                         store_stream.wait_stream barrier (V2's batched_from_gpu
#                         has none, which is why the first attempt crashed).
#     -> drain dominates: the copy is a MIRAGE.  Removing it only relocates the
#                         stall into lmcache_engine.store(), whose
#                         batched_from_gpu already calls store_stream.synchronize()
#                         for every host-resident memory object
#                         (gpu_connectors.py:410).  The target is then the
#                         synchronisation, not the copy.
#
#   The probe synchronises the current stream immediately BEFORE the copy.
#   That changes no semantics -- the pageable copy already waits for exactly
#   that -- so it cannot perturb what it measures.
#
#   Comparability: this is record 2's TP=4 lane, same arm (`ipstoreprobe`),
#   same pinned pool.  It must reproduce loop=143.64 ms/step and
#   wait_for_save=73.9 ms/step.  If it does not, the probe perturbed the run
#   and the split is VOID.
set -uo pipefail
cd /home/bo/vast_profiling_problem
exec 2>&1

OUT=/home/bo/vast_profiling_problem/results/slotprobe
mkdir -p "$OUT" logs

while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 15; done
sleep 10

echo "######## CHAIN26: starting tp4_slotprobe at $(date +%H:%M:%S)"
env GPUS=0,1,2,3 TP=4 NUM_BLOCKS=30000 PROBE=1 \
    LMC_SLOTPROBE=1 LMC_SLOTPROBE_EVERY=200 \
    IP_YAML=lmcache_gpu_only.yaml \
    NPROMPTS=300 CONC=300 LANE_OUT="$OUT" \
    ARM=ipstoreprobe TAG=tp4_slotprobe bash scripts/lane.sh 2>&1 | tee logs/lane_tp4_slotprobe.out
echo "######## CHAIN26: arm exited at $(date +%H:%M:%S)"

clean() { sed 's/\x1b\[[0-9;]*m//g' "$OUT/tp4_slotprobe/server.log"; }

echo "######## CHAIN26: VALIDITY -- SLOTPROBE must be armed on every worker"
n=$(clean | grep -c "SLOTPROBE pid=" || true); n=${n:-0}
echo "  SLOTPROBE report lines: $n"
if [ "$n" -lt 4 ]; then
  echo "  VOID: the probe did not reach the workers; LMC_SLOTPROBE did not propagate."
fi

echo "######## CHAIN26: THE SPLIT -- last report per worker"
clean | grep -o "SLOTPROBE pid=.*" | tail -4

echo "######## CHAIN26: comparability against record 2 (ipstoreprobe)"
echo "  record 2:  loop=143.64  exec=140.47  cpu=140.25   wait_for_save=73.938  sub_ip_store=4.953"
clean | grep -o "STEPPROBE pid=.*" | tail -4

echo "######## CHAIN26: hook timers"
if [ -d "$OUT/tp4_slotprobe/timers" ]; then
  ls "$OUT/tp4_slotprobe/timers" | head
  /home/bo/vast_profiling_problem/.venv/bin/python scripts/timer_report.py "$OUT/tp4_slotprobe/timers" 2>&1 | head -30
fi

echo "######## CHAIN26: end-to-end"
/home/bo/vast_profiling_problem/.venv/bin/python - "$OUT/tp4_slotprobe/cold.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(f"  dur={d.get('duration',0):.1f}s  (record 2 ipstoreprobe: 326.0 s)")
except Exception as e:
    print("  RESULT MISSING:", e)
PY
echo "######## CHAIN26: done $(date +%H:%M:%S)"
