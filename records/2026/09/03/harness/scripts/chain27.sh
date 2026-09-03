#!/usr/bin/env bash
# chain27 -- does layerwise remove IP's forward-pass drain?
#
#   chain26 settled the mechanism of loss #2: of wait_for_save's 76.3 ms/call,
#   64.5 is draining the forward pass and 0.089 is the 480 KB slot_mapping DMA.
#   The pageable copy is a mirage; the block is what costs.  It lands at the
#   exit of vLLM's _model_forward context manager, BEFORE compute_logits and
#   sampling -- exactly where vLLM wants its CPU run-ahead.  End to end that
#   costs +27.5 s / +12.5 ms/step over `none`.
#
#   Layerwise is LMCache's own answer: save_kv_layer runs after each layer's
#   attention, so the storer is created at layer 0 and its .to() drains only
#   layer 0's kernels, and the D2H of layer i overlaps layer i+1's compute.
#
#   Pre-registered reading, against tp4_slotprobe (dur=324.4 s, sync=64.5,
#   copy=0.089, store=4.96, exec=140.4) and `none` (296.9 s, exec=81.82):
#     -> dur ~300 s, sync collapses      : layerwise IS the answer; config-only,
#                                          VAST can take it today.
#     -> dur ~324 s, sync still ~60      : the drain follows the store wherever
#                                          it goes; only the event-based async
#                                          store (removing store_stream.synchronize)
#                                          can win, and that is a transfer-path change.
#     -> dur >> 324 s                    : layerwise costs more than it saves here;
#                                          record it and move to the real fix.
#     no "SLOTPROBE-LAYERWISE" in the log -> VOID, use_layerwise did not take.
#
#   Same lane as chain26: TP=4, GPUs 0-3, 300 prompts, c=300, pool pinned at
#   30,000 blocks.  The lane's POOL_REF asserts comparability for us.
set -uo pipefail
cd /home/bo/vast_profiling_problem
exec 2>&1

OUT=/home/bo/vast_profiling_problem/results/slotprobe

# Wait for any lane still running -- but NOT for ourselves.  Record 2 already
# logged this defect once ("pgrep patterns must not match their own caller")
# and it recurred here in a new shape: the script was written with a heredoc
# and launched in the SAME tool call, so the launching shell's own command line
# contained the script text, which contains the pattern.  pgrep matched that
# ancestor and the arm waited on itself forever.  Two defences: launch by path
# (short command line), and skip every ancestor of this process.
ancestors() { local p=$$; while [ "$p" -gt 1 ]; do echo "$p"; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [ -z "$p" ] && break; done; }
lane_running() {
  local skip; skip=" $(ancestors | tr '\n' ' ')"
  for p in $(pgrep -u bo -f "bash scripts/lane[.]sh" 2>/dev/null); do
    case "$skip" in *" $p "*) continue ;; esac
    return 0
  done
  return 1
}
while lane_running; do sleep 15; done
sleep 5

echo "######## CHAIN27: starting tp4_layerwise at $(date +%H:%M:%S)"
env GPUS=0,1,2,3 TP=4 NUM_BLOCKS=30000 PROBE=1 \
    LMC_SLOTPROBE=1 LMC_SLOTPROBE_EVERY=200 \
    IP_YAML=ip_layerwise.yaml \
    NPROMPTS=300 CONC=300 LANE_OUT="$OUT" \
    ARM=ipstoreprobe TAG=tp4_layerwise bash scripts/lane.sh 2>&1 | tee logs/lane_tp4_layerwise.out
echo "######## CHAIN27: arm exited at $(date +%H:%M:%S)"

clean() { sed 's/\x1b\[[0-9;]*m//g' "$OUT/tp4_layerwise/server.log"; }

echo "######## CHAIN27: VALIDITY -- use_layerwise must have taken"
n=$(clean | grep -c "SLOTPROBE-LAYERWISE" || true); n=${n:-0}
echo "  SLOTPROBE-LAYERWISE lines: $n"
[ "$n" -lt 1 ] && echo "  VOID: the non-layerwise copy site ran, so use_layerwise never reached the adapter."
clean | grep -o "use_layerwise[^,)]*" | head -2

echo "######## CHAIN27: THE SPLIT -- last report per worker"
clean | grep -o "SLOTPROBE[A-Z-]* pid=.*" | tail -4
echo "  chain26 (non-layerwise): sync=64.47  copy=0.089  store=4.96  n_store=1930"

echo "######## CHAIN27: step probe"
clean | grep -o "STEPPROBE pid=.*" | tail -4
echo "  chain26: exec_wall=140.42 exec_cpu=140.19    none: exec=81.82"

echo "######## CHAIN27: is the store path live?"
printf "  'Stored ...' lines : %s\n" "$(clean | grep -c 'Stored [0-9]* out of total' || echo 0)"
clean | grep -o "offload_time: [0-9.]*" | tail -200 | awk '{s+=$2;n++} END{if(n)printf "  last-200 mean offload_ms=%.3f\n",s/n; else print "  no offload lines"}'

echo "######## CHAIN27: hook timers (worker 0)"
f=$(ls "$OUT/tp4_layerwise/timers"/timer_WORKER_*.json 2>/dev/null | head -1)
[ -n "$f" ] && /home/bo/vast_profiling_problem/.venv/bin/python - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
steps = d.get("steps", 0) or 1
rows = sorted(d["hooks"].items(), key=lambda kv: -kv[1]["seconds"])
print(f"  steps={steps}")
for k, v in rows[:8]:
    if v["calls"]:
        print(f"    {k:24s} calls={v['calls']:7d} {v['seconds']*1000/steps:9.3f} ms/step "
              f"cpu={v['thread_cpu_seconds']*1000/steps:8.3f}")
PY

echo "######## CHAIN27: end-to-end"
/home/bo/vast_profiling_problem/.venv/bin/python - "$OUT/tp4_layerwise/cold.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(f"  dur={d.get('duration',0):.1f}s   (non-layerwise 324.4 s, none 296.9 s)")
except Exception as e:
    print("  MISSING:", e)
PY
echo "######## CHAIN27: done $(date +%H:%M:%S)"
