#!/usr/bin/env bash
# 1k -- subdivide the 7.5 ms/step that 1j put inside get_num_new_matched_tokens.
#
# WHAT 1j ESTABLISHED (171 steady windows, 9 processes, wall 91.24 ms/step
# against 1e's uninstrumented 91.0, so the instrument is not the effect):
#     SCHEDULER hooks            8.50 ms/step
#       get_num_new_matched_tokens 7.51   <- one hook, ~0.1 calls/step
#       build_connector_meta       0.83
#     each of 8 WORKERS           0.75 ms/step, all hooks together
# The common +5.7 ms/step is on the scheduler, and it is essentially this one
# hook.  0.1 calls/step means tens of ms per admitted request, blocking the
# EngineCore loop.
#
# WHAT 1k ASKS.  The hook does four separable things, and they have four
# different fixes, so "which one" decides what to propose upstream:
#     sub_get_tracker    scheduler-side per-request bookkeeping
#     sub_create_key     tuple(token_ids) over the whole 60,000-token prompt
#     sub_submit_lookup  the blocking LOOKUP round trip (create_key is nested
#                        inside it, so the round trip alone is submit-create_key)
#     sub_check_result   the blocking QUERY_PREFETCH_STATUS round trip
# Each sub-timer records thread CPU beside wall clock.  time.thread_time()
# counts only the calling thread, so wall-minus-CPU separates "this call
# computed" from "this call waited on the server" -- a distinction the
# process-wide cpu_busy field cannot make, because the EngineCore's other
# threads keep running while the scheduler thread blocks.
#
# PRE-REGISTERED READING:
#   create_key CPU-heavy      -> the fix is to stop putting the whole prompt in
#                                the key; hashing/serialising 60k ids per
#                                request does not belong on the critical path.
#   submit/check wall >> CPU  -> the scheduler is idle-waiting on the MP server;
#                                the fix is to take the lookup off the critical
#                                path (submit, return, poll later), which is
#                                what the IP connector's async lookup client
#                                already tries to do.
#   both comparable           -> both must move, and the record says so.
#
# Everything else is 1j unchanged: same pool assertion, same descriptor gate,
# same validity gate against 1e's 91.0 ms/step.  Only the output dir differs.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
CONC="${CONC:-1000}"
PASSES="${PASSES:-cold}"
L1_GB="${L1_GB:-8}"
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

dir=$OUT/1k_timed_mp_sub; mkdir -p "$dir" "$dir/timers"
# The 8 workers and the EngineCore inherit these.
export PYTHONPATH="$REPRO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LMC_TIMER_DIR="$dir/timers"
export LMC_TIMER_REPORT_EVERY="${LMC_TIMER_REPORT_EVERY:-200}"

avail=$(free -g | awk '/^Mem:/{print $7}')
need=$((L1_GB + 250))
if (( avail < need )); then echo "ABORT: ${avail}GB available, need ${need}GB."; exit 1; fi
echo "memory check ok: ${avail}GB available, need ${need}GB"

# Fail in seconds, not after a 5-minute model load.
$PY - <<'PY' || exit 1
import sys, inspect
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import supports_hma
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
cfg = KVTransferConfig(kv_connector="TimedMPConnector",
                       kv_connector_module_path="timedconn.timed_mp_connector",
                       kv_role="kv_both")
try:
    cls = KVConnectorFactory.get_connector_class(cfg)
except Exception as e:
    print("[1k] ABORT: factory cannot load TimedMPConnector:", e); sys.exit(1)
if inspect.isabstract(cls):
    print("[1k] ABORT: TimedMPConnector is abstract."); sys.exit(1)
if cls.__mro__[1] is not LMCacheMPConnector:
    print("[1k] ABORT: not a direct subclass of LMCacheMPConnector; arms differ."); sys.exit(1)
# Every property that decides pool size, cudagraph mode and attention backend
# must read the same as the real MP connector, or 1j is not comparable to 1e.
for name, got, want in [
    ("supports_hma", supports_hma(cls), supports_hma(LMCacheMPConnector)),
    ("piecewise", cls.requires_piecewise_for_cudagraph({}),
                  LMCacheMPConnector.requires_piecewise_for_cudagraph({})),
    ("layout", cls.get_required_kvcache_layout({}),
               LMCacheMPConnector.get_required_kvcache_layout({})),
]:
    if got != want:
        print(f"[1k] ABORT: {name} differs from LMCacheMPConnector: {got} vs {want}")
        sys.exit(1)
from timedconn.timed_mp_connector import _wrapped, _skipped
# The instrument must differ from LMCacheMPConnector in EXACTLY the wrapped
# hooks and nothing else.  The first version of the wrapper turned the
# @property transfer_intermediate_tensors into a method; a bound method is
# always truthy, so every worker died at init with "Connector enables
# transfer_query but server does not."  A descriptor-level diff catches that
# whole class of mistake before the model loads instead of 90 seconds in.
allowed = set(_wrapped) | {"__init__", "__doc__", "__dict__", "__module__",
                           "__abstractmethods__", "_abc_impl"}
diff = []
for name in set(dir(cls)) | set(dir(LMCacheMPConnector)):
    if name in allowed:
        continue
    a = inspect.getattr_static(cls, name, "<missing>")
    b = inspect.getattr_static(LMCacheMPConnector, name, "<missing>")
    if a is not b:
        diff.append(f"{name}: {type(a).__name__} vs {type(b).__name__}")
if diff:
    print("[1k] ABORT: the subclass differs from LMCacheMPConnector outside the")
    print("     timed hooks, so it is not the same connector: " + "; ".join(diff))
    sys.exit(1)
if not isinstance(inspect.getattr_static(cls, "transfer_intermediate_tensors"), property):
    print("[1k] ABORT: transfer_intermediate_tensors is no longer a property.")
    sys.exit(1)
print(f"[1k] TimedMPConnector resolves, subclasses MP, {len(_wrapped)} hooks timed,"
      f" {len(_skipped)} skipped as non-functions {_skipped},")
print(f"     hma/piecewise/layout identical, and no descriptor differs outside"
      f" the timed hooks.")
PY

BASE=( serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME"
       --tensor-parallel-size 8 --max-model-len 131072 --enable-prefix-caching
       --block-size=64 --max-num-seqs 256 )
KVCFG="{\"kv_connector\":\"TimedMPConnector\",\"kv_connector_module_path\":\"timedconn.timed_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$MP_PORT,\"lmcache.mp.heartbeat_interval\":60.0}}"

echo "=== lmcache server (l1=${L1_GB}GB) $(date +%H:%M:%S) ==="
spawn "$dir/lmcache_server.log" "$VENV/bin/lmcache" server \
  --host 127.0.0.1 --port "$MP_PORT" --l1-size-gb "$L1_GB" \
  --eviction-policy noop --eviction-trigger-watermark 0.8 --eviction-ratio 0.2 \
  --chunk-size 8192 --l2-prefetch-max-in-flight 4 \
  --max-gpu-workers 8 --max-cpu-workers 8 \
  --worker-reap-timeout-seconds 180 --l1-align-bytes 1048576
mp_pid=$SPAWNED_PID
t=0; up=0
while (( t < 300 )); do
  if ss -ltn 2>/dev/null | grep -q "127.0.0.1:$MP_PORT"; then up=1; break; fi
  kill -0 "$mp_pid" 2>/dev/null || { echo "  lmcache server died after ${t}s:"; tail -30 "$dir/lmcache_server.log"; teardown; exit 1; }
  sleep 5; t=$((t+5))
done
(( up == 1 )) || { echo "  ABORT: lmcache server never listened on $MP_PORT"; tail -30 "$dir/lmcache_server.log"; teardown; exit 1; }
echo "  lmcache server listening on $MP_PORT after ${t}s"

export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/mp.yaml
echo "=== [1k_timed_mp_sub] launching vllm $(date +%H:%M:%S) ==="
spawn "$dir/server.log" "$VLLM" "${BASE[@]}" --disable-hybrid-kv-cache-manager --kv-transfer-config "$KVCFG"
pid=$SPAWNED_PID
if ! wait_health "$dir/server.log" "$pid" 1500; then
  echo "[1k] FAILED TO START"; sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | tail -30; teardown; exit 1
fi
if grep -q "marked as init failed" "$dir/server.log" 2>/dev/null; then
  echo "[1k] ABORT: LMCache init failed -- would measure degraded mode"
  sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -A12 "Failed during post_init" | head -16
  teardown; exit 1
fi

nconn=$(grep -c "Creating v1 connector with name: TimedMPConnector" "$dir/server.log" 2>/dev/null || echo 0)
echo "  TimedMPConnector instantiations by vLLM's factory: $nconn (expect 9)"
if [ "$nconn" -lt 2 ]; then
  echo "[1k] ABORT: the factory did not build TimedMPConnector on both sides."; teardown; exit 1
fi
if ! sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -q "EngineCore.*Creating v1 connector with name: TimedMPConnector"; then
  echo "[1k] ABORT: no scheduler-side connector; half the hooks would be untimed."; teardown; exit 1
fi
ntimer=$(grep -c "TimedMPConnector attached" "$dir/server.log" 2>/dev/null || echo 0)
echo "  timer wrappers announcing themselves: $ntimer"
if [ "$ntimer" -lt 2 ]; then
  echo "[1k] ABORT: the timing subclass did not initialise."; teardown; exit 1
fi
# The sub-timers are installed on live objects at connector construction, so no
# pre-flight can check them -- but without them 1k is just 1j again, 20 minutes
# for a number we already have.  Assert on what the scheduler actually logged.
sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" \
  | grep -m1 "role=SCHEDULER report_every=.* sub_timers=" | tee "$dir/sub_timers.txt"
for want in sub_get_tracker sub_create_key sub_submit_lookup sub_check_result; do
  if ! grep -q "$want" "$dir/sub_timers.txt" 2>/dev/null; then
    echo "[1k] ABORT: $want was not installed; the split would be incomplete."
    sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -i "TIMER:" | head -5
    teardown; exit 1
  fi
done
echo "  all four sub-timers installed on the scheduler."

grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"
pool=$(grep -o "GPU KV cache size: [0-9,]*" "$dir/server.log" | tail -1 | tr -cd '0-9')
echo "  pool=${pool:-unknown} (must be 13724416 to compare with 1e)"
if [ "${pool:-0}" -ne 13724416 ]; then
  echo "[1k] ABORT: pool is ${pool}, not 13,724,416; 1j and 1e are not comparable."; teardown; exit 1
fi

for c in $CONC; do
  for p in $PASSES; do
    echo "  [1k c=$c] $p pass  ($(date +%H:%M:%S))"
    $VLLM bench serve --backend vllm --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL" --served-model-name "$SERVED_NAME" --tokenizer "$MODEL" \
      --dataset-name random --random-input-len 60000 --random-output-len 1 \
      --random-range-ratio 0.0 --ignore-eos --seed 42 \
      --num-prompts "$c" --max-concurrency "$c" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 99 \
      --save-result --result-dir "$dir" --result-filename "c${c}_${p}.json" \
      > "$dir/c${c}_${p}.log" 2>&1
    $PY - "$dir/c${c}_${p}.json" "$p" <<'PYEOF'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    tps=d.get('total_token_throughput',0); ms=1000*8192/tps if tps else float('nan')
    dur=d.get('duration',0)
    print(f"    -> {sys.argv[2]} dur={dur:.1f}s tok/s={tps:.0f} -> {ms:.1f} ms/step")
    # The gate must compare like with like.  The reference table (85.3 / 91.0 /
    # 97.5) is built by scripts/engine_rate.py from vLLM's IN-ENGINE steady-state
    # counters; total_token_throughput here is the bench's END-TO-END aggregate
    # over the whole pass, ramp and drain included, and runs ~2.7 ms/step higher
    # for every arm.  The first version of this gate compared the second against
    # a band calibrated on the first and reported a false FAIL on 1j at 93.3 --
    # 1e itself scores 93.7 by that formula and would have failed its own gate.
    # Compare cold duration against 1e's cold duration, which IS like for like.
    print(f"       end-to-end {ms:.1f} ms/step (1e scores {1000*8192/87462:.1f} the same way)")
    if 665.0 <= dur <= 707.0:
        print(f"       GATE PASS: {dur:.1f}s against 1e's 686.0s, within 3%.")
    else:
        print(f"       GATE FAIL: {dur:.1f}s against 1e's 686.0s. The instrument may")
        print(f"       have moved what it measures; check before using the attribution.")
    print(f"       AUTHORITATIVE: scripts/engine_rate.py --min-outstanding=0")
    print(f"       must put this arm at 89,990 tok/s = 91.0 ms/step, like 1d and 1e.")
except Exception as e:
    print(f"    -> {sys.argv[2]} RESULT MISSING:", e)
PYEOF
  done
done
teardown
echo "=== 1k done $(date +%H:%M:%S) -> $dir ==="
echo "    timers: $dir/timers/*.json ; reports: grep TIMER $dir/server.log"
