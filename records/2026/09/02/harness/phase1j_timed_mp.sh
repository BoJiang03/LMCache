#!/usr/bin/env bash
# 1j -- MP again, with a wall clock on every LMCache hook.  ATTRIBUTION run.
#
# STATE OF THE QUESTION.  ms/step from vLLM's in-engine counters, 23 blocks over
# five sessions and both pool sizes, three non-overlapping levels:
#     85.3   no connector (1a x6, 1c x3) AND the do-nothing NullConnector (1i)
#     91.0   LMCache MP (1d x2, 1e x2)
#     97.5   LMCache IP (1b x6, 1f x2, 1g x1)
# 1i settled where the common +5.7 ms is NOT: vLLM's connector plumbing is free
# to 0.0 ms/step, aggregator and per-layer decorator included.  So the 5.7 ms is
# inside LMCache's own hook bodies, or in threads LMCache starts.  Three
# mechanisms read out of the source and sized by hand have been refuted by
# experiment, 0 for 3.  This arm stops guessing and reads the answer off a clock.
#
# WHAT IS HELD FIXED vs 1e (the 91.0 ms/step reference):
#   same model, TP=8, ISL=60000, OSL=1, c=1000, cold pass, seed 42
#   same --disable-hybrid-kv-cache-manager -> pool 13,724,416 (asserted)
#   same lmcache server (l1 8 GB, eviction noop) and configs/mp.yaml
#   the connector is LMCacheMPConnector; TimedMPConnector only subclasses it and
#   wraps each hook in two perf_counter() calls.  No hook body changes.
#
# THE VALIDITY GATE, PRE-REGISTERED.  ~100 ns of timer against a 91 ms step is
# nothing, but that is a prediction, not a measurement.  If this run does not
# land where 1e landed, the instrument moved what it measures and EVERY
# attribution number below it is void.  The gate prints the verdict; read it
# before reading anything else.
#
# CORRECTED AFTER THE FIRST RUN.  The gate originally divided the bench JSON's
# end-to-end total_token_throughput into the 8192-token step and required
# 89.0-93.0, a band calibrated on engine_rate.py's IN-ENGINE steady-state rate.
# Those are different quantities: the end-to-end aggregate includes ramp and
# drain and reads ~2.7 ms/step higher in every arm.  1j scored 93.3 and was
# declared VOID -- but 1e scores 93.7 the same way and would have failed its own
# gate.  On the like-for-like measures 1j is 91.0 ms/step (89,990 tok/s, digit
# for digit 1d and 1e) and 683.7 s cold against 1e's 686.0 s.  The instrument is
# free and 1j's attribution stands.  The gate now compares cold duration, and
# names engine_rate.py as the authority.
#
# HOW TO READ THE OUTPUT.  Each process prints, every 200 engine steps:
#     wall_ms/step   this process's own step time (worker: ~91 expected)
#     hooks_ms/step  sum of all hook wall time, per step
#     cpu_busy       rusage CPU / wall for the whole process, ALL threads
#     per-hook        ms per STEP (not per call) and calls per step
# Expected shapes:
#   hooks_ms/step ~ 5.7 on the workers  -> the tax is on-hook; the biggest hook
#       names it, and the next step is that hook's body.
#   hooks_ms/step ~ 0   but cpu_busy well above the no-connector arm -> the tax
#       is off-hook, in LMCache's background threads, and this instrument has
#       found that out rather than missing it.
#   hooks_ms/step ~ 0 and cpu_busy flat -> the cost is not CPU time at all
#       (GPU stream serialisation, or a lock the workers wait on elsewhere).
# There is no outcome that says nothing.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
CONC="${CONC:-1000}"
PASSES="${PASSES:-cold}"
L1_GB="${L1_GB:-8}"
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

dir=$OUT/1j_timed_mp; mkdir -p "$dir" "$dir/timers"
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
    print("[1j] ABORT: factory cannot load TimedMPConnector:", e); sys.exit(1)
if inspect.isabstract(cls):
    print("[1j] ABORT: TimedMPConnector is abstract."); sys.exit(1)
if cls.__mro__[1] is not LMCacheMPConnector:
    print("[1j] ABORT: not a direct subclass of LMCacheMPConnector; arms differ."); sys.exit(1)
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
        print(f"[1j] ABORT: {name} differs from LMCacheMPConnector: {got} vs {want}")
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
    print("[1j] ABORT: the subclass differs from LMCacheMPConnector outside the")
    print("     timed hooks, so it is not the same connector: " + "; ".join(diff))
    sys.exit(1)
if not isinstance(inspect.getattr_static(cls, "transfer_intermediate_tensors"), property):
    print("[1j] ABORT: transfer_intermediate_tensors is no longer a property.")
    sys.exit(1)
print(f"[1j] TimedMPConnector resolves, subclasses MP, {len(_wrapped)} hooks timed,"
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
echo "=== [1j_timed_mp] launching vllm $(date +%H:%M:%S) ==="
spawn "$dir/server.log" "$VLLM" "${BASE[@]}" --disable-hybrid-kv-cache-manager --kv-transfer-config "$KVCFG"
pid=$SPAWNED_PID
if ! wait_health "$dir/server.log" "$pid" 1500; then
  echo "[1j] FAILED TO START"; sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | tail -30; teardown; exit 1
fi
if grep -q "marked as init failed" "$dir/server.log" 2>/dev/null; then
  echo "[1j] ABORT: LMCache init failed -- would measure degraded mode"
  sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -A12 "Failed during post_init" | head -16
  teardown; exit 1
fi

nconn=$(grep -c "Creating v1 connector with name: TimedMPConnector" "$dir/server.log" 2>/dev/null || echo 0)
echo "  TimedMPConnector instantiations by vLLM's factory: $nconn (expect 9)"
if [ "$nconn" -lt 2 ]; then
  echo "[1j] ABORT: the factory did not build TimedMPConnector on both sides."; teardown; exit 1
fi
if ! sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -q "EngineCore.*Creating v1 connector with name: TimedMPConnector"; then
  echo "[1j] ABORT: no scheduler-side connector; half the hooks would be untimed."; teardown; exit 1
fi
ntimer=$(grep -c "TimedMPConnector attached" "$dir/server.log" 2>/dev/null || echo 0)
echo "  timer wrappers announcing themselves: $ntimer"
if [ "$ntimer" -lt 2 ]; then
  echo "[1j] ABORT: the timing subclass did not initialise."; teardown; exit 1
fi

grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"
pool=$(grep -o "GPU KV cache size: [0-9,]*" "$dir/server.log" | tail -1 | tr -cd '0-9')
echo "  pool=${pool:-unknown} (must be 13724416 to compare with 1e)"
if [ "${pool:-0}" -ne 13724416 ]; then
  echo "[1j] ABORT: pool is ${pool}, not 13,724,416; 1j and 1e are not comparable."; teardown; exit 1
fi

for c in $CONC; do
  for p in $PASSES; do
    echo "  [1j c=$c] $p pass  ($(date +%H:%M:%S))"
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
echo "=== 1j done $(date +%H:%M:%S) -> $dir ==="
echo "    timers: $dir/timers/*.json ; reports: grep TIMER $dir/server.log"
