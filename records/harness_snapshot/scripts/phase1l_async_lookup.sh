#!/usr/bin/env bash
# 1l -- take the blocking LOOKUP round trip off the scheduler's critical path.
#
# WHAT 1j/1k ESTABLISHED.  The common +5.7 ms/step that MP and IP both pay is
# the vLLM EngineCore thread blocked on a synchronous round trip to the LMCache
# server, once per admitted request, inside get_num_new_matched_tokens:
#     sub_submit_lookup   7.357 ms/step wall vs 0.083 ms/step thread CPU
#     sub_check_result    0.424 ms/step wall vs 0.015 ms/step thread CPU
# 98.9% waiting, not computing.  ~0.1 calls/step, so ~73 ms per admitted request.
#
# WHAT 1l ASKS.  Does removing the wait actually pay?  Two facts say it should:
# the awaited value is discarded (fut.result() in maybe_submit_lookup_request is
# not bound to anything; the loop only catches TimeoutError), and vLLM already
# has a defer contract -- get_num_new_matched_tokens may return None for "ask me
# again later", which is the Deferred counter in the stat line.  One fact says it
# may not: the IP connector already does the lookup asynchronously and is SLOWER
# (97.5 vs 91.0), and arm 1g ruled out the backoff sleep as that cost.  This is
# a test, not a fix.  It is set up so both outcomes are readable.
#
# WHAT CHANGES.  Only when the scheduler thread waits.  Same LOOKUP, same key,
# same servers, same point in the request's life:
#     stock:  send LOOKUP -> BLOCK for ack -> send QUERY -> BLOCK -> answer
#     1l:     send LOOKUP -> return None (defer) -> next step: ack arrived?
#             -> send QUERY -> BLOCK -> answer
# QUERY is never sent before the LOOKUP ack is observed, so the server-side
# ordering the stock code relies on (LOOKUP locks chunks, QUERY reports them) is
# preserved by construction rather than by luck.
#
# PRE-REGISTERED READING (written before the run, so it cannot be fitted after):
#   ms/step -> ~85.3, Deferred > 0   the wait WAS the tax; the fix is real and
#                                    the deferral it costs is affordable.
#   ms/step stays ~91.0, Deferred>0  the wait is not the tax, or deferral costs
#                                    back what it saves.  Same conclusion IP's
#                                    async path already hints at, now on MP with
#                                    one variable changed instead of twenty.
#   ms/step > 91.0                   deferral is more expensive than waiting;
#                                    that is the answer to "改成异步能解决吗" and
#                                    it is NO, with a mechanism.
#   Deferred == 0                    the patch did not take effect.  VOID, and
#                                    the assertions below should have caught it.
#
# NOT gated on duration, unlike 1j/1k: 1j/1k had to prove the instrument did not
# move the thing it measures, but 1l is SUPPOSED to move it.  The comparability
# gate that still applies is the KV pool size, and it is enforced.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
CONC="${CONC:-1000}"
PASSES="${PASSES:-cold}"
L1_GB="${L1_GB:-8}"
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

dir=$OUT/1l_async_lookup; mkdir -p "$dir" "$dir/timers"
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
cfg = KVTransferConfig(kv_connector="AsyncLookupMPConnector",
                       kv_connector_module_path="timedconn.async_mp_connector",
                       kv_role="kv_both")
try:
    cls = KVConnectorFactory.get_connector_class(cfg)
except Exception as e:
    print("[1l] ABORT: factory cannot load AsyncLookupMPConnector:", e); sys.exit(1)
if inspect.isabstract(cls):
    print("[1l] ABORT: AsyncLookupMPConnector is abstract."); sys.exit(1)
from timedconn.timed_mp_connector import TimedMPConnector
# 1l must be 1k's instrument plus one change, not a different connector: the
# chain is AsyncLookup -> Timed -> LMCacheMPConnector, so every hook number is
# directly comparable to 1j and 1k.
if cls.__mro__[1] is not TimedMPConnector or cls.__mro__[2] is not LMCacheMPConnector:
    print("[1l] ABORT: mro is", cls.__mro__[:3], "-- not AsyncLookup->Timed->MP."); sys.exit(1)
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
        print(f"[1l] ABORT: {name} differs from LMCacheMPConnector: {got} vs {want}")
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
    print("[1l] ABORT: the subclass differs from LMCacheMPConnector outside the")
    print("     timed hooks, so it is not the same connector: " + "; ".join(diff))
    sys.exit(1)
if not isinstance(inspect.getattr_static(cls, "transfer_intermediate_tensors"), property):
    print("[1l] ABORT: transfer_intermediate_tensors is no longer a property.")
    sys.exit(1)
print(f"[1l] AsyncLookupMPConnector resolves, subclasses TimedMPConnector which"
      f" subclasses MP, {len(_wrapped)} hooks timed,"
      f" {len(_skipped)} skipped as non-functions {_skipped},")
print(f"     hma/piecewise/layout identical, and no descriptor differs outside"
      f" the timed hooks.")
PY

BASE=( serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME"
       --tensor-parallel-size 8 --max-model-len 131072 --enable-prefix-caching
       --block-size=64 --max-num-seqs 256 )
KVCFG="{\"kv_connector\":\"AsyncLookupMPConnector\",\"kv_connector_module_path\":\"timedconn.async_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$MP_PORT,\"lmcache.mp.heartbeat_interval\":60.0}}"

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
echo "=== [1l_async_lookup] launching vllm $(date +%H:%M:%S) ==="
spawn "$dir/server.log" "$VLLM" "${BASE[@]}" --disable-hybrid-kv-cache-manager --kv-transfer-config "$KVCFG"
pid=$SPAWNED_PID
if ! wait_health "$dir/server.log" "$pid" 1500; then
  echo "[1l] FAILED TO START"; sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | tail -30; teardown; exit 1
fi
if grep -q "marked as init failed" "$dir/server.log" 2>/dev/null; then
  echo "[1l] ABORT: LMCache init failed -- would measure degraded mode"
  sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -A12 "Failed during post_init" | head -16
  teardown; exit 1
fi

nconn=$(grep -c "Creating v1 connector with name: AsyncLookupMPConnector" "$dir/server.log" 2>/dev/null || echo 0)
echo "  TimedMPConnector instantiations by vLLM's factory: $nconn (expect 9)"
if [ "$nconn" -lt 2 ]; then
  echo "[1l] ABORT: the factory did not build AsyncLookupMPConnector on both sides."; teardown; exit 1
fi
if ! sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -q "EngineCore.*Creating v1 connector with name: AsyncLookupMPConnector"; then
  echo "[1l] ABORT: no scheduler-side connector; half the hooks would be untimed."; teardown; exit 1
fi
ntimer=$(grep -c "TimedMPConnector attached" "$dir/server.log" 2>/dev/null || echo 0)
echo "  timer wrappers announcing themselves: $ntimer"
if [ "$ntimer" -lt 2 ]; then
  echo "[1l] ABORT: the timing subclass did not initialise."; teardown; exit 1
fi
# The patch is applied to live objects at connector construction, so no
# pre-flight can check it -- and an unpatched run is just 1k again, 20 minutes
# for a number we already have.  Assert on what the scheduler actually logged.
# NOTE sub_submit_lookup is still listed as installed and will then read ~0
# calls: 1k's timer wraps the stock blocking submit, which the async patch
# replaces outright.  sub_async_submit/sub_async_defer/sub_async_collect are the
# live ones.
sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" \
  | grep -m1 "role=SCHEDULER report_every=.* sub_timers=" | tee "$dir/sub_timers.txt"
for want in sub_get_tracker sub_create_key sub_submit_lookup sub_check_result; do
  if ! grep -q "$want" "$dir/sub_timers.txt" 2>/dev/null; then
    echo "[1l] ABORT: $want was not installed; the split would be incomplete."
    sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -i "TIMER:" | head -5
    teardown; exit 1
  fi
done
echo "  all four sub-timers installed on the scheduler."
sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" \
  | grep -m1 "AsyncLookupMPConnector attached .*role=SCHEDULER" | tee "$dir/async_patch.txt"
if ! grep -q "async_lookup=True" "$dir/async_patch.txt" 2>/dev/null; then
  echo "[1l] ABORT: the scheduler connector did not get the async patch;"
  echo "     this would silently re-run 1k and answer nothing."
  teardown; exit 1
fi
echo "  async lookup patch active on the scheduler."

grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"
pool=$(grep -o "GPU KV cache size: [0-9,]*" "$dir/server.log" | tail -1 | tr -cd '0-9')
echo "  pool=${pool:-unknown} (must be 13724416 to compare with 1e)"
if [ "${pool:-0}" -ne 13724416 ]; then
  echo "[1l] ABORT: pool is ${pool}, not 13,724,416; 1j and 1e are not comparable."; teardown; exit 1
fi

for c in $CONC; do
  for p in $PASSES; do
    echo "  [1l c=$c] $p pass  ($(date +%H:%M:%S))"
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
    # No pass/fail band here.  1j/1k had to prove the instrument did not move
    # what it measures and were gated on 1e's 686.0 s; 1l is SUPPOSED to move it.
    # The authority remains scripts/engine_rate.py --min-outstanding=0.
    print(f"       anchor, same end-to-end formula: stock MP 1e = 686.0 s / 93.7 ms/step")
    if dur < 650.0:
        print(f"       -> FASTER than stock MP by {686.0-dur:.1f} s. The wait was real cost.")
    elif dur > 720.0:
        print(f"       -> SLOWER than stock MP by {dur-686.0:.1f} s. Deferral costs more")
        print(f"          than the wait it removes.")
    else:
        print(f"       -> within noise of stock MP. Removing the wait did not pay.")
    print(f"       AUTHORITATIVE: scripts/engine_rate.py results/phase1 --min-outstanding=0")
except Exception as e:
    print(f"    -> {sys.argv[2]} RESULT MISSING:", e)
PYEOF
  done
done
# Deferred (num_skipped_waiting_reqs) is 0 in every block of stock MP.  If the
# async path took effect it must be positive; if it is 0 the patch did nothing
# and the ms/step above means nothing.
sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -o "Deferred: [0-9]*" \
  | sort | uniq -c | sort -rn | head -5 | tee "$dir/deferred.txt"
teardown
echo "=== 1l done $(date +%H:%M:%S) -> $dir ==="
echo "    timers: $dir/timers/*.json ; reports: grep TIMER $dir/server.log"
