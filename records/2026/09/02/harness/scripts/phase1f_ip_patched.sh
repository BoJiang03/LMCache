#!/usr/bin/env bash
# 1f -- IP connector with the async-lookup-client fix applied.
#
# Identical to phase1_control_1b.sh in every respect except the LMCache source:
# same config (configs/lmcache_gpu_only.yaml), same vLLM flags, same ISL, same
# c=1000.  The only variable is commit 8ea23cd1 in $LMCACHE_SRC, which moves the
# lookup backoff out from under self.lock and makes it O(1) per scheduler pass
# instead of O(pending requests).
#
# PRE-REGISTERED PREDICTION (written before the run, records/2026/09/02):
#   baseline 1b unpatched, c=1000 warm : 724.1 s / 82,864 tok/s / P99 713.0 s
#   MP 1e, same pool, never defers     : 681.5 s / 88,039 tok/s / P99 672.0 s
#   no connector 1c, same pool         : 626.6 s / 95,755 tok/s / P99 616.1 s
#   -> mechanism confirmed if 1f lands in 680-695 s (most of the 42.6 s gap
#      between IP and MP recovered).
#   -> mechanism REFUTED if 1f >= 715 s.
#   -> partial if 695-715 s.
# 1f is NOT expected to reach 1c: the ~9% connector tax common to IP and MP is a
# different mechanism and this patch does not touch it.
#
# Results land in 1f_ip_patched/ so the unpatched 1b JSONs stay untouched.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
CONC="${CONC:-1000}"
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

# Fail before burning 25 minutes if the interpreter would import unpatched
# LMCache -- this arm is meaningless without the patch and would silently be a
# second copy of 1b.
$PY - <<'PY' || exit 1
import inspect, sys
import lmcache.v1.lookup_client.lmcache_async_lookup_client as m
src = inspect.getsource(m.LMCacheAsyncLookupClient.lookup_cache)
if "_yield_to_lookup_threads" not in src:
    print("[1f] ABORT: lmcache is NOT patched; this would re-measure 1b.")
    print("[1f] imported from:", m.__file__)
    sys.exit(1)
print("[1f] patch confirmed in", m.__file__)
PY

BASE=( serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME"
       --tensor-parallel-size 8 --max-model-len 131072 --enable-prefix-caching
       --block-size=64 --max-num-seqs 256 )

run_cfg() {
  local tag="$1"; shift
  local dir="$OUT/$tag"; mkdir -p "$dir"
  echo "=== [$tag] launching $(date +%H:%M:%S) ==="
  spawn "$dir/server.log" "$VLLM" "$@"; local pid=$SPAWNED_PID
  if ! wait_health "$dir/server.log" "$pid" 1500; then
    echo "[$tag] FAILED TO START"; teardown; return 1
  fi
  if grep -q "marked as init failed" "$dir/server.log" 2>/dev/null; then
    echo "[$tag] ABORT: LMCache init failed -- would measure degraded mode, not LMCache"
    sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -A12 "Failed during post_init" | head -16
    teardown; return 1
  fi
  grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"
  pool=$(grep -o "GPU KV cache size: [0-9,]*" "$dir/server.log" | tail -1 | tr -cd '0-9')
  echo "  pool=${pool:-unknown} (must be 13724416 to be comparable with 1b)"
  if [ "${pool:-0}" -ne 13724416 ]; then
    echo "[1f] ABORT: pool is ${pool}, not 1b's 13,724,416; the arms are not comparable."
    teardown; return 1
  fi
  local c; for c in $CONC; do bench_point "$tag" "$dir" "$c"; done
  # Mechanism check, independent of wall clock: how much did the scheduler defer?
  echo "  Deferred lines: $(grep -c 'Deferred:' "$dir/server.log" 2>/dev/null || echo 0)" \
       "max: $(grep -o 'Deferred: [0-9]*' "$dir/server.log" | grep -o '[0-9]*' | sort -n | tail -1)"
  teardown
}

export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/lmcache_gpu_only.yaml
run_cfg 1f_ip_patched "${BASE[@]}" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
echo "=== 1f done $(date +%H:%M:%S) -> $OUT/1f_ip_patched ==="
