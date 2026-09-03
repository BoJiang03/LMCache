#!/usr/bin/env bash
# 1e -- LMCache MP with the KV pool forced DOWN to IP's operating point.
#
# WHY THIS ARM EXISTS
# The question is whether the "LMCache makes GPU-only slower" penalty is
# specific to the IP connector or is present in MP too.  After 1d we have:
#
#                      pool 25,798,626      pool 13,724,416
#   no connector       1a                   1c
#   IP connector       (impossible)         1b
#   MP connector       1d                   1e  <- THIS
#
# 1d/1a is MP's connector tax, but at a pool IP can never reach, so it is not
# directly comparable to 1b/1c.  1e closes that: 1e/1c and 1b/1c are the same
# ratio measured at the same pool, same flags, same workload, differing only in
# which LMCache connector is attached.  1b vs 1e is the head-to-head.
#
# --disable-hybrid-kv-cache-manager is also exactly what VAST passes for MP
# (PDF page 4), so this arm doubles as their MP configuration.
#
# L1 stays tiny (8 GB, eviction noop) so MP stores nothing steady-state, which
# is the closest analogue to IP's local_cpu:false.  "GPU only" means no tier.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
# c=1000 first: it is the decisive point (saturated, cross-session drift ~1%),
# so the answer to "is this IP-specific?" lands before the cheap c=200 point.
CONC="${CONC:-1000 200}"
L1_GB="${L1_GB:-8}"
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

avail=$(free -g | awk '/^Mem:/{print $7}')
need=$((L1_GB + 250))
if (( avail < need )); then
  echo "ABORT: ${avail}GB available, need ${need}GB."; exit 1
fi
echo "memory check ok: ${avail}GB available, need ${need}GB"

BASE=( serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME"
       --tensor-parallel-size 8 --max-model-len 131072 --enable-prefix-caching
       --block-size=64 --max-num-seqs 256 )
MP_KVCFG="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$MP_PORT,\"lmcache.mp.heartbeat_interval\":60.0}}"

dir=$OUT/1e_mp_nohybrid; mkdir -p "$dir"

echo "=== lmcache server (l1=${L1_GB}GB) $(date +%H:%M:%S) ==="
spawn "$dir/lmcache_server.log" "$VENV/bin/lmcache" server \
  --host 127.0.0.1 --port "$MP_PORT" --l1-size-gb "$L1_GB" \
  --eviction-policy noop --eviction-trigger-watermark 0.8 --eviction-ratio 0.2 \
  --chunk-size 8192 --l2-prefetch-max-in-flight 4 \
  --max-gpu-workers 8 --max-cpu-workers 8 \
  --worker-reap-timeout-seconds 180 --l1-align-bytes 1048576
# Port listening is the authoritative readiness signal; timing out must be fatal
# rather than falling through to a vLLM that cannot reach the server.
mp_pid=$SPAWNED_PID
t=0; up=0
while (( t < 300 )); do
  if ss -ltn 2>/dev/null | grep -q "127.0.0.1:$MP_PORT"; then up=1; break; fi
  kill -0 "$mp_pid" 2>/dev/null || { echo "  lmcache server died after ${t}s:"; tail -30 "$dir/lmcache_server.log"; teardown; exit 1; }
  sleep 5; t=$((t+5))
done
if (( up == 0 )); then
  echo "  ABORT: lmcache server never listened on $MP_PORT after ${t}s"; tail -30 "$dir/lmcache_server.log"; teardown; exit 1
fi
echo "  lmcache server listening on $MP_PORT after ${t}s"

export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/mp.yaml
echo "=== [1e_mp_nohybrid] launching vllm $(date +%H:%M:%S) ==="
spawn "$dir/server.log" "$VLLM" "${BASE[@]}" --disable-hybrid-kv-cache-manager --kv-transfer-config "$MP_KVCFG"
pid=$SPAWNED_PID
if ! wait_health "$dir/server.log" "$pid" 1500; then
  echo "[1e] FAILED TO START"; teardown; exit 1
fi
if grep -q "marked as init failed" "$dir/server.log" 2>/dev/null; then
  echo "[1e] ABORT: LMCache init failed -- would measure degraded mode"
  sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -A12 "Failed during post_init" | head -16
  teardown; exit 1
fi
grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"

# This arm is only meaningful at IP's pool.  If the flag did not take effect we
# would silently be re-measuring 1d -- stop instead.  Assertion is inverted
# relative to 1d on purpose.
pool=$(grep -o "GPU KV cache size: [0-9,]*" "$dir/server.log" | tail -1 | tr -cd '0-9')
echo "  pool=${pool:-unknown} (expect 13724416; 25798626 means the flag did not take effect)"
if [ "${pool:-0}" -gt 20000000 ]; then
  echo "[1e] ABORT: pool is ${pool}, the hybrid allocator is still ON."
  sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -iE "hybrid|SupportsHMA|disable_hybrid" | head -10
  teardown; exit 1
fi
if [ "${pool:-0}" -lt 1000000 ]; then
  echo "[1e] ABORT: pool unreadable (${pool})."; teardown; exit 1
fi

for c in $CONC; do bench_point 1e_mp_nohybrid "$dir" "$c"; done
teardown
echo "=== 1e done $(date +%H:%M:%S) -> $dir ==="
