#!/usr/bin/env bash
# 1d -- "GPU Only + LMCache" measured through the MP connector.
#
# WHY THIS ARM EXISTS
# VAST's chart curve 1b is "GPU Only + LMCache".  Reproducing it with the IP
# connector (our 1b) changes TWO things at once:
#   - LMCache is attached                      <- the thing we want to measure
#   - the GPU KV pool halves, 25,798,626 -> 13,724,416 tokens
#     because LMCacheConnectorV1 does not subclass SupportsHMA, so vLLM turns
#     off its hybrid allocator and provisions all 36 layers for the full 131072
#     context instead of giving the 18 sliding-window layers a 128-token ring.
#
# LMCacheMPConnector DOES subclass SupportsHMA (lmcache_mp_connector.py:273), so
# running MP *without* --disable-hybrid-kv-cache-manager keeps the full pool.
# That gives the cell our matrix is missing:
#
#                      pool 25,798,626      pool 13,724,416
#   no connector       1a                   1c
#   IP connector       (impossible today)   1b
#   MP connector       1d  <- THIS          phase2 arm C
#
# 1d vs 1a is therefore the honest "with vs without LMCache" comparison: same
# pool, same vLLM flags, the connector is the only difference.  It is also the
# best available proxy for what fixing the IP adapter would buy.
#
# L1 is deliberately tiny.  "GPU only" means no offload tier; MP's server L1 is
# its store, so 8 GB + eviction-policy noop makes it fill once and then stop
# storing -- the closest analogue to IP's local_cpu:false.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
CONC="${CONC:-600 1000}"
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

dir=$OUT/1d_mp_gpu_only; mkdir -p "$dir"

echo "=== lmcache server (l1=${L1_GB}GB) $(date +%H:%M:%S) ==="
spawn "$dir/lmcache_server.log" "$VENV/bin/lmcache" server \
  --host 127.0.0.1 --port "$MP_PORT" --l1-size-gb "$L1_GB" \
  --eviction-policy noop --eviction-trigger-watermark 0.8 --eviction-ratio 0.2 \
  --chunk-size 8192 --l2-prefetch-max-in-flight 4 \
  --max-gpu-workers 8 --max-cpu-workers 8 \
  --worker-reap-timeout-seconds 180 --l1-align-bytes 1048576
# Readiness: the authoritative signal is the port actually listening -- MP has
# never run on this box and the log wording is unverified, so do not depend on
# grepping it.  Failing to detect readiness must be fatal, not a silent 300 s
# wait followed by a vLLM that cannot reach the server.
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
echo "=== [1d_mp_gpu_only] launching vllm $(date +%H:%M:%S) ==="
# NOTE: no --disable-hybrid-kv-cache-manager.  That flag is what VAST passes and
# what the legacy _0201/_0180 modules require; this module does not need it.
spawn "$dir/server.log" "$VLLM" "${BASE[@]}" --kv-transfer-config "$MP_KVCFG"
pid=$SPAWNED_PID
if ! wait_health "$dir/server.log" "$pid" 1500; then
  echo "[1d] FAILED TO START"; teardown; exit 1
fi
if grep -q "marked as init failed" "$dir/server.log" 2>/dev/null; then
  echo "[1d] ABORT: LMCache init failed -- would measure degraded mode"
  sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -A12 "Failed during post_init" | head -16
  teardown; exit 1
fi
grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"

# The whole point of this arm is the full pool.  If SupportsHMA did not take
# effect we would be re-measuring 1b with extra steps -- stop instead.
pool=$(grep -o "GPU KV cache size: [0-9,]*" "$dir/server.log" | tail -1 | tr -cd '0-9')
echo "  pool=${pool:-unknown} (expect 25798626; 13724416 means the allocator was disabled)"
if [ "${pool:-0}" -lt 20000000 ]; then
  echo "[1d] ABORT: pool is ${pool}, hybrid allocator did NOT stay on."
  sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -iE "hybrid|SupportsHMA|disable_hybrid" | head -10
  teardown; exit 1
fi

for c in $CONC; do bench_point 1d_mp_gpu_only "$dir" "$c"; done
teardown
echo "=== 1d done $(date +%H:%M:%S) -> $dir ==="
