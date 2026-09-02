#!/usr/bin/env bash
# Phase 2 -- why is LMCache MP slower than IP in VAST's matrix?
#
# REDESIGNED 2026-09-02.  The previous design asked whether the gap came from
# MP being forced to --disable-hybrid-kv-cache-manager while IP kept the hybrid
# allocator.  That question is void: vllm 0.22.1 auto-disables the allocator for
# LMCacheConnectorV1 too (it does not subclass SupportsHMA), so BOTH arms of
# VAST's matrix ran with pool = 13,724,416 tokens.  Measured, not assumed.
#
# The real asymmetry is the storage tier.  From the PDF, verbatim:
#   MP: lmcache server --l1-size-gb 1600      <- a 1.6 TB CPU cache doing work
#   IP: local_cpu: false                      <- no tier at all; stores nothing
# So their "MP vs IP" is closer to "LMCache working" vs "LMCache idle".
# This phase tests that.
#
#   A  ip_notier   IP as VAST ran it        local_cpu:false, pool 13.7M
#   B  ip_cputier  IP with a real L1        local_cpu:true 62x8=496GB, pool 13.7M
#   C  mp_vast     MP as VAST ran it        l1=496GB, --disable-hybrid... , pool 13.7M
#   D  mp_hma      MP without that flag     l1=496GB, allocator ON, pool ~25.8M
#
#   B vs C  the fair MP-vs-IP comparison: same pool, both with a CPU tier.
#   A vs B  how much IP's tierless config flatters it.
#   C vs D  the 1.88x pool VAST gives up for free -- their MP module
#           (lmcache_mp_connector) subclasses SupportsHMA, so the flag they pass
#           is only needed by the legacy _0201/_0180 modules.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase2; mkdir -p "$OUT"
CONC="${CONC:-100 300 600}"
ISL="${ISL:-120000}"
L1_GB="${L1_GB:-496}"          # VAST used 1600. This box has ~1.2 TB free and
                               # no swap, and hosts other tenants' processes.
MIN_FREE_GB=$((L1_GB + 250))   # L1 slab + vLLM workers + headroom
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

avail=$(free -g | awk '/^Mem:/{print $7}')
if (( avail < MIN_FREE_GB )); then
  echo "ABORT: ${avail}GB available, need ${MIN_FREE_GB}GB for L1_GB=${L1_GB}."
  echo "Lower L1_GB or wait. (On 2026-09-01 an unchecked allocation OOM-killed"
  echo "two of root's k8s pods on this box.)"; exit 1
fi
echo "memory check ok: ${avail}GB available, need ${MIN_FREE_GB}GB"

BASE=( serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME"
       --tensor-parallel-size 8 --max-model-len 131072 --enable-prefix-caching
       --block-size=64 --max-num-seqs 256 )
MP_KVCFG="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$MP_PORT,\"lmcache.mp.heartbeat_interval\":60.0}}"
IP_KVCFG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

bench_isl() {   # bench_point, but with a settable input length
  local tag="$1" out="$2" c="$3" p
  local common=( --backend vllm --base-url "http://127.0.0.1:$PORT"
    --model "$MODEL" --served-model-name "$SERVED_NAME" --tokenizer "$MODEL"
    --dataset-name random --random-input-len "$ISL" --random-output-len 1
    --random-range-ratio 0.0 --ignore-eos --seed 42
    --num-prompts "$c" --max-concurrency "$c"
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 99
    --save-result --result-dir "$out" )
  for p in cold warm; do
    echo "  [$tag c=$c] $p pass ($(date +%H:%M:%S))"
    $VLLM bench serve "${common[@]}" --result-filename "c${c}_${p}.json" > "$out/c${c}_${p}.log" 2>&1
  done
  $PY - "$out/c${c}_warm.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(f"    -> warm p99_ttft={d.get('p99_ttft_ms',0)/1000:.1f}s "
          f"mean_ttft={d.get('mean_ttft_ms',0)/1000:.1f}s "
          f"total_tok/s={d.get('total_token_throughput',0):.0f}")
except Exception as e:
    print("    -> WARM RESULT MISSING:", e)
PY
}

run_cfg() {
  local tag="$1"; shift
  local dir="$OUT/$tag"; mkdir -p "$dir"
  echo "=== [$tag] launching $(date +%H:%M:%S) ==="
  spawn "$dir/server.log" "$VLLM" "$@"; local pid=$SPAWNED_PID
  if ! wait_health "$dir/server.log" "$pid" 1500; then
    echo "[$tag] FAILED TO START"; teardown; return 1
  fi
  if grep -q "marked as init failed" "$dir/server.log" 2>/dev/null; then
    echo "[$tag] ABORT: LMCache init failed -- would measure degraded mode"
    sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -A12 "Failed during post_init" | head -16
    teardown; return 1
  fi
  grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"
  local c; for c in $CONC; do bench_isl "$tag" "$dir" "$c"; done
  teardown
}

start_mp_server() {   # $1 = result dir; leaves the pid in MY_PIDS for teardown
  echo "=== lmcache server (l1=${L1_GB}GB) $(date +%H:%M:%S) ==="
  spawn "$1/lmcache_server.log" "$VENV/bin/lmcache" server \
    --host 127.0.0.1 --port "$MP_PORT" --l1-size-gb "$L1_GB" \
    --eviction-policy noop --eviction-trigger-watermark 0.8 --eviction-ratio 0.2 \
    --chunk-size 8192 --l2-prefetch-max-in-flight 4 \
    --max-gpu-workers 8 --max-cpu-workers 8 \
    --worker-reap-timeout-seconds 180 --l1-align-bytes 1048576
  local t=0
  while (( t < 300 )); do
    grep -qiE "listening|serving|ready|started" "$1/lmcache_server.log" 2>/dev/null && break
    kill -0 "$SPAWNED_PID" 2>/dev/null || { echo "  lmcache server died:"; tail -20 "$1/lmcache_server.log"; return 1; }
    sleep 5; t=$((t+5))
  done
  echo "  lmcache server up after ${t}s"
}

# ---- A: IP as VAST ran it (no storage tier) ----
export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/ip_no_l2_fixed.yaml
run_cfg A_ip_notier "${BASE[@]}" --kv-transfer-config "$IP_KVCFG"

# ---- B: IP with a real CPU tier ----
export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/ip_cpu_tier.yaml
run_cfg B_ip_cputier "${BASE[@]}" --kv-transfer-config "$IP_KVCFG"

# ---- C: MP as VAST ran it (allocator forced off) ----
unset LMCACHE_CONFIG_FILE
mkdir -p "$OUT/C_mp_vast"
if start_mp_server "$OUT/C_mp_vast"; then
  export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/mp.yaml
  run_cfg C_mp_vast "${BASE[@]}" --disable-hybrid-kv-cache-manager --kv-transfer-config "$MP_KVCFG"
else teardown; fi

# ---- D: MP with the hybrid allocator left on ----
unset LMCACHE_CONFIG_FILE
mkdir -p "$OUT/D_mp_hma"
if start_mp_server "$OUT/D_mp_hma"; then
  export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/mp.yaml
  run_cfg D_mp_hma "${BASE[@]}" --kv-transfer-config "$MP_KVCFG"
else teardown; fi

echo "=== phase2 done $(date +%H:%M:%S) -> $OUT ==="
