#!/usr/bin/env bash
# Phase 2 — is "LMCache MP is slower than IP" really about MP, or about the
# hybrid KV cache manager that MP forces off?
#
# LMCacheMPConnector hard-requires --disable-hybrid-kv-cache-manager
#   (lmcache/integration/vllm/lmcache_mp_connector_0201.py:81)
# VAST's IP config does NOT pass it. On gpt-oss-120b (18 sliding + 18 full
# layers) that one flag alone moves the GPU KV pool by a measured 1.88x
# (phase 0: 25,798,386 -> 13,724,416 tokens).
#
# Runs (ISL=120k, mirroring the PDF's 4-way matrix; no L2 needed for the core test):
#   A ip_hybridON   IP exactly as VAST ran it        (allocator ON)
#   B ip_hybridOFF  IP forced to MP's allocator      (allocator OFF)  <- the control
#   C mp            MP as VAST ran it                (allocator OFF, forced)
# If C ~= B and both << A, the gap is the flag, not MP's IO path.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase2; mkdir -p "$OUT"
CONC="${CONC:-100 200 300 400 500 600}"
ISL="${ISL:-120000}"
L1_GB="${L1_GB:-700}"          # VAST used 1600; this box has ~1422 GB available
                               # and no swap, and hosts other tenants.
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

BASE=( serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME"
       --tensor-parallel-size 8 --max-model-len 131072 --enable-prefix-caching
       --block-size=64 --max-num-seqs 256 )

bench_isl() {   # same as bench_point but with a settable input length
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
  grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"
  local c; for c in $CONC; do bench_isl "$tag" "$dir" "$c"; done
  teardown
}

# ---- A: IP as VAST ran it (hybrid allocator ON) ----
export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/ip_no_l2.yaml
run_cfg A_ip_hybridON "${BASE[@]}" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# ---- B: same IP, allocator forced OFF (the control for MP) ----
run_cfg B_ip_hybridOFF "${BASE[@]}" --disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# ---- C: MP as VAST ran it ----
unset LMCACHE_CONFIG_FILE
mkdir -p "$OUT/C_mp"
echo "=== [C_mp] launching lmcache server (l1=${L1_GB}GB) $(date +%H:%M:%S) ==="
spawn "$OUT/C_mp/lmcache_server.log" "$VENV/bin/lmcache" server \
  --host 127.0.0.1 --port "$MP_PORT" \
  --l1-size-gb "$L1_GB" \
  --eviction-policy noop \
  --eviction-trigger-watermark 0.8 --eviction-ratio 0.2 \
  --chunk-size 8192 \
  --l2-prefetch-max-in-flight 4 \
  --max-gpu-workers 8 --max-cpu-workers 8 \
  --worker-reap-timeout-seconds 180 \
  --l1-align-bytes 1048576
sleep 30
export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/mp.yaml
run_cfg C_mp "${BASE[@]}" --disable-hybrid-kv-cache-manager \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$MP_PORT,\"lmcache.mp.heartbeat_interval\":60.0}}"

echo "=== phase2 done $(date +%H:%M:%S) -> $OUT ==="
