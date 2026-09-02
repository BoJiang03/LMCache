#!/usr/bin/env bash
# Phase 1 (rest) — 1b re-run with a valid LMCache config, then 1c for attribution.
#
#   1a  plain vLLM                              pool 25,798,626  (already done)
#   1b  vLLM + LMCache IP, no tier              pool 13,724,160  <- vLLM auto-disables HMA
#   1c  plain vLLM + --disable-hybrid-kv-...    pool 13,724,160  <- same pool, no LMCache
#
#   1a vs 1c = cost of the halved pool alone
#   1c vs 1b = cost of the LMCache connector alone, at an identical pool
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
CONC="${CONC:-100 300 600 1000 1500}"
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

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
  # Never benchmark a silently-degraded LMCache again.
  if grep -q "marked as init failed" "$dir/server.log" 2>/dev/null; then
    echo "[$tag] ABORT: LMCache init failed -- would measure degraded mode, not LMCache"
    sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -A12 "Failed during post_init" | head -16
    teardown; return 1
  fi
  grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"
  local c; for c in $CONC; do bench_point "$tag" "$dir" "$c"; done
  teardown
}

export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/lmcache_gpu_only.yaml
run_cfg 1b_vllm_lmcache "${BASE[@]}" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

unset LMCACHE_CONFIG_FILE
run_cfg 1c_vllm_nohybrid "${BASE[@]}" --disable-hybrid-kv-cache-manager

echo "=== phase1 rest done $(date +%H:%M:%S) ==="
