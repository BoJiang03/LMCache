#!/usr/bin/env bash
# Control re-run: 1b (vLLM + LMCacheConnectorV1, no storage tier) at c=1000,
# back-to-back with the 1a and 1c controls on the SAME machine state.
#
# Why: the finding-(1) headline currently pairs 1b measured Sep 2 11:49 against
# 1a measured Sep 1 15:30, and this box has been shown to swing a single point
# by 28% between sessions.  c=1000 is the stable point (1a and 1c agree to 0.6%
# there, and to 0.1% at c=1500 cold), so this is the measurement that lets the
# 1.16x be quoted without an asterisk.
#
# Results land in 1b_rerun/ so the original 1b JSONs stay untouched.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
CONC="${CONC:-1000}"
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
run_cfg 1b_rerun "${BASE[@]}" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
echo "=== 1b control done $(date +%H:%M:%S) ==="
