#!/usr/bin/env bash
# 1g -- IP connector with the engine-thread lookup backoff set to ZERO.
#
# WHY THIS ARM EXISTS.  1f tested the hypothesis that the IP-only surcharge came
# from LMCacheAsyncLookupClient sleeping 10 ms *per pending request* under
# self.lock, i.e. an O(pending) stall per scheduling pass.  1f applied a patch
# that collapses those to one sleep per pass and releases the lock, and it
# measured 728.1 s vs unpatched 1b's 727.7 s -- no change at all.  That result
# only makes sense if the scheduler was already calling lookup_cache about ONCE
# per pass, so there was nothing to collapse.  The upper bound of "926 pending x
# 10 ms = 9.2 s of dead engine time in one pass" was never realised.
#
# What is left is one 10 ms sleep per scheduling pass, on the engine thread.
# Sizing it from the in-engine counters (steps are 8192 tokens each):
#     no connector 1c : 96,000 tok/s -> 11.7 steps/s -> 85.4 ms/step
#     MP 1e           : 90,000 tok/s -> 11.0 steps/s -> 91.0 ms/step  (+5.6)
#     IP 1b           : 84,000 tok/s -> 10.2 steps/s -> 97.6 ms/step  (+12.2)
# so the IP-only part is +6.6 ms/step -- the size of a 10 ms sleep firing on
# about two passes in three.
#
# THE ARM.  configs/lmcache_gpu_only_nobackoff.yaml is byte-identical to the
# config 1b and 1f used except for two added lines:
#     extra_config:
#       lookup_backoff_time: 0.0
# No LMCache source change.  sleep(0) is a bare GIL yield, so the scheduler
# returns to the forward pass immediately and collects the lookup answer on the
# next pass instead of blocking for it.
#
# PRE-REGISTERED PREDICTION (written before the run):
#   1b IP, backoff 0.01, c=1000 cold : 727.7 s / 82,454 tok/s   <- the baseline
#   1f IP, patched, backoff 0.01     : 728.1 s / 82,407 tok/s   <- no change
#   1e MP, same pool, no async lookup: 686.0 s / 87,462 tok/s   <- the floor for
#                                                                  this arm
#   1c no connector, same pool       : 623.0 s / 96,314 tok/s
#   -> CONFIRMED if 1g lands in 680-700 s, i.e. the IP-only surcharge is the
#      engine-thread sleep and IP falls to MP's level.  The fix then is to stop
#      sleeping on the engine thread at all, not to throttle the sleep.
#   -> REFUTED if 1g >= 715 s.  The IP-only 6.6 ms/step is then something else
#      and both the 1f patch and this hypothesis are dead ends.
#   -> partial if 700-715 s.
# 1g is NOT expected to reach 1c: the ~9% tax common to IP and MP (+5.6 ms/step,
# present with Deferred == 0) is a different mechanism this does not touch.
#
# Cold pass only.  Cold-vs-cold is the comparison with the tightest measured
# reproducibility we have: 1b 727.7 s vs 1f 728.1 s, 0.06% apart.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
CONC="${CONC:-1000}"
PASSES="${PASSES:-cold}"
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

CFG=$REPRO_ROOT/configs/lmcache_gpu_only_nobackoff.yaml

# Fail before burning 20 minutes if the knob does not actually reach the client.
# Without this the arm silently re-measures 1b.
$PY - "$CFG" <<'PY' || exit 1
import sys
from lmcache.v1.config import LMCacheEngineConfig
c = LMCacheEngineConfig.from_file(sys.argv[1])
ec = getattr(c, "extra_config", None) or {}
v = ec.get("lookup_backoff_time", None)
if v != 0.0:
    print(f"[1g] ABORT: lookup_backoff_time is {v!r}, not 0.0; this would re-measure 1b.")
    sys.exit(1)
print("[1g] lookup_backoff_time=0.0 confirmed from", sys.argv[1])
PY

BASE=( serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME"
       --tensor-parallel-size 8 --max-model-len 131072 --enable-prefix-caching
       --block-size=64 --max-num-seqs 256 )

bench_passes() {   # like lib.sh bench_point but honours $PASSES
  local tag="$1" out="$2" c="$3" p
  local common=( --backend vllm --base-url "http://127.0.0.1:$PORT"
    --model "$MODEL" --served-model-name "$SERVED_NAME" --tokenizer "$MODEL"
    --dataset-name random --random-input-len 60000 --random-output-len 1
    --random-range-ratio 0.0 --ignore-eos --seed 42
    --num-prompts "$c" --max-concurrency "$c"
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 99
    --save-result --result-dir "$out" )
  for p in $PASSES; do
    echo "  [$tag c=$c] $p pass  ($(date +%H:%M:%S))"
    $VLLM bench serve "${common[@]}" --result-filename "c${c}_${p}.json" \
        > "$out/c${c}_${p}.log" 2>&1
    $PY - "$out/c${c}_${p}.json" "$p" <<'PYEOF'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(f"    -> {sys.argv[2]} dur={d.get('duration',0):.1f}s "
          f"tok/s={d.get('total_token_throughput',0):.0f} "
          f"p99_ttft={d.get('p99_ttft_ms',0)/1000:.1f}s "
          f"mean_ttft={d.get('mean_ttft_ms',0)/1000:.1f}s")
except Exception as e:
    print(f"    -> {sys.argv[2]} RESULT MISSING:", e)
PYEOF
  done
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
    echo "[$tag] ABORT: LMCache init failed -- would measure degraded mode, not LMCache"
    sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -A12 "Failed during post_init" | head -16
    teardown; return 1
  fi
  grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"
  pool=$(grep -o "GPU KV cache size: [0-9,]*" "$dir/server.log" | tail -1 | tr -cd '0-9')
  echo "  pool=${pool:-unknown} (must be 13724416 to be comparable with 1b/1f)"
  if [ "${pool:-0}" -ne 13724416 ]; then
    echo "[1g] ABORT: pool is ${pool}, not 13,724,416; the arms are not comparable."
    teardown; return 1
  fi
  local c; for c in $CONC; do bench_passes "$tag" "$dir" "$c"; done
  echo "  Deferred max: $(grep -o 'Deferred: [0-9]*' "$dir/server.log" | grep -o '[0-9]*' | sort -n | tail -1)"
  teardown
}

export LMCACHE_CONFIG_FILE=$CFG
run_cfg 1g_ip_nobackoff "${BASE[@]}" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
echo "=== 1g done $(date +%H:%M:%S) -> $OUT/1g_ip_nobackoff ==="
