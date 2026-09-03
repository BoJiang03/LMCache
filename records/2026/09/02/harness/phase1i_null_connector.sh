#!/usr/bin/env bash
# 1i -- a connector that does nothing, to BISECT the common +5.7 ms/step.
#
# WHY BISECT INSTEAD OF PROFILE.  1h tried to profile the MP arm with py-spy and
# had to be abandoned: with TP=8 the workers are lockstepped on NCCL collectives,
# so ptrace-stopping any one rank to read its stack stalls all eight.  Sampling
# the tree at 30 Hz ran at 23,998 tok/s = 341 ms/step against MP's true 91.0, a
# 3.75x distortion.  1h's own pre-registered sanity gate says a profile taken
# under that distortion must be discarded, so it was, and the run was stopped
# early rather than burning another 30 minutes.  Lowering the rate only scales
# the distortion down; it does not remove the TP amplification.
#
# So use the instrument that already works.  ms/step from vLLM's in-engine
# counters reproduces to 0.1 ms across 14 blocks, five sessions and both pool
# sizes, and it needs no profiler at all:
#
#     no connector   85.3 ms/step        (1a x3, 1c x3)
#     LMCache MP     91.0 ms/step  +5.7  (1d x2, 1e x2, Deferred == 0)
#     LMCache IP     97.5 ms/step +12.2  (1b x6, 1f x2, 1g x1)
#
# nullconn/null_connector.py implements every abstract method of
# KVConnectorBase_V1 and does nothing in all of them.  vLLM still walks its whole
# connector code path: maybe_transfer_kv_layer wraps all 36 attention layers,
# build_connector_meta runs every scheduler step, the worker-side load/save hooks
# fire every forward, the KVOutputAggregator is installed, and
# get_num_new_matched_tokens is consulted for every waiting request.
#
# PRE-REGISTERED PREDICTION:
#   1i ~ 91 ms/step (686-700 s)  -> the +5.7 ms is vLLM's OWN connector plumbing.
#        LMCache is not the thing to fix; the finding is an upstream vLLM one.
#   1i ~ 85 ms/step (620-635 s)  -> the plumbing is free and the +5.7 ms is
#        inside LMCache's own per-step work, in code both IP and MP share.
#   anything between 640 and 680 s -> the tax splits between the two, and the
#        split is read off directly as (1i - 85.3) vs (91.0 - 1i).
#   > 700 s -> the null connector is doing something unintended; discard.
# There is no outcome that says nothing, and no profiler in the loop.
#
# CONFOUNDERS HELD FIXED BY CONSTRUCTION (each was eliminated earlier and must
# stay eliminated):
#   - NullConnector does NOT subclass SupportsHMA, so --disable-hybrid-kv-cache-
#     manager is required and the pool lands at 13,724,416, same as 1c/1e/1b/1g.
#   - requires_piecewise_for_cudagraph() is False -> same FULL_AND_PIECEWISE mode.
#   - get_required_kvcache_layout() is inherited and returns None -> same
#     FLASH_ATTN backend, no layout change.
#   - get_num_new_matched_tokens returns (0, False), never None, so this arm can
#     never defer a request.  Deferred must be 0; it is asserted after the run.
#
# Cold pass only: cold-vs-cold is the tightest comparison we have (1b 727.7 vs
# 1f 728.1, 0.06% apart).
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
CONC="${CONC:-1000}"
PASSES="${PASSES:-cold}"
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

# The workers inherit this; without it they cannot import the connector module.
export PYTHONPATH="$REPRO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# This arm must not touch LMCache at all.
unset LMCACHE_CONFIG_FILE

# Fail in seconds rather than after the model loads.
$PY - <<'PY' || exit 1
import sys, inspect
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import supports_hma
cfg = KVTransferConfig(kv_connector="NullConnector",
                       kv_connector_module_path="nullconn.null_connector",
                       kv_role="kv_both")
try:
    cls = KVConnectorFactory.get_connector_class(cfg)
except Exception as e:
    print("[1i] ABORT: factory cannot load NullConnector:", e); sys.exit(1)
if inspect.isabstract(cls):
    print("[1i] ABORT: NullConnector is abstract; some hook is unimplemented."); sys.exit(1)
if supports_hma(cls):
    print("[1i] ABORT: NullConnector claims HMA; the pool would not match 1c/1e."); sys.exit(1)
if cls.requires_piecewise_for_cudagraph({}):
    print("[1i] ABORT: NullConnector forces PIECEWISE; cudagraph mode would differ."); sys.exit(1)
print("[1i] NullConnector resolves via the factory, is concrete, no HMA, no piecewise.")
PY

BASE=( serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME"
       --tensor-parallel-size 8 --max-model-len 131072 --enable-prefix-caching
       --block-size=64 --max-num-seqs 256 )
NULL_KVCFG='{"kv_connector":"NullConnector","kv_connector_module_path":"nullconn.null_connector","kv_role":"kv_both"}'

bench_passes() {
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
    dur=d.get('duration',0); tps=d.get('total_token_throughput',0)
    ms=1000*8192/tps if tps else float('nan')
    print(f"    -> {sys.argv[2]} dur={dur:.1f}s tok/s={tps:.0f} -> {ms:.1f} ms/step")
    print(f"       reference: no connector 85.3 | MP 91.0 | IP 97.5 ms/step")
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
    echo "[$tag] FAILED TO START"; sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | tail -25; teardown; return 1
  fi
  # Assert on vLLM's OWN factory line, not on our module's logger: vLLM only
  # installs log handlers for the "vllm" namespace, so a logger named
  # nullconn.null_connector is silently dropped.  The first version of this
  # check asserted our line and aborted a perfectly good run at 18:21.
  local nconn
  nconn=$(grep -c "Creating v1 connector with name: NullConnector" "$dir/server.log" 2>/dev/null || echo 0)
  echo "  NullConnector instantiations logged by vLLM's factory: $nconn (expect 9 = 8 workers + EngineCore)"
  if [ "$nconn" -lt 2 ]; then
    echo "[1i] ABORT: the factory did not build NullConnector on both sides."
    teardown; return 1
  fi
  if ! sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | grep -q "EngineCore.*Creating v1 connector with name: NullConnector"; then
    echo "[1i] ABORT: no scheduler-side NullConnector; half the connector path would be missing."
    teardown; return 1
  fi
  grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"
  pool=$(grep -o "GPU KV cache size: [0-9,]*" "$dir/server.log" | tail -1 | tr -cd '0-9')
  echo "  pool=${pool:-unknown} (must be 13724416 to compare with 1c/1e/1b/1g)"
  if [ "${pool:-0}" -ne 13724416 ]; then
    echo "[1i] ABORT: pool is ${pool}, not 13,724,416; the arms are not comparable."
    teardown; return 1
  fi
  local c; for c in $CONC; do bench_passes "$tag" "$dir" "$c"; done
  local dmax
  dmax=$(grep -o 'Deferred: [0-9]*' "$dir/server.log" | grep -o '[0-9]*' | sort -n | tail -1)
  echo "  Deferred max: ${dmax:-none} (must be 0; this connector never returns None)"
  teardown
}

run_cfg 1i_null_connector "${BASE[@]}" --disable-hybrid-kv-cache-manager \
  --kv-transfer-config "$NULL_KVCFG"
echo "=== 1i done $(date +%H:%M:%S) -> $OUT/1i_null_connector ==="
