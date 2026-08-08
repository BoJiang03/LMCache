#!/usr/bin/env bash
# Probe: can DeepSeek-V4-Flash run at TP=2 on 96GB SM120 cards, and with how
# much KV headroom?
#
# Context: a reviewer asked whether the CI test can drop from TP=4 to TP=2 to
# halve its GPU footprint. The weights are 148.7 GiB (sum of safetensors on HF).
# Two 96GB cards give ~191 GiB, so the weights alone need
# gpu_memory_utilization >= ~0.78 before any KV cache exists. The question is
# whether a workable utilization exists that both fits and leaves enough KV.
#
# Stage A (this script) answers that with --load-format dummy: no 148.7GB
# download, and allocation is identical to real weights, so both the fit and the
# reported "GPU KV cache size" are accurate. One utilization takes minutes.
#
# Stage B (the real test at TP=2) is only worth running if Stage A finds a
# utilization with comfortable headroom -- see the note printed at the end.
#
# Usage:
#   bash probe-dsv4-tp2.sh                      # sweep 0.86 0.90 0.93
#   UTILS="0.90" bash probe-dsv4-tp2.sh         # single utilization
#   CUDA_VISIBLE_DEVICES=0,1 bash probe-dsv4-tp2.sh
#
# Run inside the environment that already booted DSV4 successfully (i.e. with
# the SM120 deep_gemm installed in site-packages).

set -uo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
UTILS="${UTILS:-0.86 0.90 0.93}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_ds_mla}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# The CI test's prompt is ~8k tokens; vLLM must hold its KV during prefill.
# Treat 4x that as "comfortable" -- below it, TP=2 is too tight to ship.
KV_TOKENS_COMFORTABLE="${KV_TOKENS_COMFORTABLE:-32768}"
LMCACHE_PORT="${LMCACHE_PORT:-6755}"
VLLM_PORT="${VLLM_PORT:-8200}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"

OUT_DIR="${OUT_DIR:-/tmp/dsv4_tp2_probe_$$}"
mkdir -p "$OUT_DIR"

LMCACHE_PID=""
VLLM_PID=""
cleanup() {
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null
    [ -n "$LMCACHE_PID" ] && kill "$LMCACHE_PID" 2>/dev/null
    sleep 3
    [ -n "$VLLM_PID" ] && kill -9 "$VLLM_PID" 2>/dev/null
    [ -n "$LMCACHE_PID" ] && kill -9 "$LMCACHE_PID" 2>/dev/null
    return 0
}
trap cleanup EXIT

echo "=== Node ==="
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
echo "visible devices: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "logs: $OUT_DIR"
echo ""

python3 -c "import vllm, lmcache" 2>/dev/null || {
    echo "FATAL: vllm and lmcache must be importable here." >&2; exit 2; }

if python3 -c "import deep_gemm" 2>/dev/null; then
    echo "deep_gemm (site-packages): $(python3 -c 'import deep_gemm; print(deep_gemm.__file__)')"
else
    echo "WARNING: no site-packages deep_gemm. On SM120 this will abort in"
    echo "         DeepGEMM's SF layout transform before reaching the KV report."
fi
echo ""

# One utilization: boot with dummy weights, report fit + KV cache size.
# Returns 0 if it booted, 1 otherwise. Echoes the KV token count on stdout via
# the RESULT_* globals.
run_one() {
    local util="$1"
    local vllm_log="$OUT_DIR/vllm_util${util}.log"
    local lmcache_log="$OUT_DIR/lmcache_util${util}.log"

    RESULT_VERDICT=""
    RESULT_KV_TOKENS=""
    RESULT_DETAIL=""

    lmcache server --host localhost --port "$LMCACHE_PORT" \
        --chunk-size "$CHUNK_SIZE" --l1-size-gb 40 \
        --eviction-policy LRU --max-workers 4 \
        > "$lmcache_log" 2>&1 &
    LMCACHE_PID=$!
    sleep 8

    VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
        --load-format dummy \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --enable-expert-parallel \
        --kv-cache-dtype "$KV_CACHE_DTYPE" \
        --tokenizer-mode deepseek_v4 \
        --trust-remote-code \
        --enforce-eager \
        --enable-prefix-caching \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$util" \
        --port "$VLLM_PORT" \
        --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 120}}" \
        > "$vllm_log" 2>&1 &
    VLLM_PID=$!

    local deadline=$(( $(date +%s) + READY_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if grep -q "Application startup complete" "$vllm_log" 2>/dev/null; then
            RESULT_VERDICT="BOOTED"; break
        fi
        # Distinguish "weights don't fit" from every other death.
        if grep -qiE "No available memory for the cache blocks" "$vllm_log" 2>/dev/null; then
            RESULT_VERDICT="NO_KV_ROOM"; break
        fi
        if grep -qiE "CUDA out of memory|torch.OutOfMemoryError" "$vllm_log" 2>/dev/null; then
            RESULT_VERDICT="OOM"; break
        fi
        if grep -qiE "Free memory on device .* is less than|less than desired GPU memory" \
                "$vllm_log" 2>/dev/null; then
            RESULT_VERDICT="WEIGHTS_DONT_FIT"; break
        fi
        if grep -q "Unknown SF transformation" "$vllm_log" 2>/dev/null; then
            RESULT_VERDICT="SF_TRANSFORM"; break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            RESULT_VERDICT="DIED"; break
        fi
        sleep 5
    done
    [ -z "$RESULT_VERDICT" ] && RESULT_VERDICT="TIMEOUT"

    # vLLM logs e.g. "GPU KV cache size: 123,456 tokens"
    RESULT_KV_TOKENS=$(grep -oiE "GPU KV cache size: *[0-9,]+" "$vllm_log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9,]+$" | tr -d ',')
    RESULT_DETAIL=$(grep -iE "GPU KV cache size|Available KV cache memory|model weights take|memory profiling|maximum concurrency" \
        "$vllm_log" 2>/dev/null | tail -6)

    cleanup
    VLLM_PID=""; LMCACHE_PID=""
    sleep 20   # let VRAM actually drain before the next utilization
    [ "$RESULT_VERDICT" = "BOOTED" ]
}

declare -a SUMMARY=()
for util in $UTILS; do
    echo "──────────────────────────────────────────────"
    echo "=== TP=$TENSOR_PARALLEL_SIZE, gpu_memory_utilization=$util ==="
    run_one "$util"
    echo "verdict: $RESULT_VERDICT"
    [ -n "$RESULT_KV_TOKENS" ] && echo "GPU KV cache: $RESULT_KV_TOKENS tokens"
    [ -n "$RESULT_DETAIL" ] && { echo "--- memory lines ---"; echo "$RESULT_DETAIL"; }
    SUMMARY+=("$util|$RESULT_VERDICT|${RESULT_KV_TOKENS:-n/a}")
    echo ""
done

echo "=============================================="
echo "SUMMARY  (TP=$TENSOR_PARALLEL_SIZE, ${MODEL##*/})"
echo "=============================================="
printf '%-8s %-18s %s\n' "util" "verdict" "GPU KV tokens"
best_util=""; best_kv=0
for row in "${SUMMARY[@]}"; do
    IFS='|' read -r u v k <<< "$row"
    printf '%-8s %-18s %s\n' "$u" "$v" "$k"
    if [ "$v" = "BOOTED" ] && [[ "$k" =~ ^[0-9]+$ ]] && [ "$k" -gt "$best_kv" ]; then
        best_kv="$k"; best_util="$u"
    fi
done
echo ""

if [ -z "$best_util" ]; then
    echo "VERDICT: TP=2 does NOT boot at any utilization tried."
    echo "=> The test has to stay at TP=4. Report the verdicts above."
elif [ "$best_kv" -lt "$KV_TOKENS_COMFORTABLE" ]; then
    echo "VERDICT: TP=2 boots (best util=$best_util, $best_kv KV tokens) but the"
    echo "         headroom is under the ${KV_TOKENS_COMFORTABLE}-token comfort bar."
    echo "=> Technically feasible, too tight to ship as CI. Report both numbers."
else
    echo "VERDICT: TP=2 is viable -- util=$best_util gives $best_kv KV tokens."
    echo "=> Worth running Stage B: the REAL test at TP=2, which needs the actual"
    echo "   148.7 GiB weights (check the HF cache before starting a cold pull):"
    echo ""
    echo "     TENSOR_PARALLEL_SIZE=2 GPU_MEMORY_UTILIZATION=$best_util \\"
    echo "       bash .buildkite/k3_tests/multiprocess/scripts/run-dsv4-flash-tp.sh"
    echo ""
    echo "   That asserts the two greedy outputs are byte-identical, which is"
    echo "   what actually decides whether TP=2 can ship."
fi
echo ""
echo "Full logs in $OUT_DIR"
