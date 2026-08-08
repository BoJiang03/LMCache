#!/usr/bin/env bash
# Probe: can DeepSeek-V4-Flash boot on this node, and does LMCache's chunk size
# still line up with the KV cache group geometry?
#
# Uses --load-format dummy, so there is no 148GB weight download/load: the
# DeepGEMM crash we are chasing happens in process_weights_after_loading, which
# runs for dummy weights too. One cycle is minutes, not the ~40 the real test
# needs.
#
# Every other flag is copied verbatim from run-dsv4-flash-tp.sh so the KV cache
# group geometry the probe reports is the one the real test would see.
#
# Usage:
#   ./probe-dsv4-sm120.sh                      # install SM120 DeepGEMM, fp8_ds_mla
#   INSTALL_DEEPGEMM=0 ./probe-dsv4-sm120.sh   # baseline: reproduce the crash
#   KV_CACHE_DTYPE=fp8 ./probe-dsv4-sm120.sh   # vllm#43477's SM120 recipe
#
# Run it inside an environment that already has vllm + lmcache importable
# (i.e. what .buildkite/k3_harness/setup-env.sh leaves behind).

set -uo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_ds_mla}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
# Capped (the real test uses "auto" -> 1M) purely to keep KV allocation quick.
# tokens_per_block is a per-group property, so this does not affect geometry.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Off the CI defaults (6555/8000) so this can run beside a real job.
LMCACHE_PORT="${LMCACHE_PORT:-6655}"
VLLM_PORT="${VLLM_PORT:-8100}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
INSTALL_DEEPGEMM="${INSTALL_DEEPGEMM:-1}"
# vllm-project/DeepGEMM@nv_dev+situ -- the branch carrying the SM120 kernels.
DEEPGEMM_REF="${DEEPGEMM_REF:-5f33a18079e96d26d5869c9759657eb6150f31b1}"

OUT_DIR="${OUT_DIR:-/tmp/dsv4_probe_$$}"
mkdir -p "$OUT_DIR"
VLLM_LOG="$OUT_DIR/vllm.log"
LMCACHE_LOG="$OUT_DIR/lmcache.log"
DEEPGEMM_LOG="$OUT_DIR/deepgemm_build.log"

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
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | tr -d ' ')
echo "compute capability: $COMPUTE_CAP"
echo "logs: $OUT_DIR"
echo ""

python3 -c "import vllm, lmcache" 2>/dev/null || {
    echo "FATAL: vllm and lmcache must be importable in this environment." >&2
    exit 2
}
echo "vllm:    $(python3 -c 'import vllm; print(vllm.__version__)')"
echo ""

# ── 1. Swap in a DeepGEMM that has SM120 kernels ──────────────────────
# vLLM's _import_deep_gemm() prefers a site-packages `deep_gemm` over the copy
# vendored in the wheel, so installing one here overrides it without rebuilding
# vLLM. The vendored copy is built from the revision pinned in
# cmake/external_projects/deepgemm.cmake, which carries no SM120 code.
if [ "$INSTALL_DEEPGEMM" = "1" ]; then
    if python3 -c "import deep_gemm" 2>/dev/null; then
        echo "=== DeepGEMM already installed in site-packages, skipping build ==="
    else
        echo "=== Building DeepGEMM @ ${DEEPGEMM_REF:0:12} (log: $DEEPGEMM_LOG) ==="
        DG_DIR="$OUT_DIR/deepgemm"
        {
            git clone --recursive --shallow-submodules \
                https://github.com/vllm-project/DeepGEMM.git "$DG_DIR" &&
            cd "$DG_DIR" &&
            git checkout "$DEEPGEMM_REF" &&
            python3 setup.py bdist_wheel &&
            { command -v uv >/dev/null && uv pip install dist/*.whl \
                || python3 -m pip install dist/*.whl; }
        } > "$DEEPGEMM_LOG" 2>&1
        if [ $? -ne 0 ]; then
            echo "FAIL: DeepGEMM build failed. Tail of $DEEPGEMM_LOG:" >&2
            tail -30 "$DEEPGEMM_LOG" >&2
            exit 3
        fi
        cd - >/dev/null
        echo "DeepGEMM installed."
    fi
    python3 -c "import deep_gemm; print('deep_gemm from:', deep_gemm.__file__)"
else
    echo "=== INSTALL_DEEPGEMM=0: using vLLM's vendored DeepGEMM (baseline) ==="
fi
echo ""

# ── 2. LMCache MP server (L1 only, same as the real test) ─────────────
echo "=== Launching LMCache MP server (port $LMCACHE_PORT, chunk $CHUNK_SIZE) ==="
lmcache server \
    --host localhost \
    --port "$LMCACHE_PORT" \
    --chunk-size "$CHUNK_SIZE" \
    --l1-size-gb 40 \
    --eviction-policy LRU \
    --max-workers 4 \
    > "$LMCACHE_LOG" 2>&1 &
LMCACHE_PID=$!
sleep 10

# ── 3. vLLM with dummy weights ────────────────────────────────────────
echo "=== Launching vLLM TP=$TENSOR_PARALLEL_SIZE (dummy weights, kv-cache-dtype=$KV_CACHE_DTYPE) ==="
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
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --port "$VLLM_PORT" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 120}}" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting up to ${READY_TIMEOUT}s..."
deadline=$(( $(date +%s) + READY_TIMEOUT ))
verdict=""
while [ "$(date +%s)" -lt "$deadline" ]; do
    if grep -q "Application startup complete" "$VLLM_LOG" 2>/dev/null; then
        verdict="BOOTED"; break
    fi
    if grep -q "Unknown SF transformation" "$VLLM_LOG" 2>/dev/null; then
        verdict="SF_TRANSFORM"; break
    fi
    if grep -q "Unsupported architecture" "$VLLM_LOG" 2>/dev/null; then
        verdict="UNSUPPORTED_ARCH"; break
    fi
    if grep -q "must be a multiple of engine group" "$VLLM_LOG" 2>/dev/null; then
        verdict="CHUNK_MISMATCH"; break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        verdict="DIED"; break
    fi
    sleep 5
done
[ -z "$verdict" ] && verdict="TIMEOUT"

# ── 4. Report ─────────────────────────────────────────────────────────
echo ""
echo "================ VERDICT: $verdict ================"
echo ""
echo "--- which DeepGEMM did vLLM load? ---"
grep -E "deep_gemm not found in site-packages|Imported deep_gemm|DeepGEMM (PDL|E8M0)" \
    "$VLLM_LOG" | head -5
echo "  (the 'not found in site-packages' line means the SM120 override did NOT take)"
echo ""
echo "--- kernel / backend selection ---"
grep -E "Selected .*Kernel|Mxfp4 MoE backend|indexer|attention backend|Using .*MLA" \
    "$VLLM_LOG" | head -10
echo ""

case "$verdict" in
  BOOTED)
    echo "Model is up AND LMCache accepted chunk size $CHUNK_SIZE."
    echo "=> Both unknowns resolved: SM120 DeepGEMM works, geometry unchanged."
    ;;
  CHUNK_MISMATCH)
    echo "Model loaded, but the KV group geometry differs from the H200 path:"
    grep -E "must be a multiple of engine group" "$VLLM_LOG" | head -5
    echo "=> Raise CHUNK_SIZE to the LCM of every tokens_per_block reported above,"
    echo "   then re-run to surface the next group."
    ;;
  SF_TRANSFORM|UNSUPPORTED_ARCH)
    echo "Still hitting DeepGEMM's arch gate:"
    grep -E "Unknown SF transformation|Unsupported architecture" "$VLLM_LOG" | head -3
    echo "=> The SM120 DeepGEMM either did not install or does not cover this call."
    ;;
  *)
    echo "--- last 40 lines of $VLLM_LOG ---"
    tail -40 "$VLLM_LOG"
    ;;
esac
echo ""
echo "Full logs kept in $OUT_DIR"
