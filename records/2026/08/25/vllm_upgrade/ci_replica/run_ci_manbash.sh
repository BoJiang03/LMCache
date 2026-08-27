#!/usr/bin/env bash
# Verbatim replication of upstream buildkite vllm-correctness.sh Phase 2+3
# (the only hit-path gate in GPU CI), instrumented with LMCache load provenance.
# Usage: run_ci_manbash.sh <venv_python_dir> <lmcache_tree> <gpu> <workdir>
set -uo pipefail
VENVBIN="$1"; LMTREE="$2"; GPU="$3"; WORK="$4"
mkdir -p "$WORK"; cd "$WORK"
MODEL="Qwen/Qwen2.5-14B-Instruct"
VLLM_LOG="$WORK/vllm.log"

cat <<CFG > cpu.yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 50
CFG

PORT=8377
HF_HUB_CACHE=/raid/data/hub \
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$LMTREE" \
LMCACHE_CONFIG_FILE=cpu.yaml \
VLLM_SERVER_DEV_MODE=1 \
VLLM_BATCH_INVARIANT=1 \
"$VENVBIN/vllm" serve "$MODEL" \
    --port "$PORT" \
    --trust-remote-code \
    --enforce-eager \
    --attention-backend FLASH_ATTN \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null; sleep 5; kill -9 $VLLM_PID 2>/dev/null' EXIT

READY=false
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "Qwen2.5-14B"; then READY=true; break; fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[ERROR] server died"; tail -30 "$VLLM_LOG"; exit 1; fi
    sleep 5
done
[ "$READY" = true ] || { echo "[ERROR] server not ready in 600s"; tail -30 "$VLLM_LOG"; exit 1; }
echo "[INFO] server ready"

# CI's exact context construction
CONTEXT="$(man bash | col -b | tr -s '[:space:]' ' ' | awk '{for(i=1;i<=NF;i++){printf "%s ",$i; if(++c==5000) exit}}')"
HALF_CONTEXT="$(man bash | col -b | tr -s '[:space:]' ' ' | awk '{for(i=1;i<=NF;i++){printf "%s ",$i; if(++c==2500) exit}}')"

send_completion() {
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg model "$MODEL" --arg content "$1" '{model: $model, temperature: 0, max_tokens: 100, messages: [{role:"user",content:$content}]}')" |
        jq -r '.choices[0].message.content'
}

mark() { echo "===MARK $1===" ; wc -l < "$VLLM_LOG" > "$WORK/mark_$1.lineno"; }

mark step1_before
OUT1="$(send_completion "${CONTEXT}")"
mark step2_before
curl -s -X POST "http://localhost:${PORT}/reset_prefix_cache" >/dev/null
mark step3_before
send_completion "${HALF_CONTEXT}" >/dev/null
mark step4_before
OUT2="$(send_completion "${CONTEXT}")"
mark step4_after

echo "$OUT1" > "$WORK/out1.txt"; echo "$OUT2" > "$WORK/out2.txt"
if [[ "$OUT1" == "$OUT2" ]]; then echo "[CI-GATE] PASS (outputs identical)"; else echo "[CI-GATE] FAIL (outputs differ)"; fi

# Provenance: what did LMCache actually load during step 4?
S4A=$(cat "$WORK/mark_step4_before.lineno"); S4B=$(cat "$WORK/mark_step4_after.lineno")
echo "--- LMCache activity during STEP 4 (full-context re-ask) ---"
sed -n "$((S4A+1)),${S4B}p" "$VLLM_LOG" | grep -E "hit tokens|Retrieved|Stored" | sed 's/\x1b\[[0-9;]*m//g'
echo "--- done ---"
