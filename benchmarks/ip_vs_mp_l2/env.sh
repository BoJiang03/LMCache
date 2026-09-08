# SPDX-License-Identifier: Apache-2.0
# Environment for the IP-vs-MP L2 benchmark.  Every value can be overridden
# from the caller's environment; nothing here is specific to one machine.
#
#   MODEL        model path or hub id served by vLLM
#   SERVED_NAME  --served-model-name, also the tokenizer the bench uses
#   L2_DIR       directory the L2 tier writes into.  Needs ~8.5 GB per request
#                per point; see README "Disk".
#   TP           tensor-parallel size
#   L1_GB        DRAM budget given to BOTH arms (see README "Equalising DRAM")
#
# The venv is picked up from PATH; set VLLM/PY if vllm and python are not the
# ones you want.

export MODEL="${MODEL:-openai/gpt-oss-120b}"
export SERVED_NAME="${SERVED_NAME:-gpt-oss-120b}"
export L2_DIR="${L2_DIR:-/tmp/lmcache_l2}"

export TP="${TP:-8}"
export L1_GB="${L1_GB:-1200}"
export ISL="${ISL:-120000}"

export PORT="${PORT:-8765}"
export MP_PORT="${MP_PORT:-5765}"
export HTTP_PORT="${HTTP_PORT:-8766}"

export VLLM="${VLLM:-$(command -v vllm)}"
export LMCACHE_BIN="${LMCACHE_BIN:-$(command -v lmcache)}"
export PY="${PY:-$(command -v python3)}"

# Block device backing $L2_DIR, read from /sys/block/$L2_DEV/stat to account
# the bytes each measured round actually moves.  Set it to your device (md0,
# nvme0n1, ...); the accounting is skipped if it does not exist.
export L2_DEV="${L2_DEV:-}"

# /reset_prefix_cache is only mounted in dev mode
# (vllm/entrypoints/serve/cache/api_router.py: `if not envs.VLLM_SERVER_DEV_MODE: return`).
# It adds debug endpoints only -- scheduling and memory behaviour are unchanged,
# it is in the compile cache key's ignored_factors, and both arms get it.
export VLLM_SERVER_DEV_MODE=1

# One .data file per key per rank, and c=600 keeps ~8400 chunks in flight.
ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null || true
