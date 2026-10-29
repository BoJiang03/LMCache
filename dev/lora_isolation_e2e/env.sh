#!/usr/bin/env bash
# Shared settings; every value can be overridden from the environment.
E2E=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$E2E/../.." && pwd)
PY=${PY:-/home/bo/perf-regression-testing/.venv/bin/python}
MODEL=${MODEL:-/raid/models/Qwen2.5-1.5B-Instruct}
GPU=${GPU:-1}
OUT_ROOT=${OUT_ROOT:-$E2E/out}
ADAPTERS=${ADAPTERS:-$E2E/adapters}
CONTROLS=${CONTROLS:-$E2E/trees}
BASELINE_PORT=${BASELINE_PORT:-19100}
VLLM_PORT=${VLLM_PORT:-19200}
LMCACHE_PORT=${LMCACHE_PORT:-26557}
# The system nvcc (11.5) cannot build FlashInfer's JIT kernels.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH=$CUDA_HOME/bin:$PATH
