# AgentX smoke run: Qwen3-Coder-30B-A3B (plain GQA, 96 KiB/token, 262k ctx)
export HF_HUB_CACHE=/raid/data/hub
export HF_HOME=/home/bo/.cache/huggingface
export REPO=/home/bo/LMCache-worktrees/lazy_offloading
export PY=/home/bo/venvs/vllm-lazy/bin/python
export VLLM=/home/bo/venvs/vllm-lazy/bin/vllm
export AIPERF=/home/bo/venvs/aiperf/bin/aiperf

export GPUS=2
export TP=1
export MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
export MAX_MODEL_LEN=131072
export KV_BYTES_PER_TOKEN=98304          # 2*4*128*48*2
export POOL_GIB=24
export POOL_BLOCKS=$(( POOL_GIB * (1<<30) / KV_BYTES_PER_TOKEN / 16 ))
export L1_GB=200
export HORIZON=2.5

export MP_PORT=26575
export HTTP_PORT=28185
export VLLM_PORT=28190
export SD=/tmp/claude-1016/-home-bo-LMCache-worktrees-lazy-offloading/7445f449-aa3b-46be-b351-5a22de3af76a/scratchpad/smoke
export LOGDIR=$SD/logs
export HF_DATASETS_CACHE=/raid/data/hub/datasets_cache
# Python.h for triton's runtime launcher compile (no python3.12-dev on this host)
export CPATH=/raid/data/hub/pr4499_agentic/pydev/usr/include/python3.12:/raid/data/hub/pr4499_agentic/pydev/usr/include
