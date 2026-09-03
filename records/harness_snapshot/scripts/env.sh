# Shared env for the VAST<>LMCache reproduction
export REPRO_ROOT=/home/bo/vast_profiling_problem
export VENV=$REPRO_ROOT/.venv
export PY=$VENV/bin/python
export VLLM=$VENV/bin/vllm
export LMCACHE_SRC=/home/bo/LMCache-worktrees/vast_repro
export MODEL=/raid/rui/gpt-oss-120b
export SERVED_NAME=gpt-oss-120b
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$VENV/bin:$PATH
export PORT=8765
export MP_PORT=5765
