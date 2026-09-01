#!/usr/bin/env bash
set -euo pipefail
export SMOKE_GPU=1,2,3,4
export SMOKE_TP=4
export SMOKE_HORIZON=2.5
export SMOKE_PYTHON=/home/bo/venvs/vllm-lazy/bin/python
export SMOKE_VLLM=/home/bo/venvs/vllm-lazy/bin/vllm
export SMOKE_REPO=/home/bo/LMCache-worktrees/lazy_offloading_policy_pr
export SMOKE_LOGDIR=/home/bo/.claude/jobs/a9b3c1ce/tmp/gsm8k_pass5/logs
export HF_HUB_CACHE=/raid/data/hub
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/raid/data/hub/pr4499_agentic/pydev/usr/include/python3.12:/raid/data/hub/pr4499_agentic/pydev/usr/include
export REPETITIONS=1 QUESTIONS=120 CONCURRENCY=4 L1_GB=68
exec /home/bo/LMCache-worktrees/lazy_offload_repro/repro/pr4499/run_gsm8k.sh
