# GSM8K gate 2 env: pr4499 harness measuring the PR worktree's extracted code.
# Differs from gate 1 only in SMOKE_REPO and SMOKE_LOGDIR. The PR worktree
# root carries an untracked sitecustomize.py that strips the venv's editable
# lmcache finder (driver.py overwrites PYTHONPATH with SMOKE_REPO, so the
# guard must live inside the repo root to be importable at engine startup).
export SMOKE_GPU=1,2,3,4
export SMOKE_TP=4
export SMOKE_HORIZON=2.5
export SMOKE_PYTHON=/home/bo/venvs/vllm-lazy/bin/python
export SMOKE_VLLM=/home/bo/venvs/vllm-lazy/bin/vllm
export SMOKE_REPO=/home/bo/LMCache-worktrees/lazy_offloading_policy_pr
export SMOKE_LOGDIR=/tmp/claude-1016/-home-bo-LMCache-worktrees-lazy-offloading/50d4856f-54c7-41f7-9ec8-745ca2242d0f/scratchpad/gsm8k_pr/logs
export HF_HUB_CACHE=/raid/data/hub
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/raid/data/hub/pr4499_agentic/pydev/usr/include/python3.12:/raid/data/hub/pr4499_agentic/pydev/usr/include
