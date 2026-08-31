# GSM8K gate env: pr4499 harness measuring THIS worktree's policy code.
export SMOKE_GPU=1,2,3,4
export SMOKE_TP=4
export SMOKE_HORIZON=2.5
export SMOKE_PYTHON=/home/bo/venvs/vllm-lazy/bin/python
export SMOKE_VLLM=/home/bo/venvs/vllm-lazy/bin/vllm
export SMOKE_REPO=/home/bo/LMCache-worktrees/lazy_offloading
export SMOKE_LOGDIR=/tmp/claude-1016/-home-bo-LMCache-worktrees-lazy-offloading/aa12d55f-c087-43f5-b7de-9e19b1dcd21f/scratchpad/gsm8k/logs
export HF_HUB_CACHE=/raid/data/hub
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/raid/data/hub/pr4499_agentic/pydev/usr/include/python3.12:/raid/data/hub/pr4499_agentic/pydev/usr/include
