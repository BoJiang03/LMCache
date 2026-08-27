#!/bin/bash
# setup_venv.sh <version>  -- create /home/bo/venvs/vllm-bisect-<version> with vllm==<version> + lmcache deps
set -euo pipefail
V="$1"
VENV="/home/bo/venvs/vllm-bisect-$V"
rm -rf "$VENV"; UV_PYTHON_PREFERENCE=only-managed uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" "vllm==$V" -r /home/bo/LMCache-worktrees/multi_modal/requirements/common.txt
"$VENV/bin/python" -c "import vllm,torch;print('READY vllm',vllm.__version__,'torch',torch.__version__)"
