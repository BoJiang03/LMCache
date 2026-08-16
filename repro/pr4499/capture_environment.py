#!/usr/bin/env python3
"""Print the software, source, and GPU identity attached to a benchmark."""

import json
import os
import platform
import subprocess

import torch
import vllm


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


result = {
    "git_sha": command("git", "rev-parse", "HEAD"),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "vllm": vllm.__version__,
    "gpu": command(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "-i",
        os.environ.get("SMOKE_GPU", "0"),
        "--format=csv,noheader",
    ).splitlines(),
}
print(json.dumps(result, indent=2))
