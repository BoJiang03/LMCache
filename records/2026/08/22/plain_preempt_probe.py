# SPDX-License-Identifier: Apache-2.0
"""Control: does a mamba-align hybrid survive block-pool pressure WITHOUT LMCache?

Same engine shape as the preemption scenario (tiny num_gpu_blocks_override,
mamba_cache_mode=align, one-block step budget, 6 concurrent ignore_eos
decodes) but with no KV connector at all. If this crashes the same way the
connector run does, the defect is upstream and LMCache is not implicated.

usage: python plain_preempt_probe.py <blocks> <budget> <out_json> [hf_id block gpu_util]
"""

import json
import os
import pathlib
import sys

blocks, budget, out_json = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
HF_ID = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen3.5-2B"
BLOCK = int(sys.argv[5]) if len(sys.argv) > 5 else 544
GPU_UTIL = float(sys.argv[6]) if len(sys.argv) > 6 else 0.35

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("PYTHONHASHSEED", "0")
venv_include = pathlib.Path(sys.prefix) / "include"
if (venv_include / "Python.h").exists():
    os.environ["CPATH"] = f"{venv_include}:{os.environ.get('CPATH', '')}"

E2E = pathlib.Path("/home/bo/LMCache-worktrees/multi_modal/tests/e2e_mm")
sys.path.insert(0, str(E2E))

os.environ["LMCACHE_MM_E2E_PRE_PAD_WORDS"] = str(BLOCK * 2)
os.environ["LMCACHE_MM_E2E_POST_PAD_WORDS"] = str(BLOCK * 4)
os.environ["LMCACHE_MM_E2E_MID_PAD_WORDS"] = str(BLOCK * 2)

from catalog import preemption_requests  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

requests = preemption_requests(6, 112)
llm = LLM(
    model=HF_ID,
    max_model_len=BLOCK * 8,
    gpu_memory_utilization=GPU_UTIL,
    enforce_eager=True,
    limit_mm_per_prompt={"image": 2, "video": 1},
    num_gpu_blocks_override=blocks,
    mamba_cache_mode="align",
    enable_prefix_caching=True,
    max_num_batched_tokens=budget,
    max_num_seqs=6,
    disable_log_stats=False,
)
result: dict[str, object] = {"model": HF_ID, "block": BLOCK, "blocks": blocks, "budget": budget}
try:
    outs = llm.chat(
        [r.messages() for r in requests],
        SamplingParams(temperature=0.0, max_tokens=112, ignore_eos=True),
    )
    result["texts"] = [o.outputs[0].text[:60] for o in outs]
    result["prompt_tokens"] = [len(o.prompt_token_ids) for o in outs]
    from harness import vllm_preemption_total

    result["preemptions"] = vllm_preemption_total()
    result["ok"] = True
except Exception as exc:  # noqa: BLE001 - the crash IS the measurement
    import traceback

    result["ok"] = False
    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()[-3000:]

pathlib.Path(out_json).write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, indent=2))
