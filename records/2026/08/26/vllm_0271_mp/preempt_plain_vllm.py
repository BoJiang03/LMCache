"""Control: plain vLLM (no LMCache) under the preemption scenario's config.

Runs the same six requests as isolated_cases.run_preemption, in ONE batch
(the scenario's own baseline runs them one at a time, so it never applies
batch pressure and cannot answer whether vLLM alone recovers from
preemption). Reports vLLM's own preemption counter and whether the batch
finished.

Usage: preempt_plain_vllm.py <e2e_dir>   [env BLOCKS=<n>]
"""
import os
import sys
import time

e2e_dir = sys.argv[1]
sys.path.insert(0, e2e_dir)

from catalog import preemption_requests  # noqa: E402
from harness import (  # noqa: E402
    configure_environment,
    effective_max_tokens,
    mm_limits,
    spec_engine_kwargs,
    vllm_preemption_total,
)
from specs import MODEL_SPECS  # noqa: E402
import isolated_cases as IC  # noqa: E402

configure_environment()
spec = MODEL_SPECS[os.environ.get("MODEL_KEY", "qwen2-vl-2b")]
requests = preemption_requests(IC.PREEMPTION_N, IC.PREEMPTION_MAX_TOKENS)
kwargs = dict(
    model=spec.hf_id,
    max_model_len=IC.PREEMPTION_MAX_MODEL_LEN,
    gpu_memory_utilization=IC.isolated_gpu_utilization(spec),
    enforce_eager=True,
    enable_prefix_caching=False,
    limit_mm_per_prompt=mm_limits(spec),
)
kwargs.update(spec_engine_kwargs(spec))
kwargs.update(
    {
        "num_gpu_blocks_override": int(os.environ.get("BLOCKS", "128")),
        "max_model_len": IC.PREEMPTION_MAX_MODEL_LEN,
        "max_num_seqs": IC.PREEMPTION_N,
        "disable_log_stats": False,
    }
)
print(f"[control] blocks={kwargs['num_gpu_blocks_override']} reqs={len(requests)}", flush=True)

from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(**kwargs)
before = vllm_preemption_total()
start = time.time()
outputs = llm.chat(
    [r.messages() for r in requests],
    sampling_params=[
        SamplingParams(
            temperature=0.0,
            max_tokens=effective_max_tokens(spec, r),
            seed=0,
            ignore_eos=r.ignore_eos,
        )
        for r in requests
    ],
    chat_template_kwargs=dict(spec.chat_template_kwargs) or None,
    use_tqdm=False,
)
elapsed = time.time() - start
print(
    f"[control] RESULT outputs={len(outputs)} preemptions={vllm_preemption_total() - before} "
    f"elapsed={elapsed:.1f}s",
    flush=True,
)
