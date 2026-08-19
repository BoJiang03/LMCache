# SPDX-License-Identifier: Apache-2.0
"""Baseline output generator for the multimodal acceptance suite.

Runs a plain vLLM engine (no LMCache, no prefix caching) over the requests
given in the input JSON and writes {request_key: output_text} to the output
JSON. Requests are executed one at a time to keep batching identical to the
test-side sequential execution.

Usage: python baseline_runner.py <input.json> <output.json>
"""

# Standard
import json
import sys


def main(in_path: str, out_path: str) -> None:
    """Generate baseline outputs for every request in ``in_path``."""
    # Third Party
    from vllm import LLM, SamplingParams

    spec = json.loads(open(in_path).read())
    llm = LLM(
        model=spec["model"],
        max_model_len=spec["max_model_len"],
        gpu_memory_utilization=spec["gpu_memory_utilization"],
        enforce_eager=True,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 2},
    )
    results: dict[str, str] = {}
    for request in spec["requests"]:
        outputs = llm.chat(
            request["messages"],
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=request["max_tokens"], seed=0
            ),
            use_tqdm=False,
        )
        results[request["key"]] = outputs[0].outputs[0].text
    with open(out_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
