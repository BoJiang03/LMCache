#!/usr/bin/env python3
"""Measure the eager APC-backfill effect over repeated hardware trials.

The experiment creates an APC-only prefix by clearing LMCache after an initial
request while leaving vLLM alive. Replaying the prompt should rebuild L1 from
GPU KV. Distinct prompts then displace it from GPU, and a third request reveals
whether the rebuilt lower-tier copy is retrieved or the prefix is recomputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import driver


def run(repetitions: int, output: Path) -> dict[str, object]:
    name = "APC_BACKFILL"
    output.parent.mkdir(parents=True, exist_ok=True)
    driver.LOGDIR.mkdir(parents=True, exist_ok=True)
    server = driver.start_server(name)
    vllm = driver.start_vllm_under(
        server,
        name,
        {},  # Eager MP connector: lazy offload disabled.
        [
            "--gpu-memory-utilization",
            "0.5",
            "--max-num-seqs",
            "4",
            "--max-model-len",
            "3072",
            "--max-num-batched-tokens",
            "512",
            "--num-gpu-blocks-override",
            "448",
        ],
    )
    try:
        rebuilt: list[int] = []
        latencies: list[float] = []
        correct: list[bool] = []
        for repetition in range(repetitions):
            prompt = driver.long_prompt(f"apc-target-{repetition}", 100)
            first = driver.complete(
                prompt, max_tokens=16, request_id=f"first-{repetition}"
            )
            time.sleep(3)
            driver.cache_clear()

            second = driver.complete(
                prompt, max_tokens=16, request_id=f"second-{repetition}"
            )
            time.sleep(3)
            rebuilt.append(driver.cache_object_count())

            for index in range(4):
                driver.complete(
                    driver.long_prompt(f"displace-{repetition}-{index}", 100),
                    max_tokens=16,
                    request_id=f"displace-{repetition}-{index}",
                )
            time.sleep(2)

            started = time.perf_counter()
            third = driver.complete(
                prompt, max_tokens=16, request_id=f"third-{repetition}"
            )
            latencies.append(time.perf_counter() - started)
            correct.append(first == second == third)
            time.sleep(2)
            print(
                f"[apc] rep={repetition} rebuilt={rebuilt[-1]} "
                f"third_s={latencies[-1]:.3f} correct={correct[-1]}"
            )

        result: dict[str, object] = {
            "code_sha": subprocess.check_output(
                ["git", "-C", driver.REPO, "rev-parse", "HEAD"], text=True
            ).strip(),
            "rebuilt_objects": rebuilt,
            "third_request_seconds": latencies,
            "third_request_median_seconds": statistics.median(latencies),
            "retrieved_token_ranges": driver.grep_retrieved(name),
            "correct": correct,
            "warnings": driver.grep_warnings(name),
            "tracebacks": driver.grep_tracebacks(name),
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("RESULT " + json.dumps(result, sort_keys=True))
        return result
    finally:
        driver.teardown([vllm, server])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repetitions, args.output)
    return 0 if all(result["correct"]) and not result["tracebacks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
