#!/usr/bin/env python
"""Replay whole SWE-agent trajectories and sweep the L1 budget.

The first agentic sweep capped every session at 12 steps and kept only
trajectories whose final prompt landed in an 8K--22K window. That cap put
the median continuation prompt at 5635 tokens -- below the ~6000-token
point where retrieving a prefix from L1 starts to beat recomputing it -- so
half the requests were in a regime where no cache policy can win, and the
coverage the policy bought did not show up in latency.

This runner removes the cap. Every trajectory in the pool is replayed as
recorded: no step limit, no token window, and a serving context (40960,
Qwen3-8B's native maximum) that holds the longest one. The prompt
distribution is then the dataset's own -- p50 11K, p75 19K, p90 25K, 72%
above the crossover.

Pressure is swept with the L1 budget rather than the cohort size, because
the pool is only 74 distinct trajectories: holding the workload fixed and
moving the budget keeps every point comparing the same requests, and the
budget is what decides whether eager's write churn destroys reuse.

Variants: `eager`, `lazy` at the production default, and `lazy-d4`, the
drain budget the attribution run identified as the fix for the per-step
free-queue read.

Environment:
    AGENTIC_FULL_SLOTS     concurrent slots (default 14)
    AGENTIC_FULL_BUDGETS   comma-separated L1 GiB budgets (default 20,40,10)
    AGENTIC_FULL_VARIANTS  comma-separated variant names (default all)
    AGENTIC_FULL_REP       repetition label (default `full`)
    plus every variable agentic.py reads.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import driver  # noqa: E402

HERE = Path(__file__).resolve().parent
SLOTS = os.environ.get("AGENTIC_FULL_SLOTS", "14")
BUDGETS = [
    int(value)
    for value in os.environ.get("AGENTIC_FULL_BUDGETS", "20,40,10").split(",")
    if value.strip()
]
REP = os.environ.get("AGENTIC_FULL_REP", "full")

#: (name, config, extra connector settings) per variant.
VARIANTS = {
    "eager": ("eager", {}),
    "lazy": ("lazy", {}),
    "lazy-d4": ("lazy", {"lmcache.mp.lazy_offload_max_drain_per_step": 4}),
}
WANTED = [
    name
    for name in os.environ.get("AGENTIC_FULL_VARIANTS", "eager,lazy-d4,lazy").split(",")
    if name.strip()
]


def gpu_memory() -> dict[int, int]:
    """Used MiB per GPU this run owns."""
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    owned = {int(value) for value in driver.GPU.split(",")}
    used = {}
    for line in raw.strip().splitlines():
        index, memory = (part.strip() for part in line.split(","))
        if int(index) in owned:
            used[int(index)] = int(memory)
    return used


def wait_gpus_free(timeout: float = 300.0) -> None:
    """Block until every owned GPU is idle, so a run starts from a clean slate.

    Raises:
        RuntimeError: if a GPU is still busy when the timeout expires.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        used = gpu_memory()
        if all(value < 1000 for value in used.values()):
            print(f"[full] GPUs ready: {used}", flush=True)
            return
        time.sleep(10)
    raise RuntimeError(f"target GPUs did not become free: {gpu_memory()}")


def main() -> int:
    """Run every (budget, variant) point in a fresh process.

    Returns:
        Process exit code; 1 if any point failed.
    """
    unknown = [name for name in WANTED if name not in VARIANTS]
    if unknown:
        print(f"[full] unknown variants {unknown}", file=sys.stderr)
        return 2
    print(
        f"[full] slots={SLOTS} budgets={BUDGETS} variants={WANTED} "
        f"rep={REP} gpus={driver.GPU}",
        flush=True,
    )
    failures = 0
    for budget in BUDGETS:
        for name in WANTED:
            config, extra = VARIANTS[name]
            suffix = "" if name == config else f"_{name.split('-', 1)[1]}"
            env = dict(
                os.environ,
                AGENTIC_EXTRA_CONFIG=json.dumps(extra),
                AGENTIC_TAG_SUFFIX=suffix,
                AGENTIC_L1_GB=str(budget),
            )
            wait_gpus_free()
            print(f"[full] START {name} at L1={budget}GiB", flush=True)
            started = time.time()
            completed = subprocess.run(
                [sys.executable, str(HERE / "agentic.py"), config, REP, SLOTS],
                cwd=str(HERE),
                env=env,
            )
            print(
                f"[full] DONE {name} L1={budget} rc={completed.returncode} "
                f"in {time.time() - started:.0f}s",
                flush=True,
            )
            failures += completed.returncode != 0
    print("[full] ALL_DONE", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
