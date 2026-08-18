#!/usr/bin/env python
"""Sweep the agentic cohort size across the L1 budget, eager versus lazy.

Every point is a fresh MP server and engine, so no run inherits another's
cache. Policy order is reversed in the second repetition: if a difference is
an artifact of which policy ran first on a warm machine, the two repetitions
disagree.

Environment:
    AGENTIC_SWEEP_SIZES    comma-separated cohort sizes (default 8,16,24,32)
    AGENTIC_SWEEP_REP      repetition tag (default 0)
    AGENTIC_SWEEP_REVERSE  1 to run lazy before eager
    AGENTIC_RESULTS        directory for the per-point JSON
    plus every AGENTIC_* / SMOKE_* variable agentic.py reads.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agentic  # noqa: E402
import driver  # noqa: E402

SIZES = tuple(
    int(value) for value in os.environ.get("AGENTIC_SWEEP_SIZES", "8,16,24,32").split(",")
)
MODES = (
    ("lazy", "eager") if os.environ.get("AGENTIC_SWEEP_REVERSE") == "1" else ("eager", "lazy")
)
REP = os.environ.get("AGENTIC_SWEEP_REP", "0")
RESULTS = Path(os.environ.get("AGENTIC_RESULTS", str(Path(__file__).parent / "results")))
GPU_IDS = tuple(int(value) for value in driver.GPU.split(","))


def gpu_memory() -> dict[int, int]:
    """Used MiB per GPU under test."""
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    values = {}
    for line in raw.splitlines():
        index, used = line.split(",")
        values[int(index.strip())] = int(used.strip())
    return {index: values[index] for index in GPU_IDS}


def wait_gpus_free(timeout: float = 300.0) -> None:
    """Block until every GPU under test is idle.

    Raises:
        RuntimeError: if they are still busy when the timeout expires.
    """
    started = time.time()
    while time.time() - started < timeout:
        values = gpu_memory()
        if all(value < 1000 for value in values.values()):
            print(f"[sweep] GPUs ready: {values}", flush=True)
            return
        time.sleep(2)
    raise RuntimeError(f"target GPUs did not become free: {gpu_memory()}")


def run_point(sessions: int, mode: str) -> None:
    """Run one (cohort size, policy) point unless its result already exists."""
    out = RESULTS / f"AG_{mode}_s{sessions}_{REP}.json"
    if out.exists():
        print(f"[sweep] SKIP existing {out.name}", flush=True)
        return
    wait_gpus_free()
    print(f"[sweep] START {out.name}", flush=True)
    started = time.time()
    result = agentic.run(mode, rep=REP, sessions=sessions)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(
        f"[sweep] DONE {out.name} in {time.time() - started:.0f}s "
        f"failed={len(result['failed'])}",
        flush=True,
    )


def main() -> int:
    """Run every size and policy in order."""
    print(f"[sweep] agentic sizes={SIZES} modes={MODES} rep={REP} gpus={GPU_IDS}", flush=True)
    for sessions in SIZES:
        for mode in MODES:
            run_point(sessions, mode)
    print("[sweep] ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
