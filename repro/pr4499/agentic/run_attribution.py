#!/usr/bin/env python
"""Attribute the eviction-aware policy's per-step cost in agentic replay.

The sweep measures what the policy is worth. This script measures what it
charges, at a cohort size small enough that **no** store is ever due: the
GPU pool holds every session, so the pending queue only grows and the drain
never fires. Whatever separates the variants below is decision-loop cost,
not transfer cost.

Variants, in the order they run:

- `off`      -- no KV connector at all: the engine's own floor.
- `eager`    -- the connector, storing immediately, never buffering.
- `lazy`     -- the policy at its production defaults.
- `lazy-d4`  -- the same, with `max_drain_per_step` at 4 instead of 64.
- `lazy-d256`-- the same, with `max_drain_per_step` at 256.

The last two are the causal test. `collect_due` reads the free queue to a
depth of `danger_depth + max_drain_per_step x largest pending operation`, so
if that read is what costs, decode time moves with `max_drain_per_step`
while everything else is held fixed.

Each variant writes its own `ag_AG_<config>_..._<rep><suffix>.json` under
SMOKE_LOGDIR; `attribution_table.py` renders them.

Environment: AGENTIC_ATTR_SESSIONS (default 8), AGENTIC_ATTR_REP (default
`attr`), plus every variable agentic.py reads.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SESSIONS = os.environ.get("AGENTIC_ATTR_SESSIONS", "8")
REP = os.environ.get("AGENTIC_ATTR_REP", "attr")

#: (name, config, extra connector settings) per variant.
VARIANTS = (
    ("off", "off", {}),
    ("eager", "eager", {}),
    ("lazy", "lazy", {}),
    ("lazy-d4", "lazy", {"lmcache.mp.lazy_offload_max_drain_per_step": 4}),
    ("lazy-d256", "lazy", {"lmcache.mp.lazy_offload_max_drain_per_step": 256}),
)


def main() -> int:
    """Run every variant in a fresh process so its config is read once.

    Returns:
        Process exit code; 1 if any variant failed.
    """
    failures = 0
    for name, config, extra in VARIANTS:
        suffix = "" if name == config else f"_{name.split('-', 1)[1]}"
        env = dict(
            os.environ,
            AGENTIC_EXTRA_CONFIG=json.dumps(extra),
            AGENTIC_TAG_SUFFIX=suffix,
        )
        print(f"[attr] START {name}", flush=True)
        started = time.time()
        completed = subprocess.run(
            [sys.executable, str(HERE / "agentic.py"), config, REP, SESSIONS],
            cwd=str(HERE),
            env=env,
        )
        print(
            f"[attr] DONE {name} rc={completed.returncode} "
            f"in {time.time() - started:.0f}s",
            flush=True,
        )
        failures += completed.returncode != 0
    print("[attr] ALL_DONE", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
