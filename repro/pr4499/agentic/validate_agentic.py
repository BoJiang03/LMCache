#!/usr/bin/env python
"""Check every agentic run against the guards that make it interpretable.

A run is reported invalid when any of these fails:

- the request count differs from `sessions x steps`;
- any step failed or the engine/server exited non-zero;
- the engine's own prompt-token count for a step differs from the cohort's
  tokenizer count, which would mean the replay did not send the prompts the
  cohort selection reasoned about;
- a lazy run's counter ledger does not close (`admitted` must equal
  `pending` plus every terminal outcome);
- the engine log carries a traceback;
- vLLM preempted, which would make the latency numbers a story about the
  scheduler recovering rather than about the cache.

Usage:
    python validate_agentic.py <cohort.json> <results_dir_or_files...>
"""

import json
from pathlib import Path
import sys

#: Ledger fields that consume an admission. `deduplicated`,
#: `rejected_unhashed` and `rejected_prefix_broken` are *not* here: those
#: operations are rejected at admission, so they never counted as admitted.
#: `throttled_drains` counts drains rather than operations. This is the same
#: equation `driver.check_ledger` asserts.
_SINKS = (
    "emitted",
    "dropped_evicted",
    "rejected_short_prefix",
    "dropped_on_request_drop",
    "dropped_failed_store",
    "dropped_id_reuse",
    "pending",
)


def check(document: dict, cohort: dict[str, list[int]]) -> list[str]:
    """Return the guard failures of one run, empty when it is valid."""
    failures = []
    expected = document["expected_requests"]
    if document["requests"] != expected:
        failures.append(f"{document['requests']} requests, expected {expected}")
    if document["failed"]:
        failures.append(f"{len(document['failed'])} failed steps")
    for name in ("rc_engine", "rc_server"):
        code = document[name]
        if code not in (0, -2, -15):
            failures.append(f"{name}={code}")
    mismatches = 0
    for record in document["records"]:
        if not record.get("ok"):
            continue
        if cohort[record["instance_id"]][record["step"]] != record["prompt_tokens"]:
            mismatches += 1
    if mismatches:
        failures.append(f"{mismatches} prompts differ from the cohort's token count")
    ledger = document["ledger"]
    if ledger:
        total = sum(ledger.get(name, 0) for name in _SINKS)
        if total != ledger.get("admitted", -1):
            failures.append(f"ledger does not close: admitted {ledger.get('admitted')} vs {total}")
    elif document["config"] == "lazy":
        failures.append("lazy run wrote no counter ledger")
    if document["tracebacks"]:
        failures.append(f"{len(document['tracebacks'])} tracebacks")
    if document["delta"]["preemptions"]:
        failures.append(f"{document['delta']['preemptions']:.0f} preemptions")
    return failures


def main() -> int:
    """Validate every run named on the command line.

    Returns:
        Process exit code; 1 if any run failed a guard.
    """
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cohort = {
        session["instance_id"]: session["step_prompt_tokens"]
        for session in json.loads(Path(sys.argv[1]).read_text())["cohort"]
    }
    paths: list[Path] = []
    for argument in sys.argv[2:]:
        path = Path(argument)
        paths.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    invalid = 0
    for path in paths:
        document = json.loads(path.read_text())
        failures = check(document, cohort)
        invalid += bool(failures)
        status = "; ".join(failures) if failures else "ok"
        print(f"{path.name}: {status}")
    print(f"{len(paths) - invalid}/{len(paths)} runs valid")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
