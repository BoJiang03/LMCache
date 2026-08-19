#!/usr/bin/env python
"""Attribute the short-prompt TTFT gap between a policy run and a reference.

The paired panel shows lazy paying a few milliseconds of TTFT on prompts
under 8k tokens. This tool separates the two costs hiding in that number:

- a per-request connector overhead that every variant with a KV connector
  pays (visible in the reference-vs-reference-free comparison and in the
  policy run's first octile, before L1 has anything to hit);
- retrieval on hits too small to pay for themselves (visible as a bimodal
  delta whose heavy mode tracks the hit fraction, absent at step 0, and
  arriving only once L1 warms up).

Three independent extractions triangulate it:

1. paired TTFT deltas by prompt bucket, split step-0 vs continuation;
2. the same deltas by run octile (the L1 warm-up ramp);
3. from the policy run's server log: retrieval latency by transfer size
   (``Retrieved N tokens in T seconds``) and per-request hit size
   (``Prefetch request completed: N/M retained keys``, where keys are
   256-token chunks counted across all TP ranks).

Usage:
    attribute_short_ttft.py <policy_run.json> <reference_run.json> \
        [--server-log <policy_server.log>] [--tp 4]
"""

# Standard
import argparse
import json
from pathlib import Path
import re
import statistics

#: Prompt-length bucket edges in tokens; the last bucket is open.
_BUCKETS = ((0, 4000), (4000, 8000), (8000, 16000), (16000, 45000))

#: Tokens per stored chunk.
_CHUNK_TOKENS = 256

_RETRIEVED = re.compile(r"Retrieved (\d+) tokens in ([\d.]+) seconds")
_PREFETCH = re.compile(
    r"Prefetch request completed \(L1\+L2\): (\d+)/(\d+) retained keys"
)


def load(path: Path) -> dict:
    """Read a run document and index its records by request identity."""
    run = json.loads(path.read_text())
    run["by_request"] = {
        (r["session"], r["trajectory"], r["step"]): r for r in run["records"]
    }
    return run


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile of a non-empty sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def delta_row(deltas: list[float]) -> str:
    """Mean, quartiles and the heavy-mode fraction of one delta sample."""
    heavy = sum(1 for d in deltas if d > 10) / len(deltas)
    return (
        f"n={len(deltas):>4} mean={statistics.fmean(deltas):>+7.1f} "
        f"p25={quantile(deltas, 0.25):>+7.1f} p50={quantile(deltas, 0.5):>+7.1f} "
        f"p75={quantile(deltas, 0.75):>+7.1f} frac>10ms={heavy:.2f}"
    )


def bucket_table(run: dict, reference: dict) -> None:
    """Paired TTFT deltas by prompt bucket, step-0 split out."""
    print("TTFT delta by prompt bucket (ms, run - reference):")
    for low, high in _BUCKETS:
        for group, keep in (("step0", lambda k: k[2] == 0),
                            ("cont", lambda k: k[2] > 0)):
            deltas = [
                run["by_request"][key]["ttft_ms"]
                - reference["by_request"][key]["ttft_ms"]
                for key in run["by_request"]
                if key in reference["by_request"]
                and keep(key)
                and low <= run["by_request"][key]["prompt_tokens"] < high
            ]
            if deltas:
                print(f"  {low // 1000:>2}k-{high // 1000}k {group:>5}: "
                      f"{delta_row(deltas)}")


def octile_ramp(run: dict, reference: dict) -> None:
    """Short-prompt TTFT deltas by completion octile: the warm-up ramp.

    A hit-driven cost is absent while L1 is cold and appears as coverage
    ramps; a fixed decision cost is flat from the first octile.
    """
    keys = sorted(
        (key for key in run["by_request"]
         if key in reference["by_request"] and key[2] > 0
         and run["by_request"][key]["prompt_tokens"] < 8000),
        key=lambda key: run["by_request"][key]["finished_s"],
    )
    print("Short-prompt (sub-8k) TTFT delta by run octile:")
    count = len(keys)
    for index in range(8):
        chunk = keys[index * count // 8:(index + 1) * count // 8]
        deltas = [run["by_request"][key]["ttft_ms"]
                  - reference["by_request"][key]["ttft_ms"] for key in chunk]
        print(f"  O{index + 1}: {delta_row(deltas)}")


def server_log_tables(log_path: Path, tp: int) -> None:
    """Retrieval latency by size, and hit sizes by prompt bucket."""
    retrievals: list[tuple[int, float]] = []
    hits: list[tuple[int, int]] = []
    for line in log_path.open(errors="replace"):
        matched = _RETRIEVED.search(line)
        if matched:
            retrievals.append(
                (int(matched.group(1)), float(matched.group(2)) * 1000.0)
            )
            continue
        matched = _PREFETCH.search(line)
        if matched:
            # Keys are 256-token chunks counted across all TP ranks, so
            # keys * 256 / tp converts to tokens.
            scale = _CHUNK_TOKENS // tp
            hits.append(
                (int(matched.group(2)) * scale, int(matched.group(1)) * scale)
            )
    print(f"Retrieval latency by transfer size ({len(retrievals)} transfers):")
    for low, high in ((0, 2000), (2000, 4000), (4000, 8000),
                      (8000, 16000), (16000, 45000)):
        sample = [ms for tokens, ms in retrievals if low <= tokens < high]
        if sample:
            print(f"  {low // 1000:>2}k-{high // 1000}k: n={len(sample):>5} "
                  f"p50={quantile(sample, 0.5):>5.1f}ms "
                  f"p90={quantile(sample, 0.9):>5.1f}ms")
    print(f"Hit size by prompt bucket ({len(hits)} prefetch completions):")
    for low, high in _BUCKETS:
        sample = [hit for prompt, hit in hits if low <= prompt < high]
        if not sample:
            continue
        small = sum(1 for hit in sample if hit <= 2048) / len(sample)
        nonzero = sorted(hit for hit in sample if hit) or [0]
        print(f"  {low // 1000:>2}k-{high // 1000}k: n={len(sample):>5} "
              f"frac<=2k={small:.2f} "
              f"median|hit>0={nonzero[len(nonzero) // 2]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="policy run JSON")
    parser.add_argument("reference", type=Path, help="reference run JSON")
    parser.add_argument("--server-log", type=Path, default=None,
                        help="the policy run's mp-server log")
    parser.add_argument("--tp", type=int, default=4,
                        help="TP rank count behind the retained-keys counts")
    args = parser.parse_args()
    run, reference = load(args.run), load(args.reference)
    bucket_table(run, reference)
    print()
    octile_ramp(run, reference)
    if args.server_log:
        print()
        server_log_tables(args.server_log, args.tp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
