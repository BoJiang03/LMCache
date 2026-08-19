#!/usr/bin/env python
"""Census of vLLM preemptions in a replay run, with surrounding activity.

A preemption means one scheduler step could not allocate blocks, and under
this workload it happens against sampled pool usage nowhere near full --
the 10-second usage sampler cannot see a sub-second admission burst. What
it coincides with is readable from the logs: the policy draining large
whole-prefix ops (finished requests' KV that only the lazy policy keeps
alive while ops wait) and a burst of large admissions in the same seconds.

For every preemption warning in the engine log this prints the victim, the
nearest sampled usage lines on both sides, the nearest policy ledger
(pending depth), and every store, retrieval or prefetch completion the
server log carries in a +/- window around the event.

Usage:
    preemption_census.py <run_vllm.log> [--server-log <run_server.log>]
        [--window 6]
"""

# Standard
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import re

_PREEMPTED = re.compile(
    r"\[([0-9-]+ [0-9:,]+)\] LMCache WARNING:.*<preempted> by preempted "
    r"requests: \{([^}]*)\}"
)
_USAGE = re.compile(
    r"INFO ([0-9-]+ [0-9:]+) \[loggers\.py.*Running: (\d+) reqs, "
    r"Waiting: (\d+) reqs, GPU KV cache usage: ([0-9.]+)%"
)
_LEDGER = re.compile(
    r"\[([0-9-]+ [0-9:,]+)\] LMCache INFO:.*Lazy offload counters:.*"
    r"pending=(\d+)"
)
_ACTIVITY = re.compile(
    r"\[([0-9-]+ [0-9:,]+)\] LMCache INFO:.*?"
    r"((?:Stored|Retrieved) \d+ tokens"
    r"|Prefetch request completed \(L1\+L2\): \d+/\d+ retained keys)"
)


def stamp(text: str, year: int) -> datetime:
    """Parse a log timestamp; engine INFO lines carry no year."""
    text = text.replace(",", ".")
    if text.count("-") == 1:  # MM-DD HH:MM:SS
        return datetime.strptime(f"{year}-{text}", "%Y-%m-%d %H:%M:%S")
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vllm_log", type=Path)
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--window", type=float, default=6.0,
                        help="seconds of context on each side of an event")
    args = parser.parse_args()

    events: list[tuple[datetime, str]] = []
    usage: list[tuple[datetime, str]] = []
    ledgers: list[tuple[datetime, int]] = []
    year = datetime.now().year
    for line in args.vllm_log.open(errors="replace"):
        matched = _PREEMPTED.search(line)
        if matched:
            events.append((stamp(matched.group(1), year), matched.group(2)))
            continue
        matched = _USAGE.search(line)
        if matched:
            usage.append((
                stamp(matched.group(1), year),
                f"run={matched.group(2)} wait={matched.group(3)} "
                f"kv={matched.group(4)}%",
            ))
            continue
        matched = _LEDGER.search(line)
        if matched:
            ledgers.append((stamp(matched.group(1), year),
                            int(matched.group(2))))

    activity: list[tuple[datetime, str]] = []
    if args.server_log:
        for line in args.server_log.open(errors="replace"):
            matched = _ACTIVITY.search(line)
            if matched:
                activity.append((stamp(matched.group(1), year),
                                 matched.group(2)))

    print(f"{len(events)} preemption(s) in {args.vllm_log.name}")
    span = timedelta(seconds=args.window)
    for when, victims in events:
        print(f"\n== {when} victims: {victims}")
        before = [entry for entry in usage if entry[0] <= when]
        after = [entry for entry in usage if entry[0] > when]
        if before:
            print(f"  usage {before[-1][0].time()} {before[-1][1]}")
        if after:
            print(f"  usage {after[0][0].time()} {after[0][1]}")
        pending = [entry for entry in ledgers if entry[0] <= when]
        if pending:
            print(f"  ledger {pending[-1][0].time()} "
                  f"pending={pending[-1][1]}")
        window = [entry for entry in activity
                  if when - span <= entry[0] <= when + span]
        # One store or retrieval logs once per TP rank; collapse repeats.
        seen: dict[str, int] = {}
        for _, text in window:
            seen[text] = seen.get(text, 0) + 1
        for text, count in seen.items():
            print(f"  {text} x{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
