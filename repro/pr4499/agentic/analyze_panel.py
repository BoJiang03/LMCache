#!/usr/bin/env python
"""Compare agentic replay runs that share a request stream.

Every variant in a panel replays the same cohort through the same slot
schedule, so a request is identified by its slot, trajectory and step in
all of them, and the comparison can be per request rather than per
distribution. That
matters here: the effect being measured is a few tens of milliseconds on a
population whose own spread is hundreds, so unpaired percentiles cannot
resolve it.

Reported against a reference variant (``off``, the engine with no KV
connector, unless told otherwise):

- per-variant coverage, external hit rate, tail latencies and the policy
  ledger, including the per-step cost sensors;
- paired mean deltas overall, by prompt length, and -- for decode -- by
  position in the run, which is where a cost that grows with the pending
  backlog shows up and a cost that does not, does not.

Usage:
    analyze_panel.py <run.json> [<run.json> ...] [--reference NAME]
"""

# Standard
import argparse
import json
from pathlib import Path
import statistics

#: Prompt-length bucket edges in tokens, chosen so each holds a comparable
#: share of the untruncated cohort.
_LENGTH_EDGES = (4000, 8000, 16000, 24000)

#: Slices of a run, by request completion order.
_OCTILES = 8


def load(path: Path) -> dict:
    """Read one run document and index its records by request identity.

    Args:
        path: Path to an ``ag_<tag>.json`` produced by ``agentic.py``.

    Returns:
        The run document with an added ``by_request`` mapping from
        ``(slot, trajectory, step)`` to the record.

    Raises:
        ValueError: If two records share an identity, which would make the
            pairing ambiguous.
    """
    run = json.loads(path.read_text())
    by_request: dict[tuple[int, int, int], dict] = {}
    for record in run["records"]:
        key = (record["session"], record["trajectory"], record["step"])
        if key in by_request:
            raise ValueError(f"{path.name}: duplicate request identity {key}")
        by_request[key] = record
    run["by_request"] = by_request
    return run


def decode_ms(record: dict) -> float:
    """Time one request spent generating after its first token.

    Reported whole rather than per token because every request in the
    cohort asks for the same number of output tokens: the whole quantity is
    what the request's end-to-end latency actually carries, and dividing by
    a constant only makes the numbers smaller.

    Returns:
        End-to-end time minus time to first token, in milliseconds.
    """
    return record["e2e_ms"] - record["ttft_ms"]


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile of a non-empty sample."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[index]


def variant_name(run: dict) -> str:
    """Short label for a run: its config plus any tag suffix."""
    tag = run["tag"]
    rep = run["rep"]
    marker = f"_{rep}"
    suffix = tag.split(marker, 1)[1].lstrip("_") if marker in tag else ""
    return f"{run['config']}-{suffix}" if suffix else run["config"]


def summarise(run: dict) -> dict[str, float]:
    """Cache effectiveness and latency tails of one run."""
    delta = run["delta"]
    records = run["records"]
    ttfts = [r["ttft_ms"] for r in records]
    e2es = [r["e2e_ms"] for r in records]
    decodes = [decode_ms(r) for r in records]
    queries = delta["apc_queries"] or 1.0
    ext_queries = delta["ext_queries"] or 1.0
    return {
        "coverage": (delta["apc_hits"] + delta["ext_hits"]) / queries,
        "ext_hit": delta["ext_hits"] / ext_queries,
        "cycles": run["cycles"],
        "preemptions": delta["preemptions"],
        "ttft_p50": percentile(ttfts, 0.50),
        "ttft_p90": percentile(ttfts, 0.90),
        "e2e_p50": percentile(e2es, 0.50),
        "e2e_p90": percentile(e2es, 0.90),
        "e2e_p99": percentile(e2es, 0.99),
        "decode_mean": statistics.fmean(decodes),
    }


def paired_deltas(run: dict, reference: dict) -> dict[str, float]:
    """Mean per-request latency difference against the reference run."""
    shared = set(run["by_request"]) & set(reference["by_request"])
    ttft, decode, e2e = [], [], []
    for key in shared:
        mine, theirs = run["by_request"][key], reference["by_request"][key]
        ttft.append(mine["ttft_ms"] - theirs["ttft_ms"])
        decode.append(decode_ms(mine) - decode_ms(theirs))
        e2e.append(mine["e2e_ms"] - theirs["e2e_ms"])
    return {
        "paired": len(shared),
        "ttft": statistics.fmean(ttft),
        "decode": statistics.fmean(decode),
        "e2e": statistics.fmean(e2e),
    }


def by_length(run: dict, reference: dict) -> list[tuple[str, int, float, float]]:
    """Paired TTFT and E2E deltas per prompt-length bucket.

    Returns:
        One (label, count, mean TTFT delta, mean E2E delta) per bucket.
    """
    edges = (0,) + _LENGTH_EDGES + (10**9,)
    rows = []
    for low, high in zip(edges, edges[1:], strict=False):
        ttft, e2e = [], []
        for key, mine in run["by_request"].items():
            theirs = reference["by_request"].get(key)
            if theirs is None or not low <= mine["prompt_tokens"] < high:
                continue
            ttft.append(mine["ttft_ms"] - theirs["ttft_ms"])
            e2e.append(mine["e2e_ms"] - theirs["e2e_ms"])
        if not ttft:
            continue
        label = f"{low // 1000}-{'inf' if high > 10**8 else high // 1000}k"
        rows.append((label, len(ttft), statistics.fmean(ttft), statistics.fmean(e2e)))
    return rows


def decode_octiles(run: dict) -> list[float]:
    """Mean decode latency per eighth of the run, in completion order.

    A cost that scales with the pending backlog rises across these; a cost
    that is per-request does not.
    """
    ordered = sorted(run["records"], key=lambda r: r["finished_s"])
    size = len(ordered) // _OCTILES
    return [
        statistics.fmean([decode_ms(r) for r in ordered[i * size : (i + 1) * size]])
        for i in range(_OCTILES)
    ]


def main() -> int:
    """Print the panel tables.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--reference", default="off")
    args = parser.parse_args()

    runs: dict[str, dict] = {}
    for path in args.runs:
        run = load(path)
        name = variant_name(run)
        # Two runs of the same variant from different panels are a drift
        # control, not a mistake: keep both, told apart by repetition label.
        if name in runs:
            previous = runs.pop(name)
            runs[f"{name}/{previous['rep']}"] = previous
            name = f"{name}/{run['rep']}"
        runs[name] = run
    reference = runs.get(args.reference)

    print(
        f"{'variant':<12} {'cover':>6} {'exthit':>7} {'cycles':>7} {'ttftp50':>8} "
        f"{'ttftp90':>8} {'e2ep50':>7} {'e2ep90':>7} {'e2ep99':>7} {'decode':>7} "
        f"{'preempt':>7}"
    )
    for name, run in runs.items():
        s = summarise(run)
        print(
            f"{name:<12} {s['coverage']:>6.3f} {s['ext_hit']:>7.3f} "
            f"{s['cycles']:>7.0f} {s['ttft_p50']:>8.0f} {s['ttft_p90']:>8.0f} "
            f"{s['e2e_p50']:>7.0f} {s['e2e_p90']:>7.0f} {s['e2e_p99']:>7.0f} "
            f"{s['decode_mean']:>7.1f} {s['preemptions']:>7.0f}"
        )

    if reference is not None:
        print(f"\npaired means vs {args.reference} (ms/request)")
        print(f"{'variant':<12} {'n':>5} {'ttft':>8} {'decode':>8} {'e2e':>8}")
        for name, run in runs.items():
            if name == args.reference:
                continue
            d = paired_deltas(run, reference)
            print(
                f"{name:<12} {d['paired']:>5} {d['ttft']:>8.1f} "
                f"{d['decode']:>8.1f} {d['e2e']:>8.1f}"
            )

        print(f"\npaired TTFT / E2E by prompt length vs {args.reference} (ms)")
        for name, run in runs.items():
            if name == args.reference:
                continue
            cells = " ".join(
                f"{label}:{ttft:+.0f}/{e2e:+.0f}({count})"
                for label, count, ttft, e2e in by_length(run, reference)
            )
            print(f"  {name:<12} {cells}")

    print("\nmean decode by run octile (ms)")
    for name, run in runs.items():
        print(f"  {name:<12} " + " ".join(f"{v:5.0f}" for v in decode_octiles(run)))

    print("\npolicy ledger")
    for name, run in runs.items():
        ledger = run.get("ledger") or {}
        if not ledger:
            continue
        steps = ledger.get("drain_steps", 0)
        cost = ""
        if steps:
            cost = (
                f" | drains={steps} "
                f"read/step={ledger['free_queue_blocks_read'] / steps:.1f} "
                f"validated_blocks/step={ledger['blocks_validated'] / steps:.1f} "
                f"validated_reqs/step={ledger['requests_validated'] / steps:.2f}"
            )
        print(
            f"  {name:<12} admitted={ledger['admitted']} "
            f"emitted={ledger['emitted']} evicted={ledger['dropped_evicted']} "
            f"throttled={ledger['throttled_drains']} "
            f"pending={ledger['pending']}{cost}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
