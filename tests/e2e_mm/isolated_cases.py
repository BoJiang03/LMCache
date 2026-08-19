# SPDX-License-Identifier: Apache-2.0
"""Isolated-engine scenarios for the multimodal acceptance suite.

These scenarios need an engine configured differently from the shared
session harness (a tiny scheduler token budget, or a tiny LMCache capacity),
so each runs in its own subprocess:

    python isolated_cases.py <scenario> <model_key> <out_json>

The process writes a JSON report {"scenario", "model", "failures", "metrics"}
and exits nonzero if any check failed. ``test_isolated_paths.py`` wraps this
in pytest. Verification is semantic-probe and self-equivalence based (no
plain-vLLM baseline): chunked scheduling changes batch shapes, so exact
cross-engine token matching would be noise, while a cache-contamination
failure still flips the probe or the pass-vs-pass equality.
"""

# Standard
import json
import pathlib
import sys

# First Party (test-local)
from catalog import eviction_requests, long_prefix_color_request
from harness import LMCACHE_TEST_CHUNK_SIZE as CHUNK
from harness import MMHarness
from specs import MODEL_SPECS, ModelSpec

IMAGE_SPAN_MARGIN = 4 * CHUNK

# Scheduler token budget for the chunked-prefill scenario: small enough that
# every scenario prompt (padded prefix + image span + question) needs several
# prefill steps, so step boundaries land inside the image placeholder span.
CHUNKED_TOKEN_BUDGET = 128
# Pad lengths sweep the step boundary across different offsets of the span.
CHUNKED_PAD_PHASES = (40, 56, 72, 88)

# LMCache local CPU capacity (GB) for the eviction scenario, and the number
# of distinct images pushed through it. The images' KV must overflow the
# capacity several times over; the scenario verifies that it actually did
# and fails if the traffic never reached capacity.
EVICTION_CAPACITY_GB = 0.05
EVICTION_N = 32

# Isolated engines coexist with (at most) one session engine on the GPU, so
# they claim a smaller fraction than the spec default.
ISOLATED_GPU_UTILIZATION = 0.35


def _expect(failures: list[str], condition: bool, message: str) -> None:
    """Record ``message`` in ``failures`` when ``condition`` is false."""
    if not condition:
        failures.append(message)


def run_chunked_prefill(spec: ModelSpec) -> dict:
    """T0.9: correctness when scheduler steps split an image span.

    With ``max_num_batched_tokens`` far below the prompt length, vLLM's
    chunked prefill computes each prompt across several scheduler steps, so
    LMCache's store path receives token prefixes that END INSIDE the image
    placeholder span (the truncated-span branch of the substitution). For
    each pad phase: a fresh request must miss, its repeat must fully hit and
    reproduce the same text, and a different image behind the identical
    padded prefix must not hit into the image region.

    Args:
        spec: The model under certification.

    Returns:
        Report dict with ``failures`` (empty = pass) and ``metrics``.
    """
    harness = MMHarness(
        spec,
        baselines={},
        extra_engine_kwargs={
            "gpu_memory_utilization": ISOLATED_GPU_UTILIZATION,
            "max_num_batched_tokens": CHUNKED_TOKEN_BUDGET,
            "max_num_seqs": 4,
        },
    )
    failures: list[str] = []
    metrics: dict[str, dict] = {}
    stored_before = harness.stored_tokens_total()
    total_missed = 0
    try:
        for pad in CHUNKED_PAD_PHASES:
            salt = f"t09c-{pad}"
            req_a = long_prefix_color_request(f"t09-p{pad}-A", salt, pad, 0)
            req_b = long_prefix_color_request(f"t09-p{pad}-B", salt, pad, 2)

            a1 = harness.run(req_a)
            _expect(
                failures,
                a1.lookup_tokens > CHUNKED_TOKEN_BUDGET,
                f"pad {pad}: prompt ({a1.lookup_tokens} tokens) does not "
                f"exceed the step budget {CHUNKED_TOKEN_BUDGET}; the "
                f"scenario is not exercising chunked prefill",
            )
            _expect(
                failures,
                a1.lookup_hits == 0,
                f"pad {pad}: fresh request hit {a1.lookup_hits} tokens",
            )
            _expect(
                failures,
                harness.probe_ok(req_a, a1.text),
                f"pad {pad}: miss-path probe failed: {a1.text!r}",
            )

            a2 = harness.run(req_a)
            _expect(
                failures,
                a2.text == a1.text,
                f"pad {pad}: hit-path text {a2.text!r} != miss-path {a1.text!r}",
            )
            _expect(
                failures,
                a2.lookup_hits >= a2.lookup_tokens - 2 * CHUNK,
                f"pad {pad}: repeat hit only {a2.lookup_hits} of "
                f"{a2.lookup_tokens} tokens",
            )

            b1 = harness.run(req_b)
            _expect(
                failures,
                harness.probe_ok(req_b, b1.text),
                f"pad {pad}: different-image probe failed: {b1.text!r}",
            )
            _expect(
                failures,
                b1.lookup_hits <= a2.lookup_hits - IMAGE_SPAN_MARGIN,
                f"pad {pad}: image B hit {b1.lookup_hits} tokens, too close "
                f"to A's full hit {a2.lookup_hits} -- cross-image false hit",
            )
            total_missed += (
                (a1.lookup_tokens - a1.lookup_hits)
                + (a2.lookup_tokens - a2.lookup_hits)
                + (b1.lookup_tokens - b1.lookup_hits)
            )
            metrics[f"pad_{pad}"] = {
                "prompt_tokens": a1.lookup_tokens,
                "full_hit": a2.lookup_hits,
                "b_hit": b1.lookup_hits,
            }

        # Storage conservation under chunked prefill: split-step stores must
        # still add up to (at least) everything the lookups missed.
        stored_delta = harness.stored_tokens_total() - stored_before
        n_requests = 3 * len(CHUNKED_PAD_PHASES)
        _expect(
            failures,
            stored_delta >= total_missed - n_requests * CHUNK,
            f"under-storage across chunked prefill: missed {total_missed} "
            f"tokens but only {stored_delta} were store-requested",
        )
        metrics["stored_delta"] = {"stored": stored_delta, "missed": total_missed}
    finally:
        harness.close()
    return {"failures": failures, "metrics": metrics}


def run_capacity_eviction(spec: ModelSpec) -> dict:
    """T0.10: correctness and conservation once the cache overflows.

    Runs ``EVICTION_N`` distinct images through a cache capped at
    ``EVICTION_CAPACITY_GB``. Eviction must keep resident bytes under the
    cap, never manufacture false hits, and evicted requests must recompute
    to exactly their first-pass output.

    Args:
        spec: The model under certification.

    Returns:
        Report dict with ``failures`` (empty = pass) and ``metrics``.
    """
    harness = MMHarness(
        spec,
        baselines={},
        extra_engine_kwargs={"gpu_memory_utilization": ISOLATED_GPU_UTILIZATION},
        max_local_cpu_gb=EVICTION_CAPACITY_GB,
    )
    failures: list[str] = []
    metrics: dict[str, object] = {}
    capacity_bytes = int(EVICTION_CAPACITY_GB * 1024**3)
    try:
        requests = eviction_requests(EVICTION_N)
        pass1 = [harness.run(r) for r in requests]

        for req, res in zip(requests, pass1, strict=True):
            _expect(
                failures,
                harness.probe_ok(req, res.text),
                f"{req.key}: probe failed under eviction: {res.text!r}",
            )
        _expect(
            failures,
            pass1[0].lookup_hits == 0,
            f"fresh salt hit {pass1[0].lookup_hits} tokens",
        )
        # The shared text prefix may itself get evicted, so hits may drop
        # BELOW the steady state; they must never exceed it (a false hit).
        steady = pass1[1].lookup_hits
        for i, res in enumerate(pass1[1:], start=1):
            _expect(
                failures,
                res.lookup_hits <= steady,
                f"request {i}: hit {res.lookup_hits} tokens, above the "
                f"text-prefix steady state {steady} -- false hit under "
                f"eviction",
            )

        # Conservation under the cap: the traffic must overflow capacity
        # (else this scenario is vacuous) while resident bytes stay bounded.
        snapshot = harness.storage()
        stored_tokens = harness.stored_tokens_total()
        _expect(
            failures,
            snapshot.num_keys > 0,
            "no resident keys after the eviction traffic",
        )
        bytes_per_token = snapshot.total_bytes / max(1, snapshot.num_keys * CHUNK)
        intended_bytes = int(stored_tokens * bytes_per_token)
        _expect(
            failures,
            intended_bytes > 2 * capacity_bytes,
            f"traffic stored only ~{intended_bytes} bytes against a "
            f"{capacity_bytes}-byte cap -- eviction never exercised; raise "
            f"EVICTION_N or lower EVICTION_CAPACITY_GB",
        )
        _expect(
            failures,
            snapshot.total_bytes <= int(capacity_bytes * 1.10),
            f"resident bytes {snapshot.total_bytes} exceed the "
            f"{capacity_bytes}-byte capacity -- eviction is not bounding "
            f"the backend",
        )
        metrics["resident"] = {
            "num_keys": snapshot.num_keys,
            "total_bytes": snapshot.total_bytes,
            "capacity_bytes": capacity_bytes,
            "intended_bytes": intended_bytes,
        }

        # Evicted work must recompute, not corrupt: the earliest request is
        # long evicted, the latest may still be resident; both must equal
        # their own first-pass output exactly (sequential greedy runs).
        for index in (0, EVICTION_N - 1):
            req, first = requests[index], pass1[index]
            again = harness.run(req)
            _expect(
                failures,
                again.text == first.text,
                f"{req.key}: post-eviction rerun {again.text!r} != first "
                f"pass {first.text!r}",
            )
    finally:
        harness.close()
    return {"failures": failures, "metrics": metrics}


SCENARIOS = {
    "chunked_prefill": run_chunked_prefill,
    "capacity_eviction": run_capacity_eviction,
}


def main(argv: list[str]) -> int:
    """CLI entry point: run one scenario and write its JSON report.

    Args:
        argv: ``[scenario, model_key, out_json]``.

    Returns:
        Process exit code (0 = all checks passed).
    """
    if len(argv) != 3:
        print(
            "usage: python isolated_cases.py <scenario> <model_key> <out_json>",
            file=sys.stderr,
        )
        return 2
    scenario_name, model_key, out_json = argv
    if scenario_name not in SCENARIOS:
        print(f"unknown scenario {scenario_name!r}", file=sys.stderr)
        return 2
    spec = MODEL_SPECS[model_key]
    report = SCENARIOS[scenario_name](spec)
    report["scenario"] = scenario_name
    report["model"] = model_key
    pathlib.Path(out_json).write_text(json.dumps(report, indent=2))
    if report["failures"]:
        print(json.dumps(report["failures"], indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
