# SPDX-License-Identifier: Apache-2.0
"""Isolated-engine scenarios for the multimodal acceptance suite.

These scenarios need an engine configured differently from the shared
session harness (a tiny scheduler token budget, or a tiny LMCache capacity),
so each runs in its own subprocess:

    python isolated_cases.py <scenario> <model_key> <out_json>

The process writes a JSON report {"scenario", "model", "failures", "metrics"}
and exits nonzero if any check failed. ``test_isolated_paths.py`` wraps this
in pytest. Correctness uses the same oracle as the main suite: a plain-vLLM
baseline computed in a subprocess under the SAME engine config (semantic
probe only as nondeterminism rescue). A bare probe is not enough — small
models misname colors behind long pad prefixes even without LMCache, which
would misattribute model weakness to the cache.
"""

# Standard
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import traceback
import time
import urllib.error
import urllib.request

# First Party (test-local; none of these import lmcache at module level)
from catalog import (
    MMRequest,
    catalog,
    color_request,
    eviction_requests,
    long_prefix_color_request,
    preemption_requests,
)
from harness import LMCACHE_TEST_CHUNK_SIZE as CHUNK
from harness import (
    MMHarness,
    MPHarness,
    compute_baselines,
    vllm_preemption_total,
)
from specs import MODEL_SPECS, ModelSpec

# Pin THIS repo's lmcache package (same rationale as conftest.py). This
# script runs as a standalone subprocess, so pytest's sys.path pinning does
# not reach it: sys.path[0] is this directory, and the next `lmcache` on the
# path is whatever editable install the shared venv happens to carry --
# possibly a DIFFERENT source tree, silently certifying the wrong code.
# Must run before anything imports lmcache (all such imports happen inside
# functions, after this module-level block).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_LMCACHE_SPEC = importlib.util.find_spec("lmcache")
if _LMCACHE_SPEC is None or _LMCACHE_SPEC.origin is None:
    raise RuntimeError("lmcache is not importable from the isolated scenario")
if pathlib.Path(_LMCACHE_SPEC.origin).resolve().parents[1] != _REPO_ROOT:
    raise RuntimeError(
        f"lmcache would resolve to {_LMCACHE_SPEC.origin}, expected the tree "
        f"under {_REPO_ROOT}; the scenario would certify the wrong source tree"
    )

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

# Preemption scenario: a deliberately tiny GPU block pool (16-token blocks)
# that admits every prompt but cannot absorb the decode growth of the whole
# batch, so the scheduler MUST preempt (asserted via the vLLM preemption
# counter -- the scenario fails as vacuous if it never happens).
PREEMPTION_GPU_BLOCKS = 128
PREEMPTION_N = 6
PREEMPTION_MAX_TOKENS = 112

# MP connector scenario: ports for the cache server subprocess, derived from
# the PID so concurrent runs on one host do not collide.
MP_SERVER_L1_GB = 4
MP_SERVER_START_TIMEOUT_S = 120


def _expect(failures: list[str], condition: bool, message: str) -> None:
    """Record ``message`` in ``failures`` when ``condition`` is false."""
    if not condition:
        failures.append(message)


def _check_text(
    failures: list[str],
    harness: MMHarness,
    request: MMRequest,
    text: str,
    where: str,
) -> None:
    """Run the harness baseline/probe policy, recording instead of raising."""
    try:
        harness.check_text(request, text, where)
    except AssertionError as exc:
        failures.append(str(exc))


def _check_replay(
    failures: list[str],
    harness: MMHarness,
    request: MMRequest,
    reference_text: str,
    text: str,
    where: str,
) -> None:
    """Run the harness replay policy, recording instead of raising."""
    try:
        harness.check_replay_text(request, reference_text, text, where)
    except AssertionError as exc:
        failures.append(str(exc))


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
    engine_kwargs = {
        "gpu_memory_utilization": ISOLATED_GPU_UTILIZATION,
        "max_num_batched_tokens": CHUNKED_TOKEN_BUDGET,
        "max_num_seqs": 4,
    }
    pairs = {
        pad: (
            long_prefix_color_request(f"t09-p{pad}-A", f"t09c-{pad}", pad, 0),
            long_prefix_color_request(f"t09-p{pad}-B", f"t09c-{pad}", pad, 2),
        )
        for pad in CHUNKED_PAD_PHASES
    }
    with tempfile.TemporaryDirectory() as tmp:
        baselines = compute_baselines(
            spec,
            [r for pair in pairs.values() for r in pair],
            pathlib.Path(tmp),
            extra_engine_kwargs=engine_kwargs,
        )
    harness = MMHarness(spec, baselines=baselines, extra_engine_kwargs=engine_kwargs)
    failures: list[str] = []
    metrics: dict[str, dict] = {}
    stored_before = harness.stored_tokens_total()
    total_missed = 0
    try:
        for pad, (req_a, req_b) in pairs.items():
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
            _check_text(failures, harness, req_a, a1.text, f"T0.9 pad {pad} miss A")

            a2 = harness.run(req_a)
            _check_replay(
                failures, harness, req_a, a1.text, a2.text, f"T0.9 pad {pad} repeat A"
            )
            _expect(
                failures,
                a2.lookup_hits >= a2.lookup_tokens - 2 * CHUNK,
                f"pad {pad}: repeat hit only {a2.lookup_hits} of "
                f"{a2.lookup_tokens} tokens",
            )

            b1 = harness.run(req_b)
            _check_text(
                failures, harness, req_b, b1.text, f"T0.9 pad {pad} different B"
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
    requests = eviction_requests(EVICTION_N)
    with tempfile.TemporaryDirectory() as tmp:
        baselines = compute_baselines(
            spec,
            requests,
            pathlib.Path(tmp),
            extra_engine_kwargs={"gpu_memory_utilization": ISOLATED_GPU_UTILIZATION},
        )
    harness = MMHarness(
        spec,
        baselines=baselines,
        extra_engine_kwargs={"gpu_memory_utilization": ISOLATED_GPU_UTILIZATION},
        max_local_cpu_gb=EVICTION_CAPACITY_GB,
    )
    failures: list[str] = []
    metrics: dict[str, object] = {}
    capacity_bytes = int(EVICTION_CAPACITY_GB * 1024**3)
    try:
        pass1 = [harness.run(r) for r in requests]

        for req, res in zip(requests, pass1, strict=True):
            _check_text(failures, harness, req, res.text, f"T0.10 pass1 {req.key}")
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
            _check_replay(
                failures,
                harness,
                req,
                first.text,
                again.text,
                f"T0.10 post-eviction rerun {req.key}",
            )
    finally:
        harness.close()
    return {"failures": failures, "metrics": metrics}


def run_preemption(spec: ModelSpec) -> dict:
    """T0.11: correctness when the scheduler preempts and recomputes.

    A tiny GPU block pool plus forced-length decodes (``ignore_eos``) makes
    the running batch outgrow KV memory, so vLLM preempts at least one
    request (asserted via the ``vllm:num_preemptions`` counter; zero
    preemptions fails the scenario as vacuous) and later re-prefills it.
    The re-prefill goes through LMCache's preemption path (restored token
    ids, fresh block ids); every output must still verify against the
    config-matched plain-vLLM baseline, and afterwards every request must
    fully hit and verify again -- proving the preemption round-trip neither
    corrupted KV nor poisoned the cache.

    Args:
        spec: The model under certification.

    Returns:
        Report dict with ``failures`` (empty = pass) and ``metrics``.
    """
    engine_kwargs = {
        "gpu_memory_utilization": ISOLATED_GPU_UTILIZATION,
        "num_gpu_blocks_override": PREEMPTION_GPU_BLOCKS,
        # vLLM refuses a block pool smaller than one max-length request, so
        # the context length must shrink along with the pool.
        "max_model_len": PREEMPTION_GPU_BLOCKS * CHUNK,
        "max_num_seqs": PREEMPTION_N,
        "disable_log_stats": False,  # the preemption counter needs stats on
    }
    requests = preemption_requests(PREEMPTION_N, PREEMPTION_MAX_TOKENS)
    with tempfile.TemporaryDirectory() as tmp:
        baselines = compute_baselines(
            spec,
            requests,
            pathlib.Path(tmp),
            extra_engine_kwargs=engine_kwargs,
        )
    harness = MMHarness(spec, baselines=baselines, extra_engine_kwargs=engine_kwargs)
    failures: list[str] = []
    metrics: dict[str, object] = {}
    stored_before = harness.stored_tokens_total()
    try:
        preemptions_before = vllm_preemption_total()
        batch = harness.run_batch(requests)
        preemptions = vllm_preemption_total() - preemptions_before
        _expect(
            failures,
            preemptions > 0,
            f"no preemption occurred (counter delta {preemptions}) -- the "
            f"scenario is vacuous; lower PREEMPTION_GPU_BLOCKS or raise "
            f"PREEMPTION_MAX_TOKENS",
        )
        for req, text in zip(requests, batch.texts, strict=True):
            _check_text(failures, harness, req, text, f"T0.11 batch {req.key}")

        # The preemption round-trip must leave a usable, uncorrupted cache:
        # every request replays to a full hit and its output verifies against
        # the config-matched baseline (which runs solo/sequential -- the SAME
        # regime as this replay). Byte-equality against the batch text is
        # deliberately NOT asserted: the concurrent preempted batch is a
        # different numeric regime, and the ignore_eos garbage tail amplifies
        # kernel-level numeric differences chaotically; contamination is
        # still caught hard, because the probe would name the wrong color.
        for req in requests:
            again = harness.run(req)
            _check_text(failures, harness, req, again.text, f"T0.11 replay {req.key}")
            _expect(
                failures,
                again.lookup_hits >= again.lookup_tokens - 2 * CHUNK,
                f"{req.key}: replay hit only {again.lookup_hits} of "
                f"{again.lookup_tokens} tokens after the preemption batch",
            )

        # Under-storage guard: preemption must not silently drop stores.
        # (No upper bound here: a preempted request legitimately re-stores.)
        stored_delta = harness.stored_tokens_total() - stored_before
        missed = batch.lookup_tokens - batch.lookup_hits
        _expect(
            failures,
            stored_delta >= missed - PREEMPTION_N * CHUNK,
            f"under-storage across preemption: batch missed {missed} tokens "
            f"but only {stored_delta} were store-requested",
        )
        metrics["preemptions"] = preemptions
        metrics["batch"] = {
            "lookup_tokens": batch.lookup_tokens,
            "lookup_hits": batch.lookup_hits,
            "stored_delta": stored_delta,
        }
    finally:
        harness.close()
    return {"failures": failures, "metrics": metrics}


def _start_mp_server(zmq_port: int, http_port: int, log_path: pathlib.Path):
    """Launch the LMCache MP cache server and wait until it is healthy.

    Args:
        zmq_port: ZMQ port for the connector traffic.
        http_port: HTTP port for the observability API.
        log_path: File capturing the server's stdout/stderr.

    Returns:
        The server ``subprocess.Popen`` handle.

    Raises:
        RuntimeError: If the server does not become healthy in time.
    """
    log_file = open(log_path, "w")
    # The server must run THIS repo's lmcache too: `-m` resolves through the
    # child's own sys.path, where the venv's editable install would otherwise
    # win (see the pinning at the top of this file). PYTHONPATH entries
    # precede site-packages.
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{_REPO_ROOT}:{existing_pp}" if existing_pp else str(_REPO_ROOT)
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lmcache.v1.multiprocess.http_server",
            "--port",
            str(zmq_port),
            "--http-port",
            str(http_port),
            "--chunk-size",
            str(CHUNK),
            "--l1-size-gb",
            str(MP_SERVER_L1_GB),
            "--eviction-policy",
            "LRU",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    deadline = time.monotonic() + MP_SERVER_START_TIMEOUT_S
    url = f"http://localhost:{http_port}/healthcheck"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return proc
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    proc.terminate()
    raise RuntimeError(
        f"MP cache server failed to become healthy; log tail:\n"
        f"{log_path.read_text()[-2000:]}"
    )


def run_mp_connector(spec: ModelSpec) -> dict:
    """T3: the T0+T1 core on the multi-process connector deployment path.

    Starts a real LMCache MP cache server subprocess, drives a vLLM engine
    through ``LMCacheMPConnector`` (this repo's version), and replays the
    core acceptance set: cross-image isolation and hit equivalence (T0.1 /
    T0.3), mixed traffic (T0.5), a concurrent batch (T0.8), prefix reuse
    (T1.2), multi-image order (T2.1), partial sharing (T2.2), store
    conservation against the server's resident-object API, and the
    detector negative control. Chunk-boundary phases (T0.4) and collision
    pressure (T0.2) are keyspace properties independent of the transport
    and stay on the in-process path.

    Args:
        spec: The model under certification.

    Returns:
        Report dict with ``failures`` (empty = pass) and ``metrics``.
    """
    zmq_port = 25000 + (os.getpid() % 5000)
    http_port = zmq_port + 5000
    cat = catalog()
    used_keys = [
        "t01-A",
        "t01-B",
        "t05-text",
        "t05-A",
        "t05-B",
        "t08-A",
        "t08-B",
        "t08-text",
        "t12-A",
        "t12-A-q2",
        "t21-AB",
        "t21-BA",
        "t22-A",
        "t22-AC",
    ]
    requests = [cat[k] for k in used_keys]
    engine_kwargs = {"gpu_memory_utilization": ISOLATED_GPU_UTILIZATION}
    failures: list[str] = []
    metrics: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as tmp:
        baselines = compute_baselines(
            spec, requests, pathlib.Path(tmp), extra_engine_kwargs=engine_kwargs
        )
        server = _start_mp_server(
            zmq_port, http_port, pathlib.Path(tmp) / "mp_server.log"
        )
        harness = MPHarness(
            spec,
            baselines=baselines,
            zmq_port=zmq_port,
            http_port=http_port,
            extra_engine_kwargs=engine_kwargs,
        )
        try:
            singles: list = []

            # T0.1 / T0.3 / T1.1 / T1.3 on the t01 pair.
            a1 = harness.run(cat["t01-A"])
            singles.append(a1)
            _expect(failures, a1.lookup_hits == 0, "fresh salt hit something")
            _check_text(failures, harness, cat["t01-A"], a1.text, "T3 t01 first A")
            b1 = harness.run(cat["t01-B"])
            singles.append(b1)
            _check_text(failures, harness, cat["t01-B"], b1.text, "T3 t01 B")
            a2 = harness.run(cat["t01-A"])
            singles.append(a2)
            _check_text(failures, harness, cat["t01-A"], a2.text, "T3 t01 repeat A")
            _check_replay(
                failures, harness, cat["t01-A"], a1.text, a2.text, "T3 t01 repeat A"
            )
            _expect(
                failures,
                a2.lookup_hits >= a2.lookup_tokens - 2 * CHUNK,
                f"repeat A hit only {a2.lookup_hits}/{a2.lookup_tokens}",
            )
            _expect(
                failures,
                b1.lookup_hits <= a2.lookup_hits - IMAGE_SPAN_MARGIN,
                f"image B hit {b1.lookup_hits} tokens, too close to A's "
                f"full hit {a2.lookup_hits} -- cross-image false hit",
            )
            metrics["t01"] = {
                "prompt_tokens": a1.lookup_tokens,
                "full_hit": a2.lookup_hits,
                "b_hit": b1.lookup_hits,
            }

            # T0.5 mixed traffic.
            t1 = harness.run(cat["t05-text"])
            singles.append(t1)
            _check_text(failures, harness, cat["t05-text"], t1.text, "T3 t05 text")
            m1 = harness.run(cat["t05-A"])
            singles.append(m1)
            _check_text(failures, harness, cat["t05-A"], m1.text, "T3 t05 A")
            t2 = harness.run(cat["t05-text"])
            singles.append(t2)
            _check_replay(
                failures, harness, cat["t05-text"], t1.text, t2.text, "T3 t05 repeat"
            )
            _expect(
                failures,
                t2.lookup_hits >= t2.lookup_tokens - 2 * CHUNK,
                f"text-only repeat hit only {t2.lookup_hits}/{t2.lookup_tokens}",
            )
            m2 = harness.run(cat["t05-B"])
            singles.append(m2)
            _check_text(failures, harness, cat["t05-B"], m2.text, "T3 t05 B")

            # T1.2 prefix reuse across questions.
            q1 = harness.run(cat["t12-A"])
            singles.append(q1)
            _check_text(failures, harness, cat["t12-A"], q1.text, "T3 t12 q1")
            q2 = harness.run(cat["t12-A-q2"])
            singles.append(q2)
            _check_text(failures, harness, cat["t12-A-q2"], q2.text, "T3 t12 q2")
            _expect(
                failures,
                q2.lookup_hits >= q1.lookup_tokens - 6 * CHUNK,
                f"prefix reuse too shallow: {q2.lookup_hits} of ~{q1.lookup_tokens}",
            )
            _expect(
                failures,
                q2.lookup_hits < q2.lookup_tokens,
                "different question must not fully hit",
            )

            # T2.1 multi-image order.
            ab = harness.run(cat["t21-AB"])
            singles.append(ab)
            _check_text(failures, harness, cat["t21-AB"], ab.text, "T3 t21 AB")
            ba = harness.run(cat["t21-BA"])
            singles.append(ba)
            _check_text(failures, harness, cat["t21-BA"], ba.text, "T3 t21 BA")
            ab2 = harness.run(cat["t21-AB"])
            singles.append(ab2)
            _check_replay(
                failures, harness, cat["t21-AB"], ab.text, ab2.text, "T3 t21 repeat"
            )
            _expect(
                failures,
                ba.lookup_hits <= ab2.lookup_hits - IMAGE_SPAN_MARGIN,
                f"swapped order hit {ba.lookup_hits} vs full {ab2.lookup_hits}",
            )

            # T2.2 partial sharing.
            s1 = harness.run(cat["t22-A"])
            singles.append(s1)
            _check_text(failures, harness, cat["t22-A"], s1.text, "T3 t22 A")
            s2 = harness.run(cat["t22-AC"])
            singles.append(s2)
            _check_text(failures, harness, cat["t22-AC"], s2.text, "T3 t22 AC")
            _expect(
                failures,
                s2.lookup_hits >= IMAGE_SPAN_MARGIN + CHUNK,
                f"shared image prefix not reused: {s2.lookup_hits}",
            )
            _expect(
                failures,
                s2.lookup_hits < s2.lookup_tokens,
                "the second image is new; a full hit here is a false hit",
            )

            # Conservation on the MP path: everything the lookups missed must
            # have been submitted for storage, and the server must actually
            # hold objects.
            missed = sum(r.lookup_tokens - r.lookup_hits for r in singles)
            stored = harness.stored_tokens_total()
            _expect(
                failures,
                stored >= missed - len(singles) * CHUNK,
                f"under-storage on MP path: missed {missed} tokens but only "
                f"{stored} were store-submitted",
            )
            snapshot = harness.storage()
            _expect(
                failures,
                snapshot.num_keys > 0 and snapshot.total_bytes > 0,
                f"MP server reports no resident objects ({snapshot})",
            )
            metrics["conservation"] = {
                "missed": missed,
                "stored": stored,
                "resident_keys": snapshot.num_keys,
                "resident_bytes": snapshot.total_bytes,
            }

            # T0.8 concurrent batch, then the batched store must be hittable.
            batch_reqs = [
                cat["t08-A"],
                cat["t08-B"],
                cat["t08-A"],
                cat["t08-text"],
                cat["t08-B"],
            ]
            batch = harness.run_batch(batch_reqs)
            for i, (req, text) in enumerate(zip(batch_reqs, batch.texts, strict=True)):
                _check_text(failures, harness, req, text, f"T3 t08 entry {i}")
            after = harness.run(cat["t08-A"])
            _check_text(failures, harness, cat["t08-A"], after.text, "T3 t08 after")
            _expect(
                failures,
                after.lookup_hits >= after.lookup_tokens - 2 * CHUNK,
                f"batched store not hittable: {after.lookup_hits}/"
                f"{after.lookup_tokens}",
            )

            # Negative control: with MM identity substitution disabled the
            # cross-image tripwire MUST fire on this path too.
            blind_a = color_request("t3blind-A", "t3blind", 0)
            blind_b = color_request("t3blind-B", "t3blind", 2)
            with harness.identity_blindness():
                nc_a = harness.run(blind_a)
                _expect(
                    failures,
                    nc_a.lookup_hits == 0,
                    "negative control: fresh salt hit something",
                )
                nc_b = harness.run(blind_b)
            _expect(
                failures,
                nc_b.lookup_hits > nc_b.lookup_tokens - IMAGE_SPAN_MARGIN,
                f"negative control did not trip on the MP path: blind B hit "
                f"only {nc_b.lookup_hits} of {nc_b.lookup_tokens} tokens",
            )
        except Exception:  # noqa: BLE001 - keep the report; see server_log_tail
            # A rare KV-load-failure flake aborts the engine mid-scenario;
            # convert the crash into a reported failure so the report (and
            # the server-side log below) survives for diagnosis.
            failures.append("engine exception:\n" + traceback.format_exc())
        finally:
            harness.close()
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
        if failures:
            log_path = pathlib.Path(tmp) / "mp_server.log"
            metrics["server_log_tail"] = log_path.read_text()[-8000:]
    return {"failures": failures, "metrics": metrics}


SCENARIOS = {
    "chunked_prefill": run_chunked_prefill,
    "capacity_eviction": run_capacity_eviction,
    "preemption": run_preemption,
    "mp_connector": run_mp_connector,
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
