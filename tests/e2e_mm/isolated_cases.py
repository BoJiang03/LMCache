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
from collections.abc import Iterator
import contextlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import traceback

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
    start_mp_cache_server,
    vllm_preemption_total,
)
from isolated_routing import ALL_SCENARIOS
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

# The same cap for the MP path, which needs a different number.
#
# The in-process tier honors 0.05 GB to the byte (measured: 53673984 resident
# against a 53687091 cap). The MP server's host allocator instead expands in
# 64 MB units, so a request below one unit is silently rounded UP: asking for
# 0.05 GB yields a 64 MB pool ("Total allocated size: 61.25 MB, free 2.75
# MB"), and eviction then correctly bounds usage to 64 MB while the scenario
# compares it against 51.2 MB and calls a working backend broken.
#
# So ask for a whole number of units -- and exactly ONE unit, which is
# forced rather than tidy. At one unit both registered hybrids overflow the
# cap well past the 2x vacuity bar (Gemma 4: 252 MB intended, 3.8x, landing
# at 0.992 of the cap; Gemma 3: 184 MB, 2.7x, at 0.771) with 76 and 153
# resident objects respectively, fine enough granularity for the 10% bound
# below. At two units Gemma 3's 184 MB would be only 1.44x and the scenario
# would fail itself as vacuous.
#
# One unit is also enough for a Qwen3.5-2B, whose recurrent-state objects are
# 12 MB (measured: 5 resident keys, 60162048 bytes, 0.897 of the cap, against
# 1.6 GB of intended traffic -- 23.9x). Only the 27B-class hybrids, whose one
# object is a ~154 MB state page, need more, and they say so via
# ``ModelSpec.eviction_capacity_gb``: the exclusion here is per-model object
# size, not per hybrid family.
EVICTION_CAPACITY_GB_MP = 0.0625

# Isolated engines coexist with (at most) one session engine on the GPU, so
# they claim a smaller fraction than the spec default. A model whose weights
# alone exceed this fraction overrides it via
# ``ModelSpec.isolated_gpu_utilization`` -- see that field for why sharing
# the GPU is not actually required.
ISOLATED_GPU_UTILIZATION = 0.35


def isolated_gpu_utilization(spec: ModelSpec) -> float:
    """GPU fraction for an engine this module starts.

    Args:
        spec: The model under certification.

    Returns:
        The model's override when it declares one, else the shared default.
    """
    return spec.isolated_gpu_utilization or ISOLATED_GPU_UTILIZATION


# Preemption scenario: a deliberately tiny GPU block pool (16-token blocks)
# that admits every prompt but cannot absorb the decode growth of the whole
# batch, so the scheduler MUST preempt (asserted via the vLLM preemption
# counter -- the scenario fails as vacuous if it never happens).
PREEMPTION_GPU_BLOCKS = 128
PREEMPTION_N = 6
PREEMPTION_MAX_TOKENS = 112

# Context length for the preemption engine, decoupled from the block count.
#
# These two used to be one expression, ``PREEMPTION_GPU_BLOCKS * CHUNK``,
# which silently assumed vLLM's block size equals LMCache's chunk. That
# holds only for uniform 16-token-block models. On a hybrid the two differ
# per group, so the product stopped describing the pool: Gemma 4-E4B needs
# 0.11 GiB for one max-length request (2048 tokens x 56 KB/token) while 128
# blocks of its 256 KB give 0.03 GiB, and vLLM refuses a pool that cannot
# hold one max-length request.
#
# Kept numerically identical (128 * 16) so every already-certified model
# sees the exact same engine, while a hybrid can now raise its pool via
# ``ModelSpec.preemption_gpu_blocks`` without also stretching the context
# it has to fit.
#
# It is a fixed number rather than a per-model one because every model that
# runs this scenario fits in it: the conftest pads a hybrid prompt to span
# HYBRID_PRE_PAD_BLOCKS + HYBRID_POST_PAD_BLOCKS = 6 whole KV blocks, which
# is 96-192 tokens for the sliding-window hybrids. It does NOT fit a
# recurrent-state hybrid (3264 tokens on Qwen3.5-2B, 4704 on the 27Bs), but
# that family cannot run this scenario at all for an unrelated reason -- see
# certify._PREEMPTION_NOT_COVERED -- so a model whose padded prompt outgrows
# this constant should raise it deliberately rather than silently. The
# failure is self-explanatory if one ever does: "maximum context length is
# 2048 tokens... your prompt contains at least 2049".
PREEMPTION_MAX_MODEL_LEN = 128 * 16

# MP connector scenario: ports for the cache server subprocess, derived from
# the PID so concurrent runs on one host do not collide.
MP_SERVER_L1_GB = 4
# Hybrid models store a fat recurrent-state page per block (~13 MB on
# Qwen3.5-2B) on top of the attention KV, and their prompts are padded to
# span several blocks. Deeper hybrids override this via
# ``ModelSpec.mp_server_l1_gb`` (~205 MB per block on Qwen3.6-27B).
MP_SERVER_L1_GB_HYBRID = 30
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


@contextlib.contextmanager
def _deployment_harness(
    spec: ModelSpec,
    requests: list[MMRequest],
    extra_engine_kwargs: dict[str, object],
    metrics: dict[str, object],
    cache_capacity_gb: float = 0.0,
) -> "Iterator[MMHarness]":
    """Yield a harness on the only deployment path *spec* can actually run.

    A model with more than one KV cache group needs vLLM's hybrid KV cache
    manager, and vLLM offers that only to connectors implementing
    ``SupportsHMA``. Of ours only ``LMCacheMPConnector`` does, so for a
    hybrid the in-process path is not merely slower: vLLM logs "Turning off
    hybrid kv cache manager because --kv-transfer-config selects a KV
    connector that does not support it" and engine init then dies inside
    ``get_attn_backends_for_group`` on a layer the collapsed spec dropped.
    That is why these scenarios were listed as not covered on every hybrid
    certificate -- not because the paths were untestable, but because they
    only ever built the harness that cannot load the model.

    Hybrids therefore bring up a real MP cache server, which is also where
    their capacity lives (the server's ``l1_size_gb``) rather than in a
    harness kwarg.

    Baselines are computed here because both paths need them under the same
    engine config, and their temporary directory has to outlive the server
    log it also holds.

    Args:
        spec: The model under certification.
        requests: Requests whose plain-vLLM baselines the scenario checks
            against.
        extra_engine_kwargs: Additional/overriding vLLM ``LLM(...)`` kwargs.
        metrics: Scenario metrics; the server's log tail is recorded here on
            exit, so a failure that only shows up server-side is diagnosable
            after the temporary directory is gone.
        cache_capacity_gb: Cache capacity to impose, in GB. 0 selects the
            path's own default, which is what every scenario except
            eviction wants.

    Yields:
        A started harness. It, and the server if one was started, are torn
        down on exit.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        baselines = compute_baselines(
            spec, requests, tmpdir, extra_engine_kwargs=extra_engine_kwargs
        )

        if not spec.hybrid_block_tokens:
            # Pass the capacity only when one is imposed, so the harness's
            # own default stays the single source of truth for it.
            capacity_kwargs: dict[str, float] = {}
            if cache_capacity_gb:
                capacity_kwargs["max_local_cpu_gb"] = cache_capacity_gb
            harness: MMHarness = MMHarness(
                spec,
                baselines=baselines,
                extra_engine_kwargs=extra_engine_kwargs,
                **capacity_kwargs,
            )
            try:
                yield harness
            finally:
                harness.close()
            return

        zmq_port = 25000 + (os.getpid() % 5000)
        http_port = zmq_port + 5000
        log_path = tmpdir / "mp_server.log"
        server = start_mp_cache_server(
            zmq_port=zmq_port,
            http_port=http_port,
            # A hybrid's chunk must be vLLM's unified block size, and its
            # per-group layers need their own cache objects.
            chunk_size=spec.hybrid_block_tokens,
            log_path=log_path,
            l1_size_gb=(
                cache_capacity_gb or spec.mp_server_l1_gb or MP_SERVER_L1_GB_HYBRID
            ),
            separate_object_groups=True,
            start_timeout_s=MP_SERVER_START_TIMEOUT_S,
        )
        harness = MPHarness(
            spec,
            baselines=baselines,
            zmq_port=zmq_port,
            http_port=http_port,
            extra_engine_kwargs=extra_engine_kwargs,
        )
        try:
            yield harness
        finally:
            harness.close()
            server.process.terminate()
            try:
                server.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.process.kill()
            metrics["server_log_tail"] = log_path.read_text()[-8000:]


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
        "gpu_memory_utilization": isolated_gpu_utilization(spec),
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
    ``EVICTION_CAPACITY_GB`` (``EVICTION_CAPACITY_GB_MP`` on the MP path,
    whose allocator cannot honor a sub-unit capacity). Eviction must keep
    resident bytes under the cap, never manufacture false hits, and evicted
    requests must recompute to exactly their first-pass output.

    Args:
        spec: The model under certification.

    Returns:
        Report dict with ``failures`` (empty = pass) and ``metrics``.
    """
    requests = eviction_requests(EVICTION_N)
    failures: list[str] = []
    metrics: dict[str, object] = {}
    # Assert against the capacity actually configured, not a nominal one the
    # tier never agreed to.
    capacity_gb = spec.eviction_capacity_gb or (
        EVICTION_CAPACITY_GB_MP if spec.hybrid_block_tokens else EVICTION_CAPACITY_GB
    )
    capacity_bytes = int(capacity_gb * 1024**3)
    metrics["capacity_gb"] = capacity_gb
    with _deployment_harness(
        spec,
        requests,
        {"gpu_memory_utilization": isolated_gpu_utilization(spec)},
        metrics,
        cache_capacity_gb=capacity_gb,
    ) as harness:
        pass1 = [harness.run(r) for r in requests]

        for req, res in zip(requests, pass1, strict=True):
            _check_text(failures, harness, req, res.text, f"T0.10 pass1 {req.key}")
        _expect(
            failures,
            pass1[0].lookup_hits == 0,
            f"fresh salt hit {pass1[0].lookup_hits} tokens",
        )
        # These requests share a text prefix and differ only in image, so a
        # legitimate hit covers the leading shared region and stops at the
        # image; anything reaching the trailing blocks is another image's KV
        # (see MMHarness.image_span_margin). The bound is absolute rather
        # than relative to an earlier request in this pass: it used to be
        # `pass1[1].lookup_hits` as a "steady state", which silently assumed
        # request 0's store had landed before request 1 looked up. It has
        # not, once an object is big enough -- Qwen3.8-27B measured a steady
        # state of 0 (its 154 MB state page was still in flight for the
        # first three requests, which then hit 784-1568 and were all
        # reported as false hits) where the architecturally identical
        # Qwen3.6-27B measured 1568. A race decided the reference value, so
        # the reference could not be a measurement.
        false_hit_ceiling = pass1[1].lookup_tokens - harness.image_span_margin
        for i, res in enumerate(pass1[1:], start=1):
            _expect(
                failures,
                res.lookup_hits <= false_hit_ceiling,
                f"request {i}: hit {res.lookup_hits} of {res.lookup_tokens} "
                f"tokens, past the {false_hit_ceiling}-token shared-prefix "
                f"ceiling -- the hit reached the image span, a false hit "
                f"under eviction",
            )
        metrics["pass1_hits"] = [res.lookup_hits for res in pass1]

        # Conservation under the cap: the traffic must overflow capacity
        # (else this scenario is vacuous) while resident bytes stay bounded.
        snapshot = harness.storage()
        stored_tokens = harness.stored_tokens_total()
        _expect(
            failures,
            snapshot.num_keys > 0,
            "no resident keys after the eviction traffic",
        )
        # Per the model's own chunk, not the module default: a resident key
        # holds one chunk, so on a hybrid (chunk 32 on Gemma 4, 784 on
        # Qwen3.8) dividing by 16 would inflate bytes_per_token by the ratio
        # and make the overflow assertion below meaningless.
        bytes_per_token = snapshot.total_bytes / max(
            1, snapshot.num_keys * harness.chunk
        )
        intended_bytes = int(stored_tokens * bytes_per_token)
        _expect(
            failures,
            intended_bytes > 2 * capacity_bytes,
            f"traffic stored only ~{intended_bytes} bytes against a "
            f"{capacity_bytes}-byte cap -- eviction never exercised; raise "
            f"EVICTION_N, or lower this model's capacity (the path default "
            f"or ModelSpec.eviction_capacity_gb) toward one cache object",
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
    engine_kwargs: dict[str, object] = {
        "gpu_memory_utilization": isolated_gpu_utilization(spec),
        # vLLM refuses a block pool smaller than one max-length request. The
        # pool must still be too small for the whole batch, or nothing is
        # preempted and the scenario asserts itself vacuous below.
        "num_gpu_blocks_override": spec.preemption_gpu_blocks or PREEMPTION_GPU_BLOCKS,
        "max_model_len": PREEMPTION_MAX_MODEL_LEN,
        "max_num_seqs": PREEMPTION_N,
        "disable_log_stats": False,  # the preemption counter needs stats on
    }
    requests = preemption_requests(PREEMPTION_N, PREEMPTION_MAX_TOKENS)
    failures: list[str] = []
    metrics: dict[str, object] = {}
    with _deployment_harness(spec, requests, engine_kwargs, metrics) as harness:
        stored_before = harness.stored_tokens_total()
        preemptions_before = vllm_preemption_total()
        # A preempted request is deliberately NOT reloaded from LMCache, so
        # the batch reports hits it then recomputes; that recompute is what
        # this scenario verifies.
        with harness.unloaded_hits_allowed():
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
        # Tolerances below are in units of the MODEL's chunk (harness.chunk),
        # not the module-level CHUNK. They are the same 16 for every uniform
        # model, but a hybrid's chunk is its unified block size (32 on Gemma
        # 4, 784 on Qwen3.8) and a tolerance stated in the wrong unit is
        # either vacuous or spuriously red.
        chunk = harness.chunk
        replayed_tokens = 0
        for req in requests:
            again = harness.run(req)
            replayed_tokens += again.lookup_tokens
            _check_text(failures, harness, req, again.text, f"T0.11 replay {req.key}")
            _expect(
                failures,
                again.lookup_hits >= again.lookup_tokens - 2 * chunk,
                f"{req.key}: replay hit only {again.lookup_hits} of "
                f"{again.lookup_tokens} tokens after the preemption batch",
            )

        # Under-storage guard: preemption must not silently drop stores. This
        # is an independent signal from the replay hits above, and the only
        # one left after this scenario opts out of the unloaded-hit rule
        # (harness.unloaded_hits_allowed): the replay proves the tokens came
        # BACK, this proves LMCache was asked to save them rather than vLLM's
        # own prefix cache having served them. No upper bound -- a preempted
        # request legitimately re-stores.
        #
        # The reference is the DISTINCT prompt-token count, summed over the
        # replay pass, not `batch.lookup_tokens - batch.lookup_hits`. A
        # request that waits in the queue is looked up again on every
        # scheduler step, so the batch counters count the same tokens many
        # times over: measured on Gemma 3-4B, 26730 "missed" tokens for six
        # ~700-token prompts, a 6.4x inflation that made the old bound
        # unsatisfiable. Any request queues as soon as the block pool cannot
        # admit the whole batch at once, which is precisely the pressure this
        # scenario exists to create -- so the old formula was invalid in its
        # own target regime, and it happened to pass only while every prompt
        # was small enough to be admitted in a single step. Counting distinct
        # tokens also removes the need for the decode-relookup slack the old
        # bound carried, which existed only to absorb the same double count.
        #
        # Slack per request: the leading shared region (these requests share
        # a salt and pad and differ only in image, so a later request may
        # legitimately find that prefix already stored and skip it -- one
        # image_span_margin by construction) plus one partial chunk, which
        # LMCache never stores.
        store_slack = PREEMPTION_N * (harness.image_span_margin + chunk)
        stored_delta = harness.stored_tokens_total() - stored_before
        _expect(
            failures,
            stored_delta >= replayed_tokens - store_slack,
            f"under-storage across preemption: the batch's six prompts hold "
            f"{replayed_tokens} distinct tokens but only {stored_delta} were "
            f"store-requested (slack {store_slack} = {PREEMPTION_N} x "
            f"({harness.image_span_margin} shared prefix + {chunk} partial "
            f"chunk))",
        )
        metrics["preemptions"] = preemptions
        metrics["chunk"] = chunk
        metrics["batch"] = {
            "lookup_tokens": batch.lookup_tokens,
            "lookup_hits": batch.lookup_hits,
            "distinct_prompt_tokens": replayed_tokens,
            "stored_delta": stored_delta,
        }
    return {"failures": failures, "metrics": metrics}


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
    engine_kwargs = {"gpu_memory_utilization": isolated_gpu_utilization(spec)}
    failures: list[str] = []
    metrics: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as tmp:
        baselines = compute_baselines(
            spec, requests, pathlib.Path(tmp), extra_engine_kwargs=engine_kwargs
        )
        server = start_mp_cache_server(
            zmq_port=zmq_port,
            http_port=http_port,
            # A hybrid model's chunk size must be vLLM's unified block size,
            # and its recurrent-state layers need their own cache objects.
            chunk_size=spec.hybrid_block_tokens or CHUNK,
            log_path=pathlib.Path(tmp) / "mp_server.log",
            l1_size_gb=(
                (spec.mp_server_l1_gb or MP_SERVER_L1_GB_HYBRID)
                if spec.hybrid_block_tokens
                else MP_SERVER_L1_GB
            ),
            separate_object_groups=bool(spec.hybrid_block_tokens),
            start_timeout_s=MP_SERVER_START_TIMEOUT_S,
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
                a2.lookup_hits >= a2.lookup_tokens - 2 * harness.chunk,
                f"repeat A hit only {a2.lookup_hits}/{a2.lookup_tokens}",
            )
            _expect(
                failures,
                b1.lookup_hits <= a2.lookup_hits - harness.image_span_margin,
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
                t2.lookup_hits >= t2.lookup_tokens - 2 * harness.chunk,
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
                q2.lookup_hits >= q1.lookup_tokens - 6 * harness.chunk,
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
                ba.lookup_hits <= ab2.lookup_hits - harness.image_span_margin,
                f"swapped order hit {ba.lookup_hits} vs full {ab2.lookup_hits}",
            )

            # T2.2 partial sharing. Gated on the same spec property as the
            # in-process case: the whole check rests on [A] being a token
            # prefix of [A, C], which is false for a model whose processor
            # lays out the image SET rather than appending each item. The
            # requests still RUN -- their outputs and the conservation audit
            # below are model-independent -- only the prefix assertions are
            # skipped.
            s1 = harness.run(cat["t22-A"])
            singles.append(s1)
            _check_text(failures, harness, cat["t22-A"], s1.text, "T3 t22 A")
            s2 = harness.run(cat["t22-AC"])
            singles.append(s2)
            _check_text(failures, harness, cat["t22-AC"], s2.text, "T3 t22 AC")
            if spec.media_prefix_stable:
                _expect(
                    failures,
                    s2.lookup_hits >= harness.image_span_margin + harness.chunk,
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
                stored >= missed - len(singles) * harness.chunk,
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
                after.lookup_hits >= after.lookup_tokens - 2 * harness.chunk,
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
                nc_b.lookup_hits > nc_b.lookup_tokens - harness.image_span_margin,
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
            server.process.terminate()
            try:
                server.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.process.kill()
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

# The routing module names the scenarios; this module implements them. A name
# only in one of the two places would surface as a model quietly running
# fewer scenarios than its certificate claims, so check the two agree here
# rather than at the point where one of them is missing.
_MISSING = set(ALL_SCENARIOS) - set(SCENARIOS)
_UNROUTED = set(SCENARIOS) - set(ALL_SCENARIOS)
if _MISSING or _UNROUTED:
    raise RuntimeError(
        f"scenario registry disagrees with isolated_routing: "
        f"routed but not implemented {sorted(_MISSING)}, "
        f"implemented but never routed {sorted(_UNROUTED)}"
    )


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
    # The prompt shape is normally set by the conftest and inherited by this
    # subprocess. Set it here too so a scenario invoked directly builds the
    # same prompts -- otherwise a hand-run scenario would send a system
    # message to a model whose template rejects it and fail for a reason
    # that has nothing to do with the scenario.
    if not spec.supports_system_role:
        os.environ["LMCACHE_MM_E2E_NO_SYSTEM_ROLE"] = "1"
    if spec.media_first_template:
        os.environ["LMCACHE_MM_E2E_MEDIA_FIRST"] = "1"
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
