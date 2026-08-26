# SPDX-License-Identifier: Apache-2.0
"""Engine harness for the multimodal acceptance suite.

Runs one vLLM engine against an external LMCache MP cache server (through
``LMCacheMPConnector``) and compares outputs against a baseline computed by
a plain vLLM engine in a subprocess.

The multi-process deployment is the ONLY one this suite drives. The
in-process ``LMCacheConnectorV1`` path was removed (it survives in git
history, branch ``archive/e2e_mm-inprocess-and-mp``): vLLM offers its
hybrid KV cache manager only to connectors advertising ``SupportsHMA``, so
that path cannot serve a multi-KV-group model at all, and on vLLM >= 0.26
its fused KV layout is corrupted outright (LMCache #4463 / #4467, which
both state the MP connector is unaffected). Certifying two paths only
diluted every verdict, so the suite states one.

vLLM itself still runs single-process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``):
the connector adapters this module wraps for its counters, and the
placeholder-substitution recorder, must live in this process.
"""

# Standard
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
import contextlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import warnings

# First Party (test-local)
from catalog import MMRequest
from specs import HybridFamily, ModelSpec

LMCACHE_TEST_CHUNK_SIZE = 16

# Salt for the startup probe that checks media/text ordering. Deliberately
# not a real case salt: it must never collide with one, and it must be a
# literal that survives tokenizer round-tripping unchanged.
_SHAPE_PROBE_SALT = "shapeprobe"


@dataclass(frozen=True)
class StorageSnapshot:
    """Point-in-time contents of the MP cache server's L1 pool.

    Attributes:
        num_keys: Number of cache objects currently resident in L1.
        total_bytes: Bytes those objects occupy in the L1 pool.
    """

    num_keys: int
    total_bytes: int


@dataclass(frozen=True)
class BatchResult:
    """Outcome of one concurrently scheduled batch of requests.

    Lookup counters cannot be attributed per request inside a batch, so only
    the aggregate deltas are reported; per-request verification inside a
    batch is output-based (baseline / semantic probe).

    Attributes:
        texts: Generated texts, in request order.
        lookup_tokens: Aggregate lookup tokens across the batch.
        lookup_hits: Aggregate lookup hits across the batch.
    """

    texts: tuple[str, ...]
    lookup_tokens: int
    lookup_hits: int


@dataclass(frozen=True)
class StepResult:
    """Outcome of one request on the LMCache engine.

    Attributes:
        text: The generated text.
        lookup_tokens: Tokens requested in the LMCache lookup for this step.
        lookup_hits: Tokens hit in the LMCache lookup for this step.
        identifiers: Multimodal identifiers the LMCache connector actually
            saw for this request (diagnostic; empty for text-only requests).
    """

    text: str
    lookup_tokens: int
    lookup_hits: int
    identifiers: tuple[str, ...] = ()


def configure_environment() -> None:
    """Set the env vars the engine runs require. Idempotent.

    Must be called before importing vllm or lmcache, and before launching
    the baseline subprocess (which inherits this environment).

    Carries no LMCache engine configuration: on the MP path the cache lives
    in the server process, which takes its chunk size and L1 capacity from
    the command line (see ``start_mp_cache_server``) and reads none of the
    ``LMCACHE_*`` engine variables the removed in-process path used.
    """
    # Keep vLLM's scheduler and worker in THIS process: the counters and the
    # identifier recorder are installed by wrapping connector-adapter and
    # vLLM methods here, and would observe nothing in a spawned worker.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("PYTHONHASHSEED", "0")
    # Triton JIT compiles small C launchers against Python.h using the
    # sysconfig include dir; when the interpreter's dev headers live in the
    # venv's include dir instead (headers installed manually), expose them
    # via CPATH so gcc finds them.
    venv_include = pathlib.Path(sys.prefix) / "include"
    if (venv_include / "Python.h").exists():
        existing = os.environ.get("CPATH", "")
        if str(venv_include) not in existing.split(":"):
            os.environ["CPATH"] = (
                f"{venv_include}:{existing}" if existing else str(venv_include)
            )


# Heartbeat window for every MP-path engine the suite starts, and the
# matching server-side reap timeout. Both are deliberately enormous: this
# turns the liveness probe into a single ping at startup and then, for the
# length of any run, silence.
#
# Why the suite must not run a live heartbeat. The interval is also the
# ping's patience (``send_ping(timeout=self._interval)``), and ONE ping
# that does not come back is read as server death -- no retry, no
# consecutive-failure count. The connector then reports its in-flight
# retrieve's blocks as load errors, and on a hybrid model vLLM cannot
# recover from that at all (see HYBRID_NOT_COVERED in certify.py): the run
# dies instead of degrading.
#
# Measured on Gemma 4-E4B (2026-08-21), three MME parity runs, each killed
# by the FIRST ping issued after the request flood began:
#
#   interval  workers  died at              ping
#   60s       1        300s after start     5th
#   60s       4        300s after start     5th
#   300s      16       600s after start     2nd
#
# Patience and pool size both moved the deadline and neither prevented it,
# so the ping is not merely queued behind the data plane -- its future
# never resolves while the client is saturated. That is the client-side
# response-dispatch defect already recorded in records/2026/08/21 (a real
# retrieve is reported back 0.3-20s after the server finishes it in ~3ms);
# Gemma 4 just amplifies it, because a 32-token chunk turns one 800-token
# prompt into ~25 lookups where a GDN hybrid needs one.
#
# Disabling the probe costs the suite nothing it was measuring: no oracle
# reads the health event, and a server that really died fails the run
# anyway (every load times out). What it does mean is that degraded-mode
# behaviour is NOT exercised here, which the certificate already states.
# The server refuses a reap timeout above its worker-registration grace
# ("worker registration grace must be >= the worker reap timeout"), so the
# grace is raised alongside it.
MP_HEARTBEAT_INTERVAL_S = 21600.0
MP_WORKER_REAP_TIMEOUT_S = 86400.0
MP_WORKER_REGISTRATION_GRACE_S = 86400.0

# CPU-side thread pool size for every MP cache server the suite starts,
# raised from the server's default of ONE because that pool serves PING and
# LOOKUP together, and Gemma 4's small chunk makes lookups frequent. It
# shortens queue waits; it is not what keeps the heartbeat quiet (see
# above).
MP_SERVER_CPU_WORKERS = 16

# L1 read-lock TTL, raised from the server default of 300s.
#
# A read lock is taken at LOOKUP time (StorageManager.submit_prefetch_task ->
# L1Manager.reserve_read) and consumed much later at transfer time
# (read_prefetched_results -> unsafe_read). TTLLock.is_locked() is
# `counter > 0 AND now < expiration`, and only lock() refreshes the
# expiration -- so if more than the TTL elapses between the two, the entry is
# silently no longer readable. unsafe_read then returns KEY_NOT_READABLE, the
# retrieve reports failure, and the load returns nothing.
#
# 300s is a sane bound for a live server, where lookup-to-transfer is
# milliseconds. It is the wrong bound for this suite, which submits all 2374
# MME questions in one llm.chat() call: vLLM looks a request's prefix up when
# it enters the waiting queue but only transfers once blocks free, so on a
# slow model the queue wait crosses 300s and every lock reserved before the
# crossing expires. That is what corrupted Gemma 4 -- the failures start 332s
# into pass2, not at question 1 -- while Gemma 3, whose whole two-pass run is
# 643s, never held a lock long enough to notice.
#
# Same reasoning as the reap timeout above: a batch benchmark is not a live
# server, so make the timeout irrelevant rather than tune it. This does not
# paper over a leak -- finish_read_prefetched still releases every lock; it
# only stops the clock from firing mid-queue.
MP_SERVER_L1_READ_TTL_S = 86400


@dataclass(frozen=True)
class MPServerHandle:
    """A running LMCache MP cache server subprocess.

    Attributes:
        process: The server ``subprocess.Popen`` handle.
        zmq_port: ZMQ port the connector must talk to.
        http_port: HTTP observability port (residency snapshots).
    """

    process: subprocess.Popen
    zmq_port: int
    http_port: int


def start_mp_cache_server(
    zmq_port: int,
    http_port: int,
    chunk_size: int,
    log_path: pathlib.Path,
    l1_size_gb: float,
    separate_object_groups: bool = False,
    start_timeout_s: float = 120.0,
) -> MPServerHandle:
    """Launch an LMCache MP cache server and wait until it is healthy.

    Args:
        zmq_port: ZMQ port for the connector traffic.
        http_port: HTTP port for the observability API.
        chunk_size: Server chunk size in tokens. For a hybrid model this
            MUST be vLLM's unified block size (or a multiple); the
            connector refuses to register otherwise.
        log_path: File capturing the server's stdout/stderr.
        l1_size_gb: L1 (host memory) pool size in GB.
        separate_object_groups: Give each KV cache group its own cache
            objects. Required for Mamba/GDN hybrids, whose recurrent-state
            layers must not share objects with the full-attention layers.
        start_timeout_s: How long to wait for the health endpoint.

    Returns:
        The running server handle.

    Raises:
        RuntimeError: If the server does not become healthy in time.
    """
    # Standard
    import time
    import urllib.error
    import urllib.request

    log_file = open(log_path, "w")
    # The server must run THIS repo's lmcache too: `-m` resolves through the
    # child's own sys.path, where the venv's editable install would
    # otherwise win. PYTHONPATH entries precede site-packages.
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{existing}" if existing else str(repo_root)
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.multiprocess.http_server",
        "--port",
        str(zmq_port),
        "--http-port",
        str(http_port),
        "--chunk-size",
        str(chunk_size),
        "--l1-size-gb",
        str(l1_size_gb),
        "--eviction-policy",
        "LRU",
        "--worker-reap-timeout-seconds",
        str(MP_WORKER_REAP_TIMEOUT_S),
        "--worker-registration-grace-seconds",
        str(MP_WORKER_REGISTRATION_GRACE_S),
        "--max-cpu-workers",
        str(MP_SERVER_CPU_WORKERS),
        "--l1-read-ttl-seconds",
        str(MP_SERVER_L1_READ_TTL_S),
    ]
    if separate_object_groups:
        command.append("--separate-object-groups")
    proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    deadline = time.monotonic() + start_timeout_s
    url = f"http://localhost:{http_port}/healthcheck"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return MPServerHandle(
                        process=proc, zmq_port=zmq_port, http_port=http_port
                    )
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    proc.terminate()
    raise RuntimeError(
        f"MP cache server failed to become healthy; log tail:\n"
        f"{log_path.read_text()[-2000:]}"
    )


def hybrid_engine_kwargs(
    block_tokens: int, family: HybridFamily = HybridFamily.RECURRENT_STATE
) -> dict[str, object]:
    """vLLM engine kwargs a multi-KV-group model requires.

    Empty when ``block_tokens`` is 0 (single KV cache group), and empty for
    a ``SLIDING_WINDOW`` hybrid too: its groups are all ordinary paged KV,
    differing only in window and block size, so it needs no cache mode of
    its own and keeps the suite's default scheduling (verified on Gemma
    4-E4B: the MP connector registers and loads with prefix caching off).

    For a ``RECURRENT_STATE`` hybrid the three settings are mandatory, not
    tuning (see ``docs/source/mp/hybrid_models.rst``): ``align`` is the
    only Mamba cache mode the GDN backends support, it only works with
    vLLM prefix caching enabled, and a scheduler step must advance at
    least one whole unified block for the state snapshot to land on a
    block boundary.

    Args:
        block_tokens: LMCache chunk size for the model
            (``ModelSpec.hybrid_block_tokens``); 0 for a single-group model.
        family: Which kind of multi-group cache this is
            (``ModelSpec.hybrid_family``). Defaults to
            ``RECURRENT_STATE``, the only family that existed when callers
            passed the block size alone.

    Returns:
        Engine kwargs to merge into every engine for this model (test
        engine, baseline engine, isolated scenarios, MME parity).
    """
    if not block_tokens or family is not HybridFamily.RECURRENT_STATE:
        return {}
    return {
        "mamba_cache_mode": "align",
        "enable_prefix_caching": True,
        "max_num_batched_tokens": block_tokens,
    }


def spec_engine_kwargs(spec: ModelSpec) -> dict[str, object]:
    """Engine kwargs every engine for this model MUST share.

    Every setting here changes the numeric regime, the model's geometry, or
    whether the model can be built at all, so an engine that has them and a
    baseline that does not are not comparable and their difference would be
    misattributed to LMCache. Kept in one function so the test engine, the
    baseline subprocess, the isolated scenarios and the MME parity runs
    cannot drift apart.

    Args:
        spec: The model under certification.

    Returns:
        Engine kwargs to merge into every engine for this model.
    """
    kwargs = hybrid_engine_kwargs(spec.hybrid_block_tokens, spec.hybrid_family)
    if spec.hf_overrides:
        kwargs["hf_overrides"] = dict(spec.hf_overrides)
    if spec.mm_encoder_attn_backend:
        kwargs["mm_encoder_attn_backend"] = spec.mm_encoder_attn_backend
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True
    return kwargs


def reset_vllm_prefix_cache(llm) -> str:
    """Empty vLLM's own prefix cache, by force if the public API refuses.

    Every caller that measures LMCache hits on a model running with vLLM
    prefix caching enabled (i.e. every Mamba/GDN hybrid, whose ``align``
    mode requires it) needs this. Measured on Qwen3.5-2B (2026-08-21): a
    repeated prompt whose KV is in LMCache is served ENTIRELY by vLLM's own
    cache (544 local / 0 external cached tokens) while the connector still
    reports a 544-token hit — the LMCache-side counters describe what the
    cache HELD, not what was loaded, so without this reset the hit
    arithmetic passes while the retrieve path never runs.

    ``LLM.reset_prefix_cache()`` refuses (warning, ``False``) while any GPU
    block is still referenced, and on the MP path exactly that happens: the
    connector keeps the most recent request's blocks (4 of 12405 measured)
    referenced until a later scheduler step releases them, and no step ever
    comes while the engine is idle. Retrying cannot help, so the index is
    cleared directly instead — the same two operations the public API
    performs after its refcount check, which is safe here because no
    request is running: the still-referenced blocks keep their data for the
    pending store, they merely stop being hit candidates, and vLLM's own
    eviction path already tolerates a hash-less block.

    Args:
        llm: The vLLM ``LLM`` engine.

    Returns:
        ``"public_api"`` if vLLM accepted the reset, ``"forced"`` if the
        index had to be cleared directly.
    """
    if llm.reset_prefix_cache():
        return "public_api"
    core = llm.llm_engine.engine_core
    core = getattr(core, "engine_core", core)
    pool = core.scheduler.kv_cache_manager.block_pool
    # Dropping the index is what stops local hits; lookups consult nothing
    # else. Hashes are reset only on blocks nobody references -- the
    # referenced ones belong to a request whose asynchronous store is still
    # in flight, and the connector tracks those by hash.
    pool.cached_block_hash_to_block = type(pool.cached_block_hash_to_block)()
    for block in pool.blocks:
        if block.ref_cnt == 0:
            block.reset_hash()
    return "forced"


class VllmPrefillCounters:
    """vLLM's own accounting of WHO served each prefilled token.

    The LMCache-side counters report what the cache held; these report what
    the engine actually skipped computing, split by provider. The pair is
    what makes a hit-count assertion meaningful: a replay served out of
    vLLM's GPU prefix cache raises ``local_cached`` and leaves
    ``external_cached`` at zero, while the LMCache counters look identical
    to a real load.

    Patches ``PrefillStats.set`` on the class, so at most one instance per
    process may be installed (the suite forces single-process vLLM, so the
    engine core runs here).

    Attributes:
        local_cached: Cumulative tokens served from vLLM's prefix cache.
        external_cached: Cumulative tokens served by the KV connector.
    """

    def __init__(self) -> None:
        self.local_cached = 0
        self.external_cached = 0

    def install(self) -> None:
        """Wrap ``PrefillStats.set``. Call before the engine is built."""
        # Third Party
        from vllm.v1.metrics.stats import PrefillStats

        counters = self
        original = PrefillStats.set

        def patched(
            stats,
            num_prompt_tokens: int,
            num_local_cached_tokens: int,
            num_external_cached_tokens: int,
        ):
            counters.local_cached += num_local_cached_tokens
            counters.external_cached += num_external_cached_tokens
            return original(
                stats,
                num_prompt_tokens,
                num_local_cached_tokens,
                num_external_cached_tokens,
            )

        PrefillStats.set = patched


def mp_kv_transfer_config(zmq_port: int):
    """vLLM KV-transfer config selecting THIS repo's MP connector.

    Shared by every caller (``MMHarness``, the MME parity run) so
    the connector module path, the server address keys and the heartbeat
    window cannot drift apart between them. The window is widened from the
    10s default for the reason given at ``MP_HEARTBEAT_INTERVAL_S``; the
    servers this module starts raise their reap timeout to match.

    Args:
        zmq_port: ZMQ port of the running MP cache server.

    Returns:
        The ``KVTransferConfig`` to hand to ``LLM(...)``.
    """
    # Third Party
    from vllm.config import KVTransferConfig

    return KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_connector_module_path="lmcache.integration.vllm.lmcache_mp_connector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "lmcache.mp.host": "tcp://localhost",
            "lmcache.mp.port": zmq_port,
            "lmcache.mp.heartbeat_interval": MP_HEARTBEAT_INTERVAL_S,
        },
    )


class MPTransportCounters:
    """Lookup and store counters for the MP deployment path.

    The MP connector reports no process-local LMCache stats, so the counters
    come from wrapping the adapter methods. The wrappers are installed on
    the CLASSES, which bounds this to at most one live instance per
    process (the suite forces single-process vLLM, so the worker adapter
    lives here too). Lookups resolve asynchronously: token counts are
    remembered at submit time and booked, with the hit count, when
    ``check_lookup_result`` first returns a value for the request.

    Attributes:
        lookup_tokens: Cumulative tokens submitted for lookup.
        lookup_hits: Cumulative tokens the server reported as hits.
        stored_tokens: Cumulative tokens submitted for storage.
        lookup_request_tokens: Prompt length of every resolved lookup, in
            resolution order. Lets a caller compute how much of a workload
            was cacheable at all (whole chunks only), which a store-token
            total cannot: identical prefixes submitted together are stored
            once and hit once per request.
    """

    def __init__(self) -> None:
        self.lookup_tokens = 0
        self.lookup_hits = 0
        self.stored_tokens = 0
        self.lookup_request_tokens: list[int] = []
        self._pending_lookup_tokens: dict[str, int] = {}

    def install(self) -> None:
        """Wrap the MP adapter methods. Call before the engine is built."""
        # First Party
        from lmcache.integration.vllm.vllm_multi_process_adapter import (
            LMCacheMPSchedulerAdapter,
            LMCacheMPWorkerAdapter,
        )

        counters = self
        original_submit = LMCacheMPSchedulerAdapter.maybe_submit_lookup_request
        original_check = LMCacheMPSchedulerAdapter.check_lookup_result
        original_store = LMCacheMPWorkerAdapter.batched_submit_store_requests

        def submit(adapter, request_id, token_ids, *args, **kwargs):
            counters._pending_lookup_tokens.setdefault(request_id, len(token_ids))
            return original_submit(adapter, request_id, token_ids, *args, **kwargs)

        def check(adapter, request_id):
            ret = original_check(adapter, request_id)
            if ret is not None:
                tokens = counters._pending_lookup_tokens.pop(request_id, None)
                if tokens is not None:
                    counters.lookup_tokens += tokens
                    counters.lookup_hits += ret
                    counters.lookup_request_tokens.append(tokens)
            return ret

        def store(adapter, request_ids, ops, *args, **kwargs):
            for op in ops:
                counters.stored_tokens += op.end - op.start
            return original_store(adapter, request_ids, ops, *args, **kwargs)

        LMCacheMPSchedulerAdapter.maybe_submit_lookup_request = submit
        LMCacheMPSchedulerAdapter.check_lookup_result = check
        LMCacheMPWorkerAdapter.batched_submit_store_requests = store


def mm_limits(spec: ModelSpec) -> dict[str, int]:
    """Per-prompt multimodal item limits for the engines of ``spec``.

    Shared by the LMCache engine, the baseline engine, and every isolated
    scenario so all of them accept the same request shapes.

    Args:
        spec: The model under certification.

    Returns:
        The ``limit_mm_per_prompt`` mapping.
    """
    limits = {"image": 2}
    if "video" in spec.modalities:
        limits["video"] = 1
    if "audio" in spec.modalities:
        # One clip: naming two clips in order was measured at 0/9 correct,
        # so a multi-clip audio case would have no usable semantic probe.
        limits["audio"] = 1
    return limits


def _prometheus_counter_totals(names: tuple[str, ...]) -> dict[str, int]:
    """Read the current totals of the given Prometheus counters.

    Args:
        names: Prometheus metric names (without the ``_total`` suffix).

    Returns:
        Mapping of metric name to its integer total (0 if not yet emitted).
    """
    # Third Party
    from prometheus_client import REGISTRY

    totals: dict[str, int] = {name: 0 for name in names}
    for metric in REGISTRY.collect():
        if metric.name in totals:
            totals[metric.name] = int(
                sum(
                    sample.value
                    for sample in metric.samples
                    if sample.name.endswith("_total")
                )
            )
    return totals


def vllm_preemption_total() -> int:
    """Cumulative vLLM scheduler preemptions since engine start.

    Requires the engine to run with ``disable_log_stats=False`` (the offline
    ``LLM`` API disables stat logging -- and thus this counter -- by
    default). Scenarios that must PROVE a preemption happened read this.

    Returns:
        The current ``vllm:num_preemptions`` counter total.
    """
    totals = _prometheus_counter_totals(("vllm:num_preemptions",))
    return totals["vllm:num_preemptions"]


_NO_EXTRA_KWARGS: Mapping[str, object] = MappingProxyType({})


def effective_max_tokens(spec: ModelSpec, request: MMRequest) -> int:
    """A request's decode budget after the spec's ``min_decode_tokens`` floor.

    Args:
        spec: The model under certification.
        request: The request whose budget to compute.

    Returns:
        ``max(request.max_tokens, spec.min_decode_tokens)`` — the budget the
        harness, the baseline runner, and any decode-token accounting must
        all use consistently.
    """
    return max(request.max_tokens, spec.min_decode_tokens)


class MMHarness:
    """Drives one model's acceptance run: baselines + LMCache engine + stats.

    The engine talks to an already-running external LMCache MP cache server
    over ZMQ through ``LMCacheMPConnector`` (this repo's tip version,
    selected via ``kv_connector_module_path``). Lookup tokens/hits and store
    intent come from ``MPTransportCounters``, whose wrappers are installed
    on the adapter CLASSES -- so at most one harness may be live per
    process; residency comes from the server's HTTP ``/status`` endpoint.

    Args:
        spec: The model under certification.
        baselines: Mapping of request key to the plain-vLLM output text.
        zmq_port: The MP cache server's ZMQ port.
        http_port: The MP cache server's HTTP observability port.
        extra_engine_kwargs: Additional/overriding vLLM ``LLM(...)`` kwargs;
            used by isolated scenarios (chunked prefill, capacity eviction)
            to reshape the engine while reusing all harness plumbing.
    """

    def __init__(
        self,
        spec: ModelSpec,
        baselines: dict[str, str],
        zmq_port: int,
        http_port: int,
        extra_engine_kwargs: Mapping[str, object] = _NO_EXTRA_KWARGS,
    ):
        configure_environment()
        self._http_port = http_port
        self._counters = MPTransportCounters()
        self.spec = spec
        self.baselines = baselines
        # Diagnostic recorder: capture the multimodal identifiers the
        # connector substitutes, so false-hit failures can name the request
        # pair involved. Installed before the engine imports the adapter.
        self._identifier_log: list[str] = []
        self._identity_blind = False
        self._install_identifier_recorder()
        # vLLM's own view of who served each prefilled token, the ground
        # truth every hit assertion is checked against.
        self._prefill = VllmPrefillCounters()
        self._prefill.install()
        self._unloaded_hits_allowed = False
        # Counter wrappers must be in place before vLLM imports the adapter.
        self._counters.install()

        # Third Party
        from vllm import LLM

        engine_kwargs: dict[str, object] = dict(
            model=spec.hf_id,
            kv_transfer_config=mp_kv_transfer_config(zmq_port),
            max_model_len=spec.max_model_len,
            gpu_memory_utilization=spec.gpu_memory_utilization,
            enforce_eager=True,
            enable_prefix_caching=False,
            limit_mm_per_prompt=mm_limits(spec),
        )
        engine_kwargs.update(spec_engine_kwargs(spec))
        engine_kwargs.update(extra_engine_kwargs)
        self.llm = LLM(**engine_kwargs)
        self._validate_block_size()
        self._validate_prompt_shape()

    # Store settling: a store is asynchronous twice over -- the worker
    # submits it around the time the engine answers, and the server holds
    # each key write-locked between reserve_write and
    # finish_write (50-300ms observed under load). Without a barrier, a
    # back-to-back request's lookup beats the previous request's store
    # (KEY_NOT_EXIST) or lands inside the write-lock window
    # (KEY_NOT_READABLE), and the prefix fold turns one such chunk into a
    # whole-prompt miss. The suite certifies cache correctness, not request
    # pacing, so each run waits for its own stores to become readable.
    _SETTLE_SUBMIT_TIMEOUT_S = 10.0
    _SETTLE_SERVER_TIMEOUT_S = 30.0
    _SETTLE_POLL_INTERVAL_S = 0.1
    _SETTLE_QUIET_POLLS = 4

    @property
    def chunk(self) -> int:
        """LMCache chunk size in tokens — the granularity of every hit.

        Tests read their hit-count tolerances from this instead of the
        module constant: a hybrid model's chunk must equal vLLM's unified
        block size (hundreds of tokens), not the 16-token default.
        """
        return self.spec.hybrid_block_tokens or LMCACHE_TEST_CHUNK_SIZE

    @property
    def objects_per_chunk(self) -> int:
        """Cache objects stored per token chunk.

        One for a plain paged-KV model; a hybrid model running with
        ``--separate-object-groups`` stores one object per KV cache group
        (full-attention KV plus recurrent state), which the
        storage-conservation bounds must account for.
        """
        return self.spec.hybrid_object_groups or 1

    @property
    def image_span_margin(self) -> int:
        """Hit-count separation that proves a hit did NOT reach the media.

        Test images are 448x448 (hundreds of placeholder tokens), so four
        chunks of separation is unambiguous at the 16-token default. At a
        hybrid model's block granularity the image span is smaller than one
        block, so the margin is expressed in blocks instead: the padded
        hybrid prompt places whole shared blocks before the image and
        several more after it, and a request that differs only in image
        content hits exactly the leading shared blocks — while a false hit
        reaches the trailing blocks too.

        The same margin serves audio, but only because the clip length was
        chosen to make it valid: audio expands at ~13 tokens/second, so
        ``catalog.AUDIO_SECONDS`` is set to give a ~105-token span, wider
        than the 64 tokens subtracted here. A shorter clip would make this
        assertion unsatisfiable rather than merely weak.
        """
        return (2 if self.spec.hybrid_block_tokens else 4) * self.chunk

    def expected_full_hit(self, lookup_tokens: int) -> int:
        """Hit count a fully cached prompt of ``lookup_tokens`` must reach.

        LMCache stores whole chunks and never serves the final token from
        cache, so a fully stored prompt hits every chunk that ends before
        its last token.

        Args:
            lookup_tokens: The prompt's lookup token count.

        Returns:
            The expected hit count for a full-hit replay.
        """
        return self.chunk * ((lookup_tokens - 1) // self.chunk)

    def _validate_block_size(self) -> None:
        """Fail loudly if a hybrid spec's declared chunk size is wrong.

        ``ModelSpec.hybrid_block_tokens`` drives the MP server's chunk
        size, the scheduler budget, and every hit tolerance in the suite.
        vLLM derives each group's real block size from the model's head
        dimensions and state size, so a stale declared value would
        silently produce meaningless (or trivially passing) assertions.

        Checks LMCache's own rule rather than equality against
        ``cache_config.block_size``: the chunk must be a multiple of every
        PAGED group's block size (recurrent-state groups keep one page per
        sequence and report ``tokens_per_block = 0``, so they impose
        nothing). Equality happens to hold for the Mamba/GDN hybrids,
        whose only paged group is full attention, but not for a
        sliding-window hybrid whose paged groups sit at different block
        sizes -- there the correct chunk is their common multiple and
        ``cache_config.block_size`` is the smallest of them.

        Raises:
            RuntimeError: If the declared chunk is not a multiple of some
                paged group's block size.
        """
        if not self.spec.hybrid_block_tokens:
            return
        chunk = self.spec.hybrid_block_tokens
        offenders = [
            f"{name} block_size={size}"
            for name, size in self._paged_group_block_sizes().items()
            if chunk % size
        ]
        if offenders:
            raise RuntimeError(
                f"{self.spec.key}: spec declares hybrid_block_tokens={chunk}, "
                f"which is not a multiple of {', '.join(offenders)}; LMCache "
                f"refuses to register such a chunk, and the value also drives "
                f"the MP server chunk size and every hit tolerance"
            )

    def _validate_prompt_shape(self) -> None:
        """Fail loudly if ``media_first_template`` misdescribes the model.

        The suite isolates its cases with a salt at the head of the prompt,
        which only holds while the chat template renders text before media.
        A model whose template hoists media above the conversation needs the
        case identity in the media bytes instead
        (``catalog.case_media_bits``), and a WRONG declaration is silent in
        both directions: declared False when true, two cases share a
        byte-identical media prefix and the cross-image assertions compare
        against the wrong entry; declared True when false, the media carries
        redundant bits and the model is measured on prompts no other model
        was.

        Checked against the live tokenizer rather than a list of model
        names: the marker's position is found by rendering the same request
        with and without its image and diffing, so it needs no per-model
        knowledge of what the placeholder looks like.

        Raises:
            RuntimeError: If the rendered order disagrees with the spec, or
                if the probe cannot be rendered at all.
        """
        # First Party (test-local)
        from catalog import color_request

        probe = color_request("shapeprobe", _SHAPE_PROBE_SALT, 0)
        with_media = probe.messages()
        without_media = [
            {
                **message,
                "content": [
                    item for item in message["content"] if item.get("type") == "text"
                ],
            }
            if isinstance(message.get("content"), list)
            else message
            for message in with_media
        ]
        tokenizer = self.llm.get_tokenizer()
        # Qwen3-Omni keeps its chat template on the PROCESSOR, not the
        # tokenizer, so a tokenizer-only render raises for it. Fall back
        # rather than fail: this check must not be able to break a model it
        # has nothing to say about.
        template = getattr(tokenizer, "chat_template", None)
        if not template:
            template = self._processor_chat_template()
        try:
            rendered = tokenizer.apply_chat_template(
                with_media,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=template,
            )
            bare = tokenizer.apply_chat_template(
                without_media,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=template,
            )
        except Exception as exc:
            raise RuntimeError(
                f"{self.spec.key}: could not render a probe prompt to check "
                f"the media/text order that case isolation depends on: {exc}"
            ) from exc
        media_at = next(
            (i for i, (a, b) in enumerate(zip(rendered, bare, strict=False)) if a != b),
            len(bare),
        )
        salt_at = rendered.find(_SHAPE_PROBE_SALT)
        if salt_at < 0:
            raise RuntimeError(
                f"{self.spec.key}: the case salt does not appear in the "
                f"rendered prompt at all, so nothing isolates one case from "
                f"another"
            )
        media_first = media_at < salt_at
        if media_first != self.spec.media_first_template:
            raise RuntimeError(
                f"{self.spec.key}: spec declares "
                f"media_first_template={self.spec.media_first_template}, but "
                f"the chat template renders the media marker at character "
                f"{media_at} and the case salt at {salt_at}. Case isolation "
                f"depends on which comes first (see "
                f"catalog.case_media_bits), so this must be declared "
                f"correctly, not left at the default"
            )

    def _processor_chat_template(self) -> str:
        """The chat template this model keeps on its processor, if any.

        Returns:
            The template string, or '' when the model has no processor-level
            template (then the tokenizer's own is the only one there is).
        """
        # Third Party
        from transformers import AutoProcessor

        try:
            processor = AutoProcessor.from_pretrained(
                self.spec.hf_id, trust_remote_code=self.spec.trust_remote_code
            )
        except Exception:
            return ""
        return getattr(processor, "chat_template", "") or ""

    def _paged_group_block_sizes(self) -> dict[str, int]:
        """Block size of every paged KV cache group, by spec class name.

        Returns:
            Mapping of KV-cache-spec class name to that group's block size,
            covering only groups vLLM pages by token (recurrent-state
            groups hold one page per sequence and are excluded).
        """
        core = self.llm.llm_engine.engine_core.engine_core
        sizes: dict[str, int] = {}
        for group in core.scheduler.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            name = type(spec).__name__
            if "Mamba" in name:
                continue
            sizes[name] = spec.block_size
        return sizes

    def _install_identifier_recorder(self) -> None:
        """Wrap the connector's placeholder substitution to log identifiers.

        Normally a read-only observation: the wrapper calls through to the
        original function unchanged. Under ``identity_blindness()`` it
        instead skips the substitution, reproducing the ur-failure-mode
        (cache keys ignore image content) so tests can prove the suite's
        detectors actually fire. Must run before vLLM imports the LMCache
        adapter (which binds the function by name at import time).
        """
        # First Party
        import lmcache.integration.vllm.utils as lmc_utils

        original = lmc_utils.apply_mm_hashes_to_token_ids
        log = self._identifier_log
        harness = self

        def recording(token_ids, mm_hashes, mm_positions):
            log.extend(mm_hashes)
            if harness._identity_blind:
                return token_ids
            return original(token_ids, mm_hashes, mm_positions)

        lmc_utils.apply_mm_hashes_to_token_ids = recording

    @contextlib.contextmanager
    def identity_blindness(self):
        """Disable MM identity substitution (negative control).

        While active, LMCache keys are computed from the raw placeholder
        token IDs — exactly the failure mode the substitution exists to
        prevent — so two different same-shape images collide with certainty.
        Use with case-unique salts only; entries stored while blind are
        isolated from other cases by their salt.
        """
        self._identity_blind = True
        try:
            yield
        finally:
            self._identity_blind = False

    def close(self) -> None:
        """Tear down the engine (the MP server is managed by the caller)."""
        del self.llm

    def run(self, request: MMRequest) -> StepResult:
        """Send one request through the LMCache engine and read its stats.

        Per-request stats are deltas of the cumulative transport counters
        (see ``_cumulative_lookup_stats``), and the returned stats are
        unaffected by the store barrier below: they are read before it
        starts.
        """
        # Third Party
        from vllm import SamplingParams

        self._reset_local_prefix_cache()
        stored_before = self.stored_tokens_total()
        tokens_before, hits_before = self._cumulative_lookup_stats()
        log_before = len(self._identifier_log)
        provenance_before = self._prefill_provenance()
        outputs = self.llm.chat(
            request.messages(),
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=effective_max_tokens(self.spec, request),
                seed=0,
                ignore_eos=request.ignore_eos,
            ),
            chat_template_kwargs=dict(self.spec.chat_template_kwargs) or None,
            use_tqdm=False,
        )
        tokens_after, hits_after = self._cumulative_lookup_stats()
        seen = self._identifier_log[log_before:]
        self._check_hit_provenance(
            hits_after - hits_before, provenance_before, request.key
        )
        result = StepResult(
            text=outputs[0].outputs[0].text,
            lookup_tokens=tokens_after - tokens_before,
            lookup_hits=hits_after - hits_before,
            identifiers=tuple(dict.fromkeys(seen)),
        )
        # Everything the lookup missed must be store-submitted, except the
        # trailing partial chunk, which LMCache never stores.
        expected_new = result.lookup_tokens - result.lookup_hits - self.chunk
        self._settle_stores(stored_before + max(0, expected_new))
        return result

    def run_batch(self, requests: list[MMRequest]) -> BatchResult:
        """Submit all requests in ONE ``llm.chat`` call (concurrent batch).

        The vLLM scheduler interleaves the requests' prefills and decodes,
        so LMCache sees concurrent lookup/store traffic — including a store
        for one request racing the lookup of an identical one. Counters are
        aggregate only (per-request attribution is impossible in a batch).

        Identical prefixes inside one batch are stored once but missed by
        every request, so the aggregate miss count overstates what will be
        submitted; the trailing store barrier therefore uses no submission
        target and relies on quiescence alone.
        """
        # Third Party
        from vllm import SamplingParams

        self._reset_local_prefix_cache()
        tokens_before, hits_before = self._cumulative_lookup_stats()
        provenance_before = self._prefill_provenance()
        outputs = self.llm.chat(
            [r.messages() for r in requests],
            sampling_params=[
                SamplingParams(
                    temperature=0.0,
                    max_tokens=effective_max_tokens(self.spec, r),
                    seed=0,
                    ignore_eos=r.ignore_eos,
                )
                for r in requests
            ],
            chat_template_kwargs=dict(self.spec.chat_template_kwargs) or None,
            use_tqdm=False,
        )
        tokens_after, hits_after = self._cumulative_lookup_stats()
        # Requests inside one batch legitimately share prefixes (that is
        # what the concurrency cases exercise), so vLLM's own cache may
        # serve some of them; only the accounting identity is checked.
        self._check_hit_provenance(
            hits_after - hits_before,
            provenance_before,
            f"batch of {len(requests)}",
            num_requests=len(requests),
            local_budget=tokens_after - tokens_before,
        )
        result = BatchResult(
            texts=tuple(o.outputs[0].text for o in outputs),
            lookup_tokens=tokens_after - tokens_before,
            lookup_hits=hits_after - hits_before,
        )
        self._settle_stores(self.stored_tokens_total())
        return result

    def _prefill_provenance(self) -> tuple[int, int]:
        """Snapshot vLLM's cumulative (local, external) cached tokens."""
        return (self._prefill.local_cached, self._prefill.external_cached)

    def _check_hit_provenance(
        self,
        lookup_hits: int,
        before: tuple[int, int],
        where: str,
        num_requests: int = 1,
        local_budget: int = 0,
    ) -> None:
        """Verify the reported hits were really served by LMCache.

        The LMCache counters report what the cache HELD for a prompt, not
        what the engine loaded: with vLLM prefix caching on (mandatory for
        hybrids) a replay can be served entirely out of GPU memory while
        the connector still reports a full hit, which would make every
        hit-count assertion in the suite pass without the retrieve path
        running once. vLLM's own per-prefill split settles it.

        Args:
            lookup_hits: Hits LMCache reported for this step.
            before: The ``_prefill_provenance()`` snapshot taken before it.
            where: Label for the error message (request key or batch size).
            num_requests: Requests in the step; each one is allowed the
                single final token vLLM always recomputes.
            local_budget: Tokens vLLM's own cache may legitimately have
                served — 0 for a single request (it cannot share a prefix
                with itself), the step's lookup tokens for a batch, whose
                requests are meant to share prefixes with each other.

        Raises:
            RuntimeError: If vLLM's cache served tokens it should not have,
                or if the reported hits were not actually loaded.
        """
        local = self._prefill.local_cached - before[0]
        external = self._prefill.external_cached - before[1]
        if local > local_budget:
            raise RuntimeError(
                f"{where}: vLLM's own prefix cache served {local} tokens "
                f"(budget {local_budget}); LMCache's {lookup_hits} reported "
                f"hits would be measuring the wrong cache"
            )
        if self._unloaded_hits_allowed:
            return
        # vLLM always recomputes a prompt's final token, so a full-prompt
        # hit loads one token fewer than the connector reports (its own log
        # line reads "LMCache hit tokens: 304, need to load: 303") — one
        # token of slack per request in the step.
        if external < lookup_hits - local - num_requests:
            raise RuntimeError(
                f"{where}: LMCache reported {lookup_hits} hit tokens but "
                f"vLLM only skipped {external} on the connector's account "
                f"({local} locally cached); the retrieve path did not run "
                f"for the difference"
            )

    @contextlib.contextmanager
    def unloaded_hits_allowed(self):
        """Suspend the "reported hits were loaded" check for a scenario.

        LMCache declines to serve a PREEMPTED request (an explicit early
        return in the connector), so the preemption scenario legitimately
        reports hits for requests that then recompute — that recompute is
        exactly what it verifies. The check that vLLM's own prefix cache
        stayed out of the way stays active.
        """
        self._unloaded_hits_allowed = True
        try:
            yield
        finally:
            self._unloaded_hits_allowed = False

    def _reset_local_prefix_cache(self) -> None:
        """Drop vLLM's OWN prefix cache so LMCache is the only hit source.

        No-op for models the suite runs with vLLM prefix caching disabled
        (the default, and what every model but a recurrent-state hybrid
        gets). A recurrent-state hybrid cannot disable it — ``align`` mode
        requires it — so without this reset vLLM would serve repeats from
        GPU memory while LMCache still reported a hit, and the suite's hit
        arithmetic would describe the wrong cache. Conditioned on the
        engine's own setting rather than on the spec being hybrid: a
        sliding-window hybrid is on the MP path with prefix caching off,
        and there is no cache to reset. That the reset worked is verified
        per step by ``_check_hit_provenance``; see
        ``reset_vllm_prefix_cache`` for why it has to be forced.
        """
        if self.llm.llm_engine.vllm_config.cache_config.enable_prefix_caching:
            reset_vllm_prefix_cache(self.llm)

    def _cumulative_lookup_stats(self) -> tuple[int, int]:
        """Cumulative (lookup_tokens, lookup_hits) since engine start."""
        return (self._counters.lookup_tokens, self._counters.lookup_hits)

    def stored_tokens_total(self) -> int:
        """Cumulative tokens submitted to the MP server for storage."""
        return self._counters.stored_tokens

    def storage(self) -> StorageSnapshot:
        """Resident object count/bytes from the MP server's /status API."""
        l1 = self._fetch_l1_status()
        return StorageSnapshot(
            num_keys=l1["total_object_count"],
            total_bytes=l1["memory_used_bytes"],
        )

    def _fetch_l1_status(self) -> dict:
        """Fetch the server L1 manager's status dict over HTTP.

        Returns:
            The ``storage_manager.l1_manager`` section of the ``/status``
            response; the keys used here are ``total_object_count``,
            ``write_locked_count``, and ``memory_used_bytes``.
        """
        # Standard
        import urllib.request

        url = f"http://localhost:{self._http_port}/status"
        with urllib.request.urlopen(url, timeout=30) as resp:
            status = json.loads(resp.read())
        return status["storage_manager"]["l1_manager"]

    def _settle_stores(self, min_stored_tokens: int) -> None:
        """Block until in-flight stores are submitted and readable.

        Two phases, both bounded and non-fatal (on timeout the caller's own
        assertions report whatever is actually wrong):

        1. Submission: wait until the engine-side store counter reaches
           ``min_stored_tokens``, so quiescence below cannot be observed in
           the gap before the worker has even submitted the store.
        2. Quiescence: wait until the server holds no write-locked objects
           and (store counter, object count, lock count) is unchanged for
           ``_SETTLE_QUIET_POLLS`` consecutive polls, i.e. the submitted
           stores have arrived and finished writing.

        Args:
            min_stored_tokens: Cumulative ``stored_tokens_total`` value to
                wait for in phase 1.
        """
        deadline = time.monotonic() + self._SETTLE_SUBMIT_TIMEOUT_S
        while (
            self.stored_tokens_total() < min_stored_tokens
            and time.monotonic() < deadline
        ):
            time.sleep(self._SETTLE_POLL_INTERVAL_S)

        deadline = time.monotonic() + self._SETTLE_SERVER_TIMEOUT_S
        quiet = 0
        previous: tuple[int, int, int] = (-1, -1, -1)
        while time.monotonic() < deadline:
            l1 = self._fetch_l1_status()
            current = (
                self.stored_tokens_total(),
                l1["total_object_count"],
                l1["write_locked_count"],
            )
            quiet = quiet + 1 if current == previous and current[2] == 0 else 0
            previous = current
            if quiet >= self._SETTLE_QUIET_POLLS:
                return
            time.sleep(self._SETTLE_POLL_INTERVAL_S)

    def check_output(self, request: MMRequest, result: StepResult, where: str) -> None:
        """Verify a step's output; see ``check_text`` for the policy."""
        self.check_text(request, result.text, where)

    def check_text(self, request: MMRequest, text: str, where: str) -> None:
        """Verify one generated text against baseline and semantic probe.

        Policy: exact match against the plain-vLLM baseline is required. If
        the exact match fails but the semantic probe still passes, the step
        passes with a warning (GPU nondeterminism); if the probe also fails,
        this is cross-image contamination and the step fails hard.

        Args:
            request: The request that produced ``text``.
            text: The generated text to verify.
            where: Human-readable context for failure messages.

        Raises:
            AssertionError: On baseline mismatch without probe rescue, or on
                probe failure.
        """
        probe_ok = self.probe_ok(request, text)
        if request.key in self.baselines:
            baseline = self.baselines[request.key]
            if text == baseline:
                return
            if probe_ok:
                warnings.warn(
                    f"[{where}] {request.key}: exact baseline mismatch but "
                    f"semantic probe passed (got {text!r}, "
                    f"baseline {baseline!r})",
                    stacklevel=2,
                )
                return
            raise AssertionError(
                f"[{where}] {request.key}: output diverged from baseline AND "
                f"semantic probe failed -- cross-image contamination. "
                f"got={text!r} baseline={baseline!r} "
                f"expected_probe={request.expected_probe}"
            )
        if not probe_ok:
            raise AssertionError(
                f"[{where}] {request.key}: semantic probe failed. "
                f"got={text!r} expected_probe={request.expected_probe}"
            )

    def extracted_answer(self, text: str) -> str:
        """The model's final answer inside ``text`` per the spec's pattern.

        Returns:
            group(1) of the LAST match of ``spec.answer_extract_pattern``
            (a preamble may open a spurious marker; the final answer is the
            last closed one), or '' when the spec has no pattern or the
            pattern does not match.
        """
        if not self.spec.answer_extract_pattern:
            return ""
        matches = re.findall(self.spec.answer_extract_pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else ""

    def check_replay_text(
        self, request: MMRequest, reference_text: str, text: str, where: str
    ) -> None:
        """Verify a hit-path replay against its own miss-path output.

        The miss pass (KV computed) and the hit pass (KV loaded from
        LMCache) are different numeric regimes, so byte equality is expected
        but not guaranteed; a verbose answer style (GLM preambles) gives the
        regime noise many tokens to flip. Policy: exact match passes; a
        mismatch passes with a warning when the extracted final answers
        match (non-empty) or the semantic probe passes; otherwise the replay
        fails hard — KV corruption or contamination flips the answer itself,
        not just the phrasing.

        Args:
            request: The request replayed.
            reference_text: The miss-path (first pass) generated text.
            text: The hit-path (replay) generated text.
            where: Human-readable context for failure messages.

        Raises:
            AssertionError: On divergence not rescued by an extracted-answer
                match or the semantic probe.
        """
        if text == reference_text:
            return
        extracted = self.extracted_answer(text)
        if extracted and extracted == self.extracted_answer(reference_text):
            warnings.warn(
                f"[{where}] {request.key}: hit-path text diverged from "
                f"miss-path but extracted answers match ({extracted!r}); "
                f"got {text!r}, reference {reference_text!r}",
                stacklevel=2,
            )
            return
        if request.expected_probe and self.probe_ok(request, text):
            warnings.warn(
                f"[{where}] {request.key}: hit-path text diverged from "
                f"miss-path but semantic probe passed (got {text!r}, "
                f"reference {reference_text!r})",
                stacklevel=2,
            )
            return
        raise AssertionError(
            f"[{where}] {request.key}: hit-path output diverged from its "
            f"miss-path reference AND no rescue (extracted answer/probe) "
            f"applies. got={text!r} reference={reference_text!r} "
            f"expected_probe={request.expected_probe}"
        )

    def probe_ok(self, request: MMRequest, text: str) -> bool:
        """Whether the expected probe words appear in ``text`` IN ORDER.

        Order matters for multi-image probes: a swapped-order answer means
        the request hit the other ordering's cache entries.

        Args:
            request: The request whose ``expected_probe`` to check.
            text: The generated text.

        Returns:
            True if every probe word appears in order (or no probe is set).
        """
        if not request.expected_probe:
            return True
        lowered = text.lower()
        position = 0
        for word in request.expected_probe:
            found = lowered.find(word, position)
            if found < 0:
                return False
            position = found + len(word)
        return True


def compute_baselines(
    spec: ModelSpec,
    requests: list[MMRequest],
    workdir: pathlib.Path,
    extra_engine_kwargs: Mapping[str, object] = _NO_EXTRA_KWARGS,
) -> dict[str, str]:
    """Run all baseline-needing requests on a plain vLLM engine (subprocess).

    Args:
        spec: The model under certification.
        requests: Requests to run; only those with ``needs_baseline`` run.
        workdir: Directory for the input/output JSON files.
        extra_engine_kwargs: Additional/overriding engine kwargs, so isolated
            scenarios get a baseline under the SAME scheduling config as the
            engine under test (must be JSON-serializable).

    Returns:
        Mapping of request key to generated text.

    Raises:
        RuntimeError: If the baseline subprocess fails.
    """
    todo = [r for r in requests if r.needs_baseline]
    # The baseline must run the same mandatory engine settings as the engine
    # under test (align mode changes the numeric regime; hf_overrides change
    # the model's geometry), so an output difference can only come from
    # LMCache.
    extra_engine_kwargs = {
        **spec_engine_kwargs(spec),
        **extra_engine_kwargs,
    }
    spec_json = {
        "model": spec.hf_id,
        "max_model_len": spec.max_model_len,
        "gpu_memory_utilization": spec.gpu_memory_utilization,
        "limit_mm_per_prompt": mm_limits(spec),
        "chat_template_kwargs": dict(spec.chat_template_kwargs),
        "extra_engine_kwargs": dict(extra_engine_kwargs),
        "requests": [
            {
                "key": r.key,
                "messages": r.messages(),
                "max_tokens": effective_max_tokens(spec, r),
                "ignore_eos": r.ignore_eos,
            }
            for r in todo
        ],
    }
    in_path = workdir / "baseline_in.json"
    out_path = workdir / "baseline_out.json"
    in_path.write_text(json.dumps(spec_json))
    runner = pathlib.Path(__file__).parent / "baseline_runner.py"
    proc = subprocess.run(
        [sys.executable, str(runner), str(in_path), str(out_path)],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Baseline runner failed (exit {proc.returncode}):\n"
            f"stdout tail: {proc.stdout[-2000:]}\nstderr tail: {proc.stderr[-2000:]}"
        )
    return json.loads(out_path.read_text())
