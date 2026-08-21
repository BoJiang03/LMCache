# SPDX-License-Identifier: Apache-2.0
"""Engine harness for the multimodal acceptance suite.

Runs one in-process vLLM engine with the LMCache connector (single-process
mode so the LMCache stats singleton is directly readable) and compares
outputs against a baseline computed by a plain vLLM engine in a subprocess.
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
import warnings

# First Party (test-local)
from catalog import MMRequest
from specs import ModelSpec

LMCACHE_TEST_CHUNK_SIZE = 16


@dataclass(frozen=True)
class StorageSnapshot:
    """Point-in-time contents of the LMCache local CPU backend.

    Attributes:
        num_keys: Number of chunk keys currently resident in the hot cache.
        total_bytes: Sum of the logical sizes of all resident memory objects.
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


def configure_environment(max_local_cpu_gb: float = 40.0) -> None:
    """Set the env vars the engine runs require. Idempotent.

    Must be called before importing vllm or lmcache, and before launching
    the baseline subprocess (which inherits this environment).

    Args:
        max_local_cpu_gb: LMCache local CPU backend capacity in GB. The
            default is far above what any test stores, so eviction never
            interferes; the capacity-eviction scenario passes a tiny value
            to force it.
    """
    # Keep scheduler+worker in this process so LMCStatsMonitor is shared.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["LMCACHE_CHUNK_SIZE"] = str(LMCACHE_TEST_CHUNK_SIZE)
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(max_local_cpu_gb)
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


def cumulative_lookup_stats(monitor) -> tuple[int, int]:
    """Cumulative (lookup_tokens, lookup_hits) since engine start.

    LMCache's stats-logger thread periodically moves the monitor's interval
    counters into Prometheus counters. The sum of the Prometheus counter and
    the monitor's current (un-cleared) interval is invariant under that move,
    so deltas of this sum are immune to the logger's clearing. Callers must
    never clear the monitor themselves.

    Args:
        monitor: The process-local ``LMCStatsMonitor`` instance.

    Returns:
        Tuple of cumulative (lookup_tokens, lookup_hits).
    """
    totals = _prometheus_counter_totals(
        ("lmcache:num_lookup_tokens", "lmcache:num_lookup_hits")
    )
    return (
        totals["lmcache:num_lookup_tokens"] + monitor.interval_lookup_tokens,
        totals["lmcache:num_lookup_hits"] + monitor.interval_lookup_hits,
    )


def cumulative_stored_tokens(monitor) -> int:
    """Cumulative tokens LMCache was ASKED to store since engine start.

    Counted at store-request time (``on_store_request``), i.e. this is the
    engine's store-side intent, not proof of residency; compare against
    ``storage_snapshot`` for the resident truth. Theft-proof the same way as
    ``cumulative_lookup_stats``.

    Args:
        monitor: The process-local ``LMCStatsMonitor`` instance.

    Returns:
        Cumulative store-requested token count.
    """
    totals = _prometheus_counter_totals(("lmcache:num_stored_tokens",))
    return totals["lmcache:num_stored_tokens"] + monitor.interval_stored_tokens


def _local_cpu_backend():
    """The in-process engine's ``LocalCPUBackend`` instance.

    Returns:
        The active backend.

    Raises:
        RuntimeError: If the LMCache engine or its local CPU backend is not
            available in this process.
    """
    # First Party
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    if engine is None or engine.storage_manager is None:
        raise RuntimeError("LMCache engine/storage manager not initialized")
    backend = engine.storage_manager.storage_backends.get("LocalCPUBackend")
    if backend is None:
        raise RuntimeError("LocalCPUBackend not active; snapshot unavailable")
    return backend


def storage_snapshot() -> StorageSnapshot:
    """Snapshot what is ACTUALLY resident in the local CPU backend.

    This is the ground truth for storage-conservation checks: chunk keys and
    bytes physically held by the hot cache, independent of what any counter
    claims was stored. Requires the in-process engine (single-process mode).

    Returns:
        The current ``StorageSnapshot``.

    Raises:
        RuntimeError: If the LMCache engine or its local CPU backend is not
            available in this process.
    """
    backend = _local_cpu_backend()
    with backend.cpu_lock:
        num_keys = len(backend.hot_cache)
        total_bytes = sum(obj.get_size() for obj in backend.hot_cache.values())
    return StorageSnapshot(num_keys=num_keys, total_bytes=total_bytes)


def resident_chunk_keys() -> list:
    """All chunk keys resident in the local CPU backend, in STORE order.

    The hot cache is an insertion-ordered mapping and LMCache stores a
    request's chunks sequentially, so the difference between two snapshots
    of this list — taken around a single request run on an otherwise idle
    engine — is that request's chunk-key chain in prompt order. The
    surgical-eviction oracles (deepstack add-on suite) rely on this to cut
    a stored request at a chosen prompt depth.

    Returns:
        Resident ``CacheEngineKey`` objects, oldest first.
    """
    backend = _local_cpu_backend()
    return backend.get_keys()


def clone_resident_kv(keys: list) -> dict:
    """Clone the resident KV tensor of each given chunk key.

    Args:
        keys: ``CacheEngineKey`` objects to clone; each must currently be
            resident in the local CPU backend.

    Returns:
        Mapping of key to a detached deep copy of its KV tensor.

    Raises:
        RuntimeError: If a key is not resident or holds no tensor.
    """
    backend = _local_cpu_backend()
    clones: dict = {}
    with backend.cpu_lock:
        for key in keys:
            obj = backend.hot_cache.get(key)
            if obj is None or obj.tensor is None:
                raise RuntimeError(f"chunk {key} not resident; cannot clone")
            clones[key] = obj.tensor.detach().clone()
    return clones


def evict_resident_keys(keys: list) -> None:
    """Force-remove the given chunk keys from the local CPU backend.

    Test-only surgical eviction: replays of the owning request then hit
    only the surviving prefix and must RECOMPUTE (and re-store) the evicted
    tail — the mid-prompt resume path that natural LRU eviction only
    produces by accident.

    Args:
        keys: ``CacheEngineKey`` objects to remove (missing keys are an
            error: a cut that silently did not happen would turn the resume
            test into a trivial full-hit replay).

    Raises:
        RuntimeError: If a key was not resident.
    """
    backend = _local_cpu_backend()
    for key in keys:
        if not backend.remove(key):
            raise RuntimeError(f"chunk {key} was not resident; cut incomplete")


def resident_kv_tensor(key):
    """The resident KV tensor of one chunk key, or None if not resident.

    Args:
        key: The ``CacheEngineKey`` to look up.

    Returns:
        The live (not copied) KV tensor, or None when the key is absent.
    """
    backend = _local_cpu_backend()
    with backend.cpu_lock:
        obj = backend.hot_cache.get(key)
    return obj.tensor if obj is not None else None


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

    Args:
        spec: The model under certification.
        baselines: Mapping of request key to the plain-vLLM output text.
        extra_engine_kwargs: Additional/overriding vLLM ``LLM(...)`` kwargs;
            used by isolated scenarios (chunked prefill, capacity eviction)
            to reshape the engine while reusing all harness plumbing.
        max_local_cpu_gb: LMCache local CPU capacity (see
            ``configure_environment``).
    """

    def __init__(
        self,
        spec: ModelSpec,
        baselines: dict[str, str],
        extra_engine_kwargs: Mapping[str, object] = _NO_EXTRA_KWARGS,
        max_local_cpu_gb: float = 40.0,
    ):
        configure_environment(max_local_cpu_gb)
        self.spec = spec
        self.baselines = baselines
        # Diagnostic recorder: capture the multimodal identifiers the
        # connector substitutes, so false-hit failures can name the request
        # pair involved. Installed before the engine imports the adapter.
        self._identifier_log: list[str] = []
        self._identity_blind = False
        self._install_identifier_recorder()
        self._install_transport_hooks()

        # Third Party
        from vllm import LLM

        engine_kwargs: dict[str, object] = dict(
            model=spec.hf_id,
            kv_transfer_config=self._kv_transfer_config(),
            max_model_len=spec.max_model_len,
            gpu_memory_utilization=spec.gpu_memory_utilization,
            enforce_eager=True,
            enable_prefix_caching=False,
            limit_mm_per_prompt=mm_limits(spec),
        )
        engine_kwargs.update(extra_engine_kwargs)
        self.llm = LLM(**engine_kwargs)
        self._setup_stats()

    def _kv_transfer_config(self):
        """Build the KV transfer config selecting the deployment path."""
        # Third Party
        from vllm.config import KVTransferConfig

        return KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")

    def _install_transport_hooks(self) -> None:
        """Install path-specific counter hooks BEFORE the engine imports.

        The in-process path needs none (stats come from the LMCache
        singleton); the MP path wraps its adapter methods here.
        """

    def _setup_stats(self) -> None:
        """Bind the stats source once the engine is up."""
        # First Party
        from lmcache.observability import LMCStatsMonitor

        self.monitor = LMCStatsMonitor.GetOrCreate()

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
        """Tear down the engine and the LMCache engine instance."""
        # First Party
        from lmcache.integration.vllm.utils import ENGINE_NAME
        from lmcache.v1.cache_engine import LMCacheEngineBuilder

        del self.llm
        LMCacheEngineBuilder.destroy(ENGINE_NAME)

    def run(self, request: MMRequest) -> StepResult:
        """Send one request through the LMCache engine and read its stats.

        Per-request stats are computed as deltas of THEFT-PROOF cumulative
        counters (see ``_cumulative_lookup_stats``); LMCache's built-in
        stats-logger thread clears the monitor's interval counters every 10
        seconds, so reading the interval directly would randomly lose the
        window for ~2% of requests.
        """
        # Third Party
        from vllm import SamplingParams

        tokens_before, hits_before = self._cumulative_lookup_stats()
        log_before = len(self._identifier_log)
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
        return StepResult(
            text=outputs[0].outputs[0].text,
            lookup_tokens=tokens_after - tokens_before,
            lookup_hits=hits_after - hits_before,
            identifiers=tuple(dict.fromkeys(seen)),
        )

    def run_batch(self, requests: list[MMRequest]) -> BatchResult:
        """Submit all requests in ONE ``llm.chat`` call (concurrent batch).

        The vLLM scheduler interleaves the requests' prefills and decodes,
        so LMCache sees concurrent lookup/store traffic — including a store
        for one request racing the lookup of an identical one. Counters are
        aggregate only (per-request attribution is impossible in a batch).
        """
        # Third Party
        from vllm import SamplingParams

        tokens_before, hits_before = self._cumulative_lookup_stats()
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
        return BatchResult(
            texts=tuple(o.outputs[0].text for o in outputs),
            lookup_tokens=tokens_after - tokens_before,
            lookup_hits=hits_after - hits_before,
        )

    def _cumulative_lookup_stats(self) -> tuple[int, int]:
        return cumulative_lookup_stats(self.monitor)

    def stored_tokens_total(self) -> int:
        """Cumulative store-requested tokens (see ``cumulative_stored_tokens``)."""
        return cumulative_stored_tokens(self.monitor)

    def storage(self) -> StorageSnapshot:
        """Resident local-CPU-backend snapshot (see ``storage_snapshot``)."""
        return storage_snapshot()

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


class MPHarness(MMHarness):
    """Harness variant driving the multi-process deployment path (T3).

    The engine talks to an external LMCache MP cache server over ZMQ via
    ``LMCacheMPConnector`` (this repo's tip version, selected through
    ``kv_connector_module_path``); the server process must already be
    running. Stats plumbing differs from the in-process path:

    - lookup tokens/hits are recorded by wrapping the scheduler adapter's
      lookup submit/check methods (class-level wrappers: at most one
      MPHarness per process);
    - store intent is recorded by wrapping the worker adapter's batched
      store submission (works because the suite forces single-process vLLM);
    - residency comes from the server's HTTP ``/status`` endpoint.

    Args:
        spec: The model under certification.
        baselines: Mapping of request key to the plain-vLLM output text.
        zmq_port: The MP cache server's ZMQ port.
        http_port: The MP cache server's HTTP observability port.
        extra_engine_kwargs: Additional/overriding vLLM ``LLM(...)`` kwargs.
    """

    def __init__(
        self,
        spec: ModelSpec,
        baselines: dict[str, str],
        zmq_port: int,
        http_port: int,
        extra_engine_kwargs: Mapping[str, object] = _NO_EXTRA_KWARGS,
    ):
        self._zmq_port = zmq_port
        self._http_port = http_port
        self._lookup_tokens_total = 0
        self._lookup_hits_total = 0
        self._stored_tokens = 0
        self._pending_lookup_tokens: dict[str, int] = {}
        super().__init__(spec, baselines, extra_engine_kwargs=extra_engine_kwargs)

    def _kv_transfer_config(self):
        """Select THIS repo's MP connector via the module-path override."""
        # Third Party
        from vllm.config import KVTransferConfig

        return KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_connector_module_path=("lmcache.integration.vllm.lmcache_mp_connector"),
            kv_role="kv_both",
            kv_connector_extra_config={
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": self._zmq_port,
            },
        )

    def _install_transport_hooks(self) -> None:
        """Wrap the MP adapters to expose per-request lookup/store counters.

        The scheduler adapter resolves lookups asynchronously: tokens are
        remembered at submit time and booked (with the hit count) when
        ``check_lookup_result`` first returns a value for the request.
        """
        # First Party
        from lmcache.integration.vllm.vllm_multi_process_adapter import (
            LMCacheMPSchedulerAdapter,
            LMCacheMPWorkerAdapter,
        )

        harness = self
        original_submit = LMCacheMPSchedulerAdapter.maybe_submit_lookup_request
        original_check = LMCacheMPSchedulerAdapter.check_lookup_result
        original_store = LMCacheMPWorkerAdapter.batched_submit_store_requests

        def submit(adapter, request_id, token_ids, *args, **kwargs):
            harness._pending_lookup_tokens.setdefault(request_id, len(token_ids))
            return original_submit(adapter, request_id, token_ids, *args, **kwargs)

        def check(adapter, request_id):
            ret = original_check(adapter, request_id)
            if ret is not None:
                tokens = harness._pending_lookup_tokens.pop(request_id, None)
                if tokens is not None:
                    harness._lookup_tokens_total += tokens
                    harness._lookup_hits_total += ret
            return ret

        def store(adapter, request_ids, ops, *args, **kwargs):
            for op in ops:
                harness._stored_tokens += op.end - op.start
            return original_store(adapter, request_ids, ops, *args, **kwargs)

        LMCacheMPSchedulerAdapter.maybe_submit_lookup_request = submit
        LMCacheMPSchedulerAdapter.check_lookup_result = check
        LMCacheMPWorkerAdapter.batched_submit_store_requests = store

    def _setup_stats(self) -> None:
        self.monitor = None

    def _cumulative_lookup_stats(self) -> tuple[int, int]:
        return (self._lookup_tokens_total, self._lookup_hits_total)

    def stored_tokens_total(self) -> int:
        """Cumulative tokens submitted to the MP server for storage."""
        return self._stored_tokens

    def storage(self) -> StorageSnapshot:
        """Resident object count/bytes from the MP server's /status API."""
        # Standard
        import urllib.request

        url = f"http://localhost:{self._http_port}/status"
        with urllib.request.urlopen(url, timeout=30) as resp:
            status = json.loads(resp.read())
        l1 = status["storage_manager"]["l1_manager"]
        return StorageSnapshot(
            num_keys=l1["total_object_count"],
            total_bytes=l1["memory_used_bytes"],
        )

    def close(self) -> None:
        """Tear down the engine (the MP server is managed by the caller)."""
        del self.llm


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
