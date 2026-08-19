# SPDX-License-Identifier: Apache-2.0
"""Engine harness for the multimodal acceptance suite.

Runs one in-process vLLM engine with the LMCache connector (single-process
mode so the LMCache stats singleton is directly readable) and compares
outputs against a baseline computed by a plain vLLM engine in a subprocess.
"""

# Standard
from dataclasses import dataclass
import json
import os
import pathlib
import subprocess
import sys
import warnings

# First Party (test-local)
from catalog import MMRequest
from specs import ModelSpec

LMCACHE_TEST_CHUNK_SIZE = 16


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
    """
    # Keep scheduler+worker in this process so LMCStatsMonitor is shared.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["LMCACHE_CHUNK_SIZE"] = str(LMCACHE_TEST_CHUNK_SIZE)
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "40"
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
    # Third Party
    from prometheus_client import REGISTRY

    totals = {"lmcache:num_lookup_tokens": 0.0, "lmcache:num_lookup_hits": 0.0}
    for metric in REGISTRY.collect():
        if metric.name in totals:
            totals[metric.name] = sum(
                sample.value
                for sample in metric.samples
                if sample.name.endswith("_total")
            )
    return (
        int(totals["lmcache:num_lookup_tokens"]) + monitor.interval_lookup_tokens,
        int(totals["lmcache:num_lookup_hits"]) + monitor.interval_lookup_hits,
    )


class MMHarness:
    """Drives one model's acceptance run: baselines + LMCache engine + stats.

    Args:
        spec: The model under certification.
        baselines: Mapping of request key to the plain-vLLM output text.
    """

    def __init__(self, spec: ModelSpec, baselines: dict[str, str]):
        configure_environment()
        self.spec = spec
        self.baselines = baselines
        # Diagnostic recorder: capture the multimodal identifiers the
        # connector substitutes, so false-hit failures can name the request
        # pair involved. Installed before the engine imports the adapter.
        self._identifier_log: list[str] = []
        self._install_identifier_recorder()

        # Third Party
        from vllm import LLM
        from vllm.config import KVTransferConfig

        # First Party
        from lmcache.observability import LMCStatsMonitor

        self.llm = LLM(
            model=spec.hf_id,
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheConnectorV1", kv_role="kv_both"
            ),
            max_model_len=spec.max_model_len,
            gpu_memory_utilization=spec.gpu_memory_utilization,
            enforce_eager=True,
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 2},
        )
        self.monitor = LMCStatsMonitor.GetOrCreate()

    def _install_identifier_recorder(self) -> None:
        """Wrap the connector's placeholder substitution to log identifiers.

        Read-only observation: the wrapper calls through to the original
        function unchanged. Must run before vLLM imports the LMCache adapter
        (which binds the function by name at import time).
        """
        # First Party
        import lmcache.integration.vllm.utils as lmc_utils

        original = lmc_utils.apply_mm_hashes_to_token_ids
        log = self._identifier_log

        def recording(token_ids, mm_hashes, mm_positions):
            log.extend(mm_hashes)
            return original(token_ids, mm_hashes, mm_positions)

        lmc_utils.apply_mm_hashes_to_token_ids = recording

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
                temperature=0.0, max_tokens=request.max_tokens, seed=0
            ),
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

    def _cumulative_lookup_stats(self) -> tuple[int, int]:
        return cumulative_lookup_stats(self.monitor)

    def check_output(self, request: MMRequest, result: StepResult, where: str) -> None:
        """Verify a step's output against baseline and semantic probe.

        Policy: exact match against the plain-vLLM baseline is required. If
        the exact match fails but the semantic probe still passes, the step
        passes with a warning (GPU nondeterminism); if the probe also fails,
        this is cross-image contamination and the step fails hard.

        Args:
            request: The request that produced ``result``.
            result: The step outcome to verify.
            where: Human-readable context for failure messages.

        Raises:
            AssertionError: On baseline mismatch without probe rescue, or on
                probe failure.
        """
        probe_ok = self._probe_ok(request, result.text)
        if request.key in self.baselines:
            baseline = self.baselines[request.key]
            if result.text == baseline:
                return
            if probe_ok:
                warnings.warn(
                    f"[{where}] {request.key}: exact baseline mismatch but "
                    f"semantic probe passed (got {result.text!r}, "
                    f"baseline {baseline!r})",
                    stacklevel=2,
                )
                return
            raise AssertionError(
                f"[{where}] {request.key}: output diverged from baseline AND "
                f"semantic probe failed -- cross-image contamination. "
                f"got={result.text!r} baseline={baseline!r} "
                f"expected_probe={request.expected_probe}"
            )
        if not probe_ok:
            raise AssertionError(
                f"[{where}] {request.key}: semantic probe failed. "
                f"got={result.text!r} expected_probe={request.expected_probe}"
            )

    def _probe_ok(self, request: MMRequest, text: str) -> bool:
        """Whether the expected probe words appear in ``text`` IN ORDER.

        Order matters for multi-image probes: a swapped-order answer means
        the request hit the other ordering's cache entries.
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
    spec: ModelSpec, requests: list[MMRequest], workdir: pathlib.Path
) -> dict[str, str]:
    """Run all baseline-needing requests on a plain vLLM engine (subprocess).

    Args:
        spec: The model under certification.
        requests: Requests to run; only those with ``needs_baseline`` run.
        workdir: Directory for the input/output JSON files.

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
        "requests": [
            {"key": r.key, "messages": r.messages(), "max_tokens": r.max_tokens}
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
