# SPDX-License-Identifier: Apache-2.0
"""Single-command support certification for one multimodal model.

Runs the full synthetic acceptance suite (T0/T1/T2 + isolated scenarios +
the detector negative control) and combines it with a benchmark-parity
result (T0.6 -- MME for image/video models, MMAU for audio) into one
machine-readable certificate. The certificate — not
any individual green test — is the artifact behind the claim "LMCache
supports model X": it records exactly what was verified, on which code, on
which deployment path, and what remains outside the claim.

Usage (from tests/e2e_mm, on a GPU machine):

    python certify.py qwen2.5-vl-3b                    # suite only -> PROVISIONAL
    python certify.py qwen2.5-vl-3b --run-parity       # suite + full MME
    python certify.py qwen2.5-vl-3b --parity-report mme_full.json  # reuse run

Exit codes: 0 = SUPPORTED, 2 = PROVISIONAL (suite green, parity not
provided), 1 = NOT_SUPPORTED.
"""

# Standard
from datetime import datetime, timezone
import argparse
import json
import os
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# First Party (test-local)
from benchmark_parity import parity_gate  # noqa: E402
from harness import LMCACHE_TEST_CHUNK_SIZE  # noqa: E402
from isolated_routing import (  # noqa: E402
    CAPACITY_EVICTION,
    CHUNKED_PREFILL,
    PREEMPTION,
    isolated_scenarios,
)
from specs import MODEL_SPECS, HybridFamily, ModelSpec  # noqa: E402

CERTIFICATE_SCHEMA_VERSION = 4

# What a SUPPORTED verdict never covers, whatever the model.
KNOWN_NOT_COVERED = [
    "tensor-parallel (TP>1) and pipeline-parallel deployments",
    "remote / disk storage backends and cross-instance sharing",
    "allocator-level buffer accounting (tracked by the pin-count project)",
]

# Audio stopped being a universal exclusion when the suite grew audio cases
# (T2.4) and a cross-modal pair (T2.5). It is now a statement about the
# MODEL, so it must not be emitted for a model certified WITH audio -- doing
# so made the first Qwen3-Omni certificate list audio in `scope.modalities`
# and disclaim it two blocks later, in the same document.
AUDIO_NOT_COVERED = (
    "audio modality: this model's spec does not declare `audio`, so none of "
    "the suite's audio or cross-modal cases ran for it and it is certified "
    "on its image/video paths only -- note that a checkpoint can carry an "
    "audio tower and still be certified this way (Gemma 4 does)"
)

# Exclusion for a model whose prompt is not append-only in media, so the
# T2.2 partial-sharing case has no prefix to measure.
MEDIA_PREFIX_NOT_COVERED = (
    "partial sharing across a growing media list (T2.2): this model's "
    "processor lays out the whole image SET rather than appending each "
    "item, so a one-image prompt is not a token prefix of the same-image-"
    "plus-one prompt (measured: they share ONE token). Reuse across "
    "requests with the SAME media list is covered as usual; what is not is "
    "reuse when the list grows"
)

# The suite drives ONE deployment. Stated on every certificate, because a
# reader who knows LMCache also ships an in-process connector would
# otherwise have to guess whether it was covered.
IN_PROCESS_NOT_COVERED = (
    "the in-process LMCacheConnectorV1 path: the suite drives the "
    "multi-process deployment only and no longer contains an in-process "
    "harness (removed 2026-08-26; git history and branch "
    "archive/e2e_mm-inprocess-and-mp carry it)"
)

# Exclusion for a model whose spec declares the deepstack add-on suite.
# That suite's oracle read stored KV back out of the in-process
# LocalCPUBackend and compared it against a pre-eviction copy; the MP cache
# server exposes no way to read a stored object back (its object listing
# covers L2 only, and its checksum API hashes GPU blocks, which a recompute
# never reproduces bit-exactly), so the suite cannot run at all. Every
# output-based oracle was MEASURED blind to this fault class -- disabling
# the injection entirely changed no output byte -- so there is nothing
# weaker to fall back on. See records/2026/08/26.
DEEPSTACK_NOT_COVERED = (
    "mid-image-span resume of the DeepStack side buffer (TD.1-TD.4): the "
    "only oracle sensitive to a lost or misaligned payload compares stored "
    "KV before and after the resume, which requires reading a stored object "
    "back -- the MP cache server has no such API, so the add-on suite was "
    "removed with the in-process path rather than replaced by a check "
    "measured to be blind"
)

# Additional exclusions for ANY multi-KV-group model.
HYBRID_NOT_COVERED = [
    "recovery from a failed KV load (the connector's degraded mode): vLLM "
    "rewinds the affected requests through "
    "`_update_requests_with_invalid_blocks`, which unpacks a single KV "
    "cache group and therefore raises on a hybrid -- so on this path a "
    "load error is fatal to the engine, not recoverable",
]

# Exclusion for any hybrid the preemption scenario is not run for -- two
# different reasons again, and the difference matters: one is a measurement
# nobody has taken, the other is a wall.
_PREEMPTION_NOT_COVERED = {
    # A sliding-window hybrid needs a pool that admits all six padded
    # prompts and still cannot hold their decode growth. That window is per
    # model -- it follows from the model's KV bytes per token AND from
    # whether its sliding window is wider than the prompt, since a window
    # narrower than the prompt makes the per-request footprint saturate and
    # the batch always fit. So it is measured, not derived, and for one
    # registered model (Gemma 4-E4B) a 2.25x sweep of pools found no value
    # that works. Either way the model declares no `preemption_gpu_blocks`
    # and its ModelSpec comment says which case it is.
    HybridFamily.SLIDING_WINDOW: [
        "preemption-driven recompute: this model declares no "
        "`preemption_gpu_blocks`, so the scenario is not run for it. The "
        "pool it needs -- large enough to admit the whole batch, too small "
        "to hold its decode growth -- is measured per model and may not "
        "exist at all when the model's sliding window is narrower than the "
        "prompt; see the model's ModelSpec comment for which applies",
    ],
    # A recurrent-state hybrid cannot run it at any pool size, and the
    # numbers below are the whole argument -- all on Qwen3.5-2B, 6 requests
    # of 3518 tokens, block 544, measured 2026-08-22.
    #
    # Above the crash region the scenario is vacuous. At the mandatory
    # minimum step budget (one block) vLLM never runs two of these requests
    # at once, so there is nothing to preempt at ANY pool: raising the
    # budget by 6 tokens is the single variable that turns 0 preemptions
    # into 1 on plain vLLM at 32 blocks. Even with that budget, the
    # connector's external prefix hits remove enough prefill work that 32
    # blocks no longer fills (0 preemptions), so pressure needs a smaller
    # pool.
    #
    # Below it the engine dies. With the MP connector attached, 24, 20 and
    # 16 blocks all abort in vLLM's block-pool bookkeeping
    # (`block_pool.cache_full_blocks: assert blk.block_hash is None`, from
    # the RUNNING branch at 24/20 and the WAITING branch at 16), while plain
    # vLLM at those exact pools completes cleanly. So the pool region that
    # would create pressure is the region that crashes, and the two do not
    # overlap.
    HybridFamily.RECURRENT_STATE: [
        "preemption-driven recompute, at any pool size: align mode's "
        "one-block step budget means two of these requests never run at "
        "once, so a pool large enough to survive has nothing to preempt "
        "(measured: 128, 48 and 32 blocks all yield 0 preemptions), while "
        "every pool small enough to create pressure aborts the engine with "
        "the connector attached -- 24, 20 and 16 blocks each hit "
        "`block_pool.cache_full_blocks: assert blk.block_hash is None`, "
        "which plain vLLM at the same pools does not. Not a missing "
        "measurement; the two regions do not overlap",
    ],
}

# Why the chunked-prefill scenario did not run, for a model it is excluded
# for. Several reasons can hold at once (Gemma 3 and Gemma 4 are hybrids AND
# mm-prefix-LMs), so this is a list of every applicable one rather than a
# single lookup: a certificate that named only the hybrid reason would read
# as if the other were absent.
_CHUNKED_PREFILL_PREFIX = (
    "chunked-prefill step boundaries falling inside an image span: "
)
_CHUNKED_PREFILL_BY_FAMILY = {
    HybridFamily.RECURRENT_STATE: (
        "that scenario pins the batched-token budget far below one prompt, "
        "while align mode needs the opposite -- a step wide enough for one "
        "whole 544-784 token block, so the state snapshot lands on a "
        "boundary. Contradictory by construction, not a plumbing gap"
    ),
    HybridFamily.SLIDING_WINDOW: (
        "this family's smaller blocks would in principle allow it (it needs "
        "no step-width guarantee at all), but it is untested here"
    ),
}
_CHUNKED_PREFILL_BIDIRECTIONAL = (
    "this model attends bidirectionally over its multimodal span "
    "(vLLM's is_mm_prefix_lm), so vLLM forces disable_chunked_mm_input and "
    "refuses to start when the batched-token budget is below the model's "
    "worst-case mm item -- a budget small enough to split an image span "
    "aborts engine init, and one large enough to start cannot split it. "
    "Contradictory by construction, like align mode"
)


def chunked_prefill_not_covered(spec: ModelSpec) -> list[str]:
    """Every reason the chunked-prefill scenario is excluded for ``spec``.

    Args:
        spec: The model under certification; only called when
            ``isolated_routing`` actually excludes the scenario.

    Returns:
        One entry per applicable reason. Empty would mean the scenario was
        excluded for a reason nobody wrote down, so callers must treat an
        empty result as a bug rather than as "no exclusion".
    """
    reasons: list[str] = []
    if spec.mm_bidirectional_attention:
        reasons.append(_CHUNKED_PREFILL_PREFIX + _CHUNKED_PREFILL_BIDIRECTIONAL)
    family = _CHUNKED_PREFILL_BY_FAMILY.get(spec.hybrid_family)
    if family:
        reasons.append(_CHUNKED_PREFILL_PREFIX + family)
    return reasons


# Additional exclusions specific to a recurrent-state (Mamba/GDN) hybrid.
RECURRENT_STATE_NOT_COVERED = [
    "bit-exact generation -- the GDN kernels have no batch-invariant mode, "
    "and a hit RESTORES a recurrent-state page rather than reproducing KV "
    "bit-for-bit, so output equality is gated by the parity benchmark's "
    "flip/score budget, not bytes",
    "genuinely concurrent execution of a submitted batch: align mode pins "
    "the step budget to one unified block, and vLLM schedules running "
    "requests first, so a single decoding request leaves too little budget "
    "for any other request's block-aligned prefill chunk (which vLLM then "
    "truncates to zero and skips). Measured on Qwen3.5-2B: a 6-request "
    "batch never had more than one request running, and adding 6 tokens to "
    "the budget is enough to change that. So the suite exercises concurrent "
    "SUBMISSION and the connector's batched store/lookup traffic, but not "
    "two of these requests occupying the GPU at the same time",
]

# How the certificate describes each hybrid family's chunk size.
_CHUNK_NOTE = {
    HybridFamily.RECURRENT_STATE: "vLLM unified block size (Mamba/GDN align mode)",
    HybridFamily.SLIDING_WINDOW: (
        "common multiple of the paged groups' block sizes (sliding-window "
        "hybrid; vLLM reports the smallest of them as cache_config.block_size)"
    ),
}

# Scheduling regimes the suite drives. The prefill and batch shapes follow
# from the model's family (align mode pins the step budget; the other two
# families leave it alone), while everything else is contributed by an
# isolated scenario -- so those entries are read from ``isolated_routing``
# rather than restated here. Restating them is exactly how Gemma 4's
# certificate came to omit two scenarios it had passed.
_PREFILL_REGIME = {
    HybridFamily.NONE: "single-step and chunked prefill",
    HybridFamily.RECURRENT_STATE: (
        "chunked prefill (inherent: a scheduler step advances one unified block)"
    ),
    HybridFamily.SLIDING_WINDOW: (
        "single-step prefill (prompts fit one scheduler step; no budget is pinned)"
    ),
}
_BATCH_REGIME = {
    HybridFamily.NONE: "concurrent batches",
    HybridFamily.RECURRENT_STATE: (
        "concurrent batch submission (vLLM executes it serially -- see "
        "known_not_covered)"
    ),
    HybridFamily.SLIDING_WINDOW: "concurrent batches",
}
_SCENARIO_REGIME = {
    CAPACITY_EVICTION: "capacity eviction",
    PREEMPTION: "preemption-driven recompute",
}
# What a single-group model drives when the chunked-prefill scenario is
# excluded for it anyway. The NONE entry above claims BOTH halves, which was
# true only while every single-group model ran the scenario.
_PREFILL_REGIME_UNCHUNKABLE = (
    "single-step prefill (chunked prefill is not available: vLLM refuses a "
    "batched-token budget below one multimodal item for this model)"
)


def _scheduling(spec: ModelSpec) -> list[str]:
    """Scheduling regimes a green run for ``spec`` actually exercised.

    The prefill entry is read against ``isolated_scenarios``, not against
    the hybrid family alone: a single-group model whose chunked-prefill
    scenario is excluded drives single-step prefill only, and claiming the
    family's usual "single-step and chunked" would assert a regime no test
    ran.

    Args:
        spec: The model under certification.

    Returns:
        The certificate's ``scope.scheduling`` list: the prefill and batch
        shapes this model actually drove, plus one entry per isolated
        scenario that adds a regime of its own and is applicable to it.
    """
    scenarios = isolated_scenarios(spec)
    prefill = _PREFILL_REGIME[spec.hybrid_family]
    if spec.hybrid_family is HybridFamily.NONE and CHUNKED_PREFILL not in scenarios:
        prefill = _PREFILL_REGIME_UNCHUNKABLE
    return [
        prefill,
        _BATCH_REGIME[spec.hybrid_family],
        *(text for name, text in _SCENARIO_REGIME.items() if name in scenarios),
    ]


def parity_benchmark_label(spec: ModelSpec, parity: dict) -> str:
    """Name the benchmark whose parity backs (or would back) the verdict.

    Read from the report that was actually used, falling back to the spec's
    declared benchmark and only then to the historical default. Hardcoding
    "MME" here is what made Qwen3-Omni's certificate claim MME parity while
    its own parity block recorded an MMAU run.

    Args:
        spec: The model under certification.
        parity: The certificate's ``parity`` block, whose ``report`` (when
            present) carries the benchmark name written by
            ``benchmark_parity.py``.

    Returns:
        The benchmark name, upper-cased for prose.
    """
    reported = (parity.get("report") or {}).get("benchmark")
    return str(reported or spec.parity_benchmark or "mme").upper()


def git_head(cwd: pathlib.Path) -> str:
    """Return the current HEAD commit, or ``"unknown"`` if git cannot say.

    Args:
        cwd: Directory to run git in.

    Returns:
        The full 40-character SHA, or ``"unknown"``.
    """
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True
    )
    return rev.stdout.strip() if rev.returncode == 0 else "unknown"


def git_dirty(cwd: pathlib.Path) -> bool:
    """Report whether the working tree has uncommitted changes.

    Args:
        cwd: Directory to run git in.

    Returns:
        True if ``git status --porcelain`` prints anything, False if it is
        clean or if git cannot be consulted.
    """
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True
    )
    return bool(status.stdout.strip()) if status.returncode == 0 else False


MP_PATH_FULL = "LMCacheMPConnector + MP cache server (single GPU, TP=1)"


def certified_scope(spec: ModelSpec) -> dict:
    """Describe exactly what a green run for ``spec`` covers.

    Every tier of the suite crosses the MP transport -- it is the only
    deployment the harness builds -- so the scope names one path, and the
    in-process connector appears in the exclusions instead.

    Args:
        spec: The model under certification.

    Returns:
        The certificate's ``scope`` block: deployment path, modalities,
        cache granularity, storage backend and scheduling regimes proven.
    """
    hybrid = bool(spec.hybrid_block_tokens)
    return {
        "deployment_paths": [MP_PATH_FULL],
        "modalities": sorted(spec.modalities),
        "chunk_size": spec.hybrid_block_tokens or LMCACHE_TEST_CHUNK_SIZE,
        "chunk_size_note": (
            _CHUNK_NOTE[spec.hybrid_family] if hybrid else "LMCache chunk size"
        ),
        "backend": (
            "MP cache server L1, separate object groups"
            if hybrid
            else "MP cache server L1"
        ),
        "scheduling": _scheduling(spec),
    }


def known_not_covered(spec: ModelSpec) -> list[str]:
    """List what a green run for ``spec`` leaves outside the claim.

    The scenario-shaped exclusions are keyed off ``isolated_scenarios``,
    the same predicate the pytest parametrization uses, so a scenario that
    starts (or stops) running for a model cannot leave a stale claim here.

    Args:
        spec: The model under certification.

    Returns:
        The universal exclusions plus the ones this model's spec implies.
    """
    scenarios = isolated_scenarios(spec)
    base = list(KNOWN_NOT_COVERED)
    base.append(IN_PROCESS_NOT_COVERED)
    if "deepstack" in spec.extra_suites:
        base.append(DEEPSTACK_NOT_COVERED)
    if "audio" not in spec.modalities:
        base.append(AUDIO_NOT_COVERED)
    # Checked for EVERY model, not just hybrids. Molmo 2 is the first
    # non-hybrid the scenario is excluded for, and the old shape -- an early
    # return for non-hybrids, with the chunked-prefill exclusion emitted
    # only after it -- would have silently dropped the exclusion from its
    # certificate: the omission of a true limit, which reads exactly like
    # the absence of one.
    if CHUNKED_PREFILL not in scenarios:
        base += chunked_prefill_not_covered(spec)
    if not spec.media_prefix_stable:
        base.append(MEDIA_PREFIX_NOT_COVERED)
    if not spec.hybrid_block_tokens:
        return base
    extra = list(HYBRID_NOT_COVERED)
    if spec.hybrid_family is HybridFamily.RECURRENT_STATE:
        extra += RECURRENT_STATE_NOT_COVERED
    if PREEMPTION not in scenarios:
        extra += _PREEMPTION_NOT_COVERED[spec.hybrid_family]
    return base + extra


def run_suite(model_key: str, pressure_n: int, workdir: pathlib.Path) -> dict:
    """Run the synthetic acceptance suite for one model via pytest.

    Args:
        model_key: Registered model key from ``specs.py``.
        pressure_n: Collision-pressure image count (T0.2).
        workdir: Directory for the junit XML.

    Returns:
        Summary dict: counts, duration, ``green`` verdict, and the pytest
        exit code. ``green`` requires at least one test run, zero failures,
        zero errors, and zero skips (a skipped suite must never certify).
    """
    junit = workdir / f"suite_{model_key}.xml"
    env = dict(os.environ)
    env.update(
        LMCACHE_MM_E2E="1",
        LMCACHE_MM_E2E_MODELS=model_key,
        LMCACHE_MM_E2E_PRESSURE_N=str(pressure_n),
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", ".", "-q", f"--junit-xml={junit}"],
        cwd=pathlib.Path(__file__).resolve().parent,
        env=env,
        timeout=7200,
    )
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    if junit.exists():
        for suite in ET.parse(junit).getroot().iter("testsuite"):
            counts["tests"] += int(suite.get("tests", 0))
            counts["failures"] += int(suite.get("failures", 0))
            counts["errors"] += int(suite.get("errors", 0))
            counts["skipped"] += int(suite.get("skipped", 0))
            counts["time"] += float(suite.get("time", 0.0))
    ran = counts["tests"] - counts["skipped"]
    green = (
        proc.returncode == 0
        and ran > 0
        and counts["failures"] == 0
        and counts["errors"] == 0
        and counts["skipped"] == 0
    )
    return {
        "green": green,
        "exit_code": proc.returncode,
        "pressure_n": pressure_n,
        **counts,
    }


def run_parity(model_key: str, limit: int, workdir: pathlib.Path) -> dict:
    """Run the MME benchmark-parity check (T0.6) for one model.

    Args:
        model_key: Registered model key from ``specs.py``.
        limit: Question limit passed through (0 = full benchmark).
        workdir: Directory for the parity report JSON.

    Returns:
        The parity report dict (including its ``gate``).
    """
    out = workdir / f"parity_{model_key}.json"
    script = pathlib.Path(__file__).resolve().parent / "benchmark_parity.py"
    spec = MODEL_SPECS[model_key]
    cmd = [
        sys.executable,
        str(script),
        "--model",
        spec.hf_id,
        "--limit",
        str(limit),
        "--out",
        str(out),
    ]
    if spec.parity_benchmark:
        cmd += ["--benchmark", spec.parity_benchmark]
    if spec.mm_encoder_attn_backend:
        cmd += ["--mm-encoder-attn-backend", spec.mm_encoder_attn_backend]
    if spec.chat_template_kwargs:
        cmd += ["--chat-template-kwargs", json.dumps(spec.chat_template_kwargs)]
    if spec.mme_mm_processor_kwargs:
        cmd += ["--mm-processor-kwargs", json.dumps(spec.mme_mm_processor_kwargs)]
    mme_tokens = spec.mme_max_tokens or spec.min_decode_tokens
    if mme_tokens > 8:
        cmd += ["--max-tokens", str(mme_tokens)]
    if spec.mme_max_flip_fraction:
        cmd += ["--max-flip-fraction", str(spec.mme_max_flip_fraction)]
    if spec.mme_min_parse_ratio:
        cmd += ["--min-parse-ratio", str(spec.mme_min_parse_ratio)]
    if spec.mme_max_local_cpu_gb:
        cmd += ["--max-local-cpu-gb", str(spec.mme_max_local_cpu_gb)]
    if spec.hybrid_block_tokens:
        cmd += ["--hybrid-block-tokens", str(spec.hybrid_block_tokens)]
    if spec.hf_overrides:
        cmd += ["--hf-overrides", json.dumps(spec.hf_overrides)]
    if spec.hybrid_family is not HybridFamily.NONE:
        cmd += ["--hybrid-family", spec.hybrid_family.value]
    if spec.trust_remote_code:
        cmd += ["--trust-remote-code"]
    subprocess.run(
        cmd,
        cwd=script.parent,
        timeout=6 * 3600,
    )
    if not out.exists():
        raise RuntimeError("parity run produced no report")
    return json.loads(out.read_text())


def load_parity_report(
    path: pathlib.Path,
    spec: ModelSpec,
    max_flip_fraction: float = 0.0,
    min_parse_ratio: float = 0.0,
) -> dict:
    """Load a previously recorded parity report and re-evaluate its gate.

    Args:
        path: Path to a report written by ``benchmark_parity.py``.
        spec: The model under certification; its id must match the report,
            and the report must have been produced on the multi-process
            deployment (a certificate must never cite another model's
            parity run, nor one measured on a path it does not claim).
        max_flip_fraction: Per-model flip-budget override
            (``ModelSpec.mme_max_flip_fraction``); 0 keeps the default.
        min_parse_ratio: Per-model parse-rate floor override
            (``ModelSpec.mme_min_parse_ratio``); 0 keeps the default.

    Returns:
        The report dict with a freshly evaluated ``gate``.

    Raises:
        ValueError: If the report is for a different model or was produced
            on a deployment path this model is not certified on.
    """
    report = json.loads(path.read_text())
    if report.get("model") != spec.hf_id:
        raise ValueError(
            f"parity report {path} is for {report.get('model')!r}, "
            f"certificate is for {spec.hf_id!r}"
        )
    # Reports recorded before the field existed are all in-process runs,
    # which this suite no longer certifies on.
    recorded_path = report.get("deployment_path", "in_process")
    if recorded_path != "mp":
        raise ValueError(
            f"parity report {path} was produced on the {recorded_path!r} "
            f"deployment path; certificates cover the multi-process path "
            f"only, so this report has to be rerun"
        )
    report["gate"] = parity_gate(report, max_flip_fraction, min_parse_ratio)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_key", choices=sorted(MODEL_SPECS))
    parser.add_argument("--pressure-n", type=int, default=64)
    parser.add_argument(
        "--run-parity",
        action="store_true",
        help="run the full MME parity check (hours; nightly/release grade)",
    )
    parser.add_argument(
        "--parity-report",
        default="",
        help="reuse a recorded benchmark_parity.py report instead of rerunning",
    )
    parser.add_argument("--parity-limit", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model_key]
    here = pathlib.Path(__file__).resolve().parent
    out_path = pathlib.Path(args.out or f"certificate_{args.model_key}.json")

    commit = git_head(here)
    dirty_at_start = git_dirty(here)

    suite = run_suite(args.model_key, args.pressure_n, out_path.parent.resolve())

    parity: dict = {"source": "not_run"}
    if args.parity_report:
        report = load_parity_report(
            pathlib.Path(args.parity_report),
            spec,
            spec.mme_max_flip_fraction,
            spec.mme_min_parse_ratio,
        )
        parity = {"source": f"recorded:{args.parity_report}", "report": report}
    elif args.run_parity:
        report = run_parity(
            args.model_key, args.parity_limit, out_path.parent.resolve()
        )
        parity = {"source": "fresh_run", "report": report}

    parity_ok = parity.get("report", {}).get("gate", {}).get("pass", False)
    if not suite["green"]:
        verdict, exit_code = "NOT_SUPPORTED", 1
    elif parity["source"] == "not_run":
        verdict, exit_code = "PROVISIONAL", 2
    elif parity_ok:
        verdict, exit_code = "SUPPORTED", 0
    else:
        verdict, exit_code = "NOT_SUPPORTED", 1

    bench = parity_benchmark_label(spec, parity)
    # Re-read HEAD now that the suite is done. A commit landing mid-run does
    # not corrupt the measurements, but it does mean `commit` no longer names
    # the tree that was tested -- so say whether it still does instead of
    # quietly recording the launch value as if it were verified.
    commit_at_finish = git_head(here)
    dirty_at_finish = git_dirty(here)
    tree_stable = (
        commit != "unknown"
        and commit == commit_at_finish
        and not dirty_at_start
        and not dirty_at_finish
    )

    certificate = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "verdict": verdict,
        "verdict_meaning": {
            "SUPPORTED": (f"synthetic suite + {bench} parity green on the paths below"),
            "PROVISIONAL": f"synthetic suite green; {bench} parity not yet recorded",
            "NOT_SUPPORTED": "at least one certification layer failed",
        }[verdict],
        "model_key": args.model_key,
        "hf_id": spec.hf_id,
        "commit": commit,
        "tested_tree": {
            "commit_at_start": commit,
            "commit_at_finish": commit_at_finish,
            "dirty_at_start": dirty_at_start,
            "dirty_at_finish": dirty_at_finish,
            "stable": tree_stable,
            "note": (
                "`commit` names the tree under test only when `stable` is "
                "true; otherwise HEAD moved or the tree carried uncommitted "
                "changes while the suite ran"
            ),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": certified_scope(spec),
        "suite": suite,
        "parity": parity,
        "known_not_covered": known_not_covered(spec),
    }
    out_path.write_text(json.dumps(certificate, indent=2))
    if not tree_stable:
        print(
            f"[certify] WARNING: the tree moved while the suite ran "
            f"(start={commit[:8]} finish={commit_at_finish[:8]} "
            f"dirty={dirty_at_start}/{dirty_at_finish}); `commit` does not "
            f"name the tested tree -- re-run on a quiet tree before "
            f"publishing this certificate",
            file=sys.stderr,
        )
    print(f"[certify] {args.model_key}: {verdict} -> {out_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
