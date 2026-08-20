# SPDX-License-Identifier: Apache-2.0
"""Single-command support certification for one multimodal model.

Runs the full synthetic acceptance suite (T0/T1/T2 + isolated scenarios +
the detector negative control) and combines it with an MME benchmark-parity
result (T0.6) into one machine-readable certificate. The certificate — not
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
from specs import MODEL_SPECS  # noqa: E402

CERTIFICATE_SCHEMA_VERSION = 2

# What a SUPPORTED verdict does NOT cover. Kept in the certificate so the
# claim is never wider than the evidence.
KNOWN_NOT_COVERED = [
    "tensor-parallel (TP>1) and pipeline-parallel deployments",
    "remote / disk storage backends and cross-instance sharing",
    "audio modality (no audio model registered yet)",
    "MP path chunk-boundary phases and collision pressure "
    "(T0.4/T0.2 run on the in-process path; keys are transport-independent)",
    "allocator-level buffer accounting (tracked by the pin-count project)",
]

# Deployment paths exercised by a green suite run (test_isolated_paths.py
# runs the mp_connector scenario for every selected model).
CERTIFIED_DEPLOYMENT_PATHS = [
    "LMCacheConnectorV1 (in-process, single GPU, TP=1)",
    "LMCacheMPConnector + MP cache server (single GPU, TP=1; "
    "T0/T1 core, see README T3)",
]


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
    if spec.chat_template_kwargs:
        cmd += ["--chat-template-kwargs", json.dumps(spec.chat_template_kwargs)]
    if spec.mme_mm_processor_kwargs:
        cmd += ["--mm-processor-kwargs", json.dumps(spec.mme_mm_processor_kwargs)]
    mme_tokens = spec.mme_max_tokens or spec.min_decode_tokens
    if mme_tokens > 8:
        cmd += ["--max-tokens", str(mme_tokens)]
    if spec.mme_max_flip_fraction:
        cmd += ["--max-flip-fraction", str(spec.mme_max_flip_fraction)]
    subprocess.run(
        cmd,
        cwd=script.parent,
        timeout=6 * 3600,
    )
    if not out.exists():
        raise RuntimeError("parity run produced no report")
    return json.loads(out.read_text())


def load_parity_report(
    path: pathlib.Path, hf_id: str, max_flip_fraction: float = 0.0
) -> dict:
    """Load a previously recorded parity report and re-evaluate its gate.

    Args:
        path: Path to a report written by ``benchmark_parity.py``.
        hf_id: Expected model id; a mismatch is refused (a certificate must
            never cite another model's parity run).
        max_flip_fraction: Per-model flip-budget override
            (``ModelSpec.mme_max_flip_fraction``); 0 keeps the default.

    Returns:
        The report dict with a freshly evaluated ``gate``.

    Raises:
        ValueError: If the report is for a different model.
    """
    report = json.loads(path.read_text())
    if report.get("model") != hf_id:
        raise ValueError(
            f"parity report {path} is for {report.get('model')!r}, "
            f"certificate is for {hf_id!r}"
        )
    report["gate"] = parity_gate(report, max_flip_fraction)
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

    commit = "unknown"
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True
    )
    if rev.returncode == 0:
        commit = rev.stdout.strip()

    suite = run_suite(args.model_key, args.pressure_n, out_path.parent.resolve())

    parity: dict = {"source": "not_run"}
    if args.parity_report:
        report = load_parity_report(
            pathlib.Path(args.parity_report),
            spec.hf_id,
            spec.mme_max_flip_fraction,
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

    certificate = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "verdict": verdict,
        "verdict_meaning": {
            "SUPPORTED": "synthetic suite + MME parity green on the paths below",
            "PROVISIONAL": "synthetic suite green; MME parity not yet recorded",
            "NOT_SUPPORTED": "at least one certification layer failed",
        }[verdict],
        "model_key": args.model_key,
        "hf_id": spec.hf_id,
        "commit": commit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "deployment_paths": CERTIFIED_DEPLOYMENT_PATHS,
            "modalities": sorted(spec.modalities),
            "chunk_size": 16,
            "backend": "LocalCPUBackend (in-process) / MP cache server L1",
            "scheduling": [
                "single-step and chunked prefill",
                "concurrent batches",
                "capacity eviction",
                "preemption-driven recompute",
            ],
        },
        "suite": suite,
        "parity": parity,
        "known_not_covered": KNOWN_NOT_COVERED,
    }
    out_path.write_text(json.dumps(certificate, indent=2))
    print(f"[certify] {args.model_key}: {verdict} -> {out_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
