# SPDX-License-Identifier: Apache-2.0
"""Pytest wrappers for the isolated-engine scenarios (T0.9-T0.11, T3).

Each scenario needs an engine configured differently from the shared session
harness (or an external MP cache server), so it runs in a subprocess (see
``isolated_cases.py``); the test asserts the subprocess's JSON report
contains no failures.
"""

# Standard
import json
import pathlib
import subprocess
import sys

# Third Party
import pytest

# First Party (test-local)
from harness import configure_environment
from specs import MODEL_SPECS, selected_model_keys


# Scenarios that drive the in-process connector. vLLM only offers its
# hybrid KV cache manager to connectors that advertise support for it, and
# LMCacheConnectorV1 does not: a Mamba/GDN model fails engine init on this
# path ("Hybrid KV cache manager is disabled but failed to convert the KV
# cache specs to one unified type"), so these scenarios are not applicable
# to hybrid models and their certificates claim the MP path only.
IN_PROCESS_SCENARIOS = ("chunked_prefill", "capacity_eviction", "preemption")
MP_SCENARIOS = ("mp_connector",)


def _scenario_cases() -> list[tuple[str, str]]:
    """Enumerate (scenario, model_key) pairs applicable to each model.

    Returns:
        The parametrization list: every selected model paired with the
        scenarios its deployment path supports.
    """
    cases: list[tuple[str, str]] = []
    for model_key in selected_model_keys():
        spec = MODEL_SPECS[model_key]
        scenarios = MP_SCENARIOS
        if not spec.hybrid_block_tokens:
            scenarios = IN_PROCESS_SCENARIOS + MP_SCENARIOS
        cases.extend((scenario, model_key) for scenario in scenarios)
    return cases


@pytest.mark.parametrize("scenario,model_key", _scenario_cases())
def test_isolated_scenario(scenario, model_key, tmp_path):
    configure_environment()
    out_json = tmp_path / f"{scenario}_{model_key}.json"
    runner = pathlib.Path(__file__).parent / "isolated_cases.py"
    proc = subprocess.run(
        [sys.executable, str(runner), scenario, model_key, str(out_json)],
        capture_output=True,
        text=True,
        timeout=2400,
    )
    if not out_json.exists():
        raise AssertionError(
            f"{scenario}[{model_key}] crashed before reporting "
            f"(exit {proc.returncode}):\nstdout tail: {proc.stdout[-2000:]}\n"
            f"stderr tail: {proc.stderr[-2000:]}"
        )
    report = json.loads(out_json.read_text())
    assert report["failures"] == [], (
        f"{scenario}[{model_key}] failed:\n"
        + "\n".join(report["failures"])
        + f"\nmetrics: {json.dumps(report['metrics'], indent=2)}"
    )
    assert proc.returncode == 0, (
        f"{scenario}[{model_key}] exited {proc.returncode} despite an empty "
        f"failure list:\nstderr tail: {proc.stderr[-2000:]}"
    )
