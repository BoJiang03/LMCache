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
from specs import selected_model_keys


@pytest.mark.parametrize("model_key", selected_model_keys())
@pytest.mark.parametrize(
    "scenario",
    ["chunked_prefill", "capacity_eviction", "preemption", "mp_connector"],
)
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
