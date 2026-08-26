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
from isolated_routing import isolated_scenarios
from specs import MODEL_SPECS, selected_model_keys

# A scenario brings up its own engine (two, counting the plain-vLLM
# baseline), so this bounds engine load plus the scenario itself. The
# slowest one measured across the nine archived suites is 249s (glm-4.6v-
# flash on the MP connector; both 27B hybrids sit at 220-228s), so 900s is
# 3.6x the worst case. It used to be 2400s, which meant a scenario that
# hangs -- the vLLM 0.27.1 preemption livelock did, see
# records/2026/08/26/3_ -- burned 40 minutes before turning red.
SCENARIO_TIMEOUT_S = 900
# The full child output goes to disk; this is only how much of it the
# assertion message repeats.
_LOG_TAIL_CHARS = 4000


def _tail(path: pathlib.Path) -> str:
    """Return the last ``_LOG_TAIL_CHARS`` characters of a log file.

    Args:
        path: The log file the scenario subprocess wrote.

    Returns:
        The tail, or a note naming the file when it cannot be read.
    """
    try:
        return path.read_text(errors="replace")[-_LOG_TAIL_CHARS:]
    except OSError as exc:
        return f"<unreadable {path}: {exc}>"


def _scenario_cases() -> list[tuple[str, str]]:
    """Enumerate (scenario, model_key) pairs applicable to each model.

    Which scenarios apply to a model, and why each exclusion is a property
    of the model rather than a list of the ones already tried, lives in
    ``isolated_routing`` -- shared with ``certify``, so what runs here and
    what the certificate claims cannot drift apart.

    Returns:
        The parametrization list: every selected model paired with the
        scenarios it can actually run.
    """
    cases: list[tuple[str, str]] = []
    for model_key in selected_model_keys():
        cases.extend(
            (scenario, model_key)
            for scenario in isolated_scenarios(MODEL_SPECS[model_key])
        )
    return cases


@pytest.mark.parametrize("scenario,model_key", _scenario_cases())
def test_isolated_scenario(scenario, model_key, tmp_path):
    configure_environment()
    out_json = tmp_path / f"{scenario}_{model_key}.json"
    runner = pathlib.Path(__file__).parent / "isolated_cases.py"
    # Streamed to files rather than captured: a scenario's stderr ends with
    # a dozen LMCache shutdown lines, so a truncated tail reliably showed
    # the shutdown noise and dropped the traceback that explained the
    # failure (measured on capacity_eviction[molmo2-4b],
    # records/2026/08/26/6_). On disk the whole thing survives, timeouts
    # included.
    stdout_log = tmp_path / f"{scenario}_{model_key}.stdout.log"
    stderr_log = tmp_path / f"{scenario}_{model_key}.stderr.log"
    with stdout_log.open("w") as out, stderr_log.open("w") as err:
        try:
            proc = subprocess.run(
                [sys.executable, str(runner), scenario, model_key, str(out_json)],
                stdout=out,
                stderr=err,
                timeout=SCENARIO_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise AssertionError(
                f"{scenario}[{model_key}] did not finish within "
                f"{SCENARIO_TIMEOUT_S}s -- treated as a hang. Full child "
                f"output: {stdout_log}, {stderr_log}\n"
                f"stderr tail: {_tail(stderr_log)}"
            ) from exc
    if not out_json.exists():
        raise AssertionError(
            f"{scenario}[{model_key}] crashed before reporting "
            f"(exit {proc.returncode}). Full child output: {stdout_log}, "
            f"{stderr_log}\nstdout tail: {_tail(stdout_log)}\n"
            f"stderr tail: {_tail(stderr_log)}"
        )
    report = json.loads(out_json.read_text())
    assert report["failures"] == [], (
        f"{scenario}[{model_key}] failed:\n"
        + "\n".join(report["failures"])
        + f"\nmetrics: {json.dumps(report['metrics'], indent=2)}"
        + f"\nchild output: {stdout_log}, {stderr_log}"
    )
    assert proc.returncode == 0, (
        f"{scenario}[{model_key}] exited {proc.returncode} despite an empty "
        f"failure list. Full child output: {stdout_log}, {stderr_log}\n"
        f"stderr tail: {_tail(stderr_log)}"
    )
