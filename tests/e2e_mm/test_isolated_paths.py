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
from specs import MODEL_SPECS, HybridFamily, selected_model_keys


# Scenarios that pick their own deployment path. Each builds its harness
# through ``isolated_cases._deployment_harness``, which routes a hybrid to a
# real MP cache server because vLLM offers its hybrid KV cache manager only
# to connectors advertising SupportsHMA and LMCacheConnectorV1 does not --
# on the in-process path a hybrid does not run slower, it fails engine init.
# These therefore apply to every model.
ROUTED_SCENARIOS = ("capacity_eviction",)
MP_SCENARIOS = ("mp_connector",)

# Of the hybrids, only the SLIDING_WINDOW family gets the routed scenarios.
# This is a cost property, not a stand-in for "the ones we tried": a
# sliding-window hybrid's objects are ordinary paged KV a few hundred KB
# wide (338 KB on Gemma 3, 896 KB on Gemma 4), so many of them fit the
# scenario's deliberately tiny cap. A recurrent-state hybrid instead holds a
# whole state page per block -- ~205 MB on Qwen3.6-27B, which is more than
# the entire cap -- so it could not store a single object, and the eviction
# and conservation assertions would fail for a reason unrelated to what they
# test. Those models need their own measured capacity first.
ROUTED_HYBRID_FAMILIES = (HybridFamily.SLIDING_WINDOW,)

# Routed like the above, but a hybrid additionally needs a MEASURED GPU
# block pool. The pool has to sit in a narrow window -- above what one
# max-length request costs, below what the whole batch costs -- and that
# window is set by the model's KV bytes per token, so it cannot be derived
# from the spec. A hybrid without ``preemption_gpu_blocks`` would fail
# engine init rather than test anything, so it stays excluded until someone
# reads the numbers out of vLLM's own refusal message (it names the maximum
# model length the default pool buys, which gives bytes per block directly).
POOL_SIZED_SCENARIOS = ("preemption",)

# In-process only, and for a reason that is not plumbing. chunked_prefill
# pins max_num_batched_tokens to a value far below one prompt so that
# scheduler steps land inside an image span; a RECURRENT_STATE hybrid needs
# the opposite (a step wide enough for one whole block, so its state
# snapshot lands on a block boundary -- see HybridFamily), and its block is
# 544-784 tokens against that budget of 128. The two requirements are
# contradictory, so this stays off for hybrids and their certificates say so.
IN_PROCESS_SCENARIOS = ("chunked_prefill",)


def _scenario_cases() -> list[tuple[str, str]]:
    """Enumerate (scenario, model_key) pairs applicable to each model.

    Returns:
        The parametrization list: every selected model paired with the
        scenarios it can actually run.
    """
    cases: list[tuple[str, str]] = []
    for model_key in selected_model_keys():
        spec = MODEL_SPECS[model_key]
        scenarios = MP_SCENARIOS
        if not spec.hybrid_block_tokens:
            scenarios = (
                IN_PROCESS_SCENARIOS
                + ROUTED_SCENARIOS
                + POOL_SIZED_SCENARIOS
                + scenarios
            )
        elif spec.hybrid_family in ROUTED_HYBRID_FAMILIES:
            scenarios = ROUTED_SCENARIOS + scenarios
            if spec.preemption_gpu_blocks:
                scenarios = POOL_SIZED_SCENARIOS + scenarios
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
