# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the parity run's preemption precondition (CPU-only).

A parity run measures the connector only while vLLM does not preempt: on a
preemption vLLM REPLACES the resumed request's block ids while the
connector only appends them, so the request keeps a freed block table and
its next store writes another request's KV under its own key. That defect
is upstream of this branch (see records/2026/08/27/7_ for the localisation
and the evidence chain), so the suite avoids the trigger -- it caps the
running batch far inside the block pool -- and asserts afterwards that the
trigger did not fire.

What these cases pin down is the DISTINCTION the precondition exists to
make: a preempting run is INVALID, not failing. The two lead to opposite
actions -- repeat the run with more headroom, versus fix a defect in this
branch -- so a preempting run must never reach a certificate as a verdict
on the model.
"""

# Standard
import pathlib

# Third Party
import pytest

# First Party (test-local)
import certify
from benchmark_parity import (
    MMEBenchmark,
    engine_kwargs,
    preemption_precondition,
)
from specs import MODEL_SPECS

_SHAPED_KEYS = ("gemma-3-4b", "internvl3.5-2b", "phi4-mm")


def _engine_kwargs(max_num_seqs: int, gpu_memory_utilization: float) -> dict:
    """``engine_kwargs`` for a plain image model, varying only the two knobs."""
    return engine_kwargs(
        "some/model",
        MMEBenchmark(),
        {},
        0,
        {},
        "",
        "",
        False,
        max_num_seqs,
        gpu_memory_utilization,
    )


def test_zero_preemptions_passes_and_stays_quiet():
    precondition = preemption_precondition(0, 0)
    assert precondition["pass"] is True
    assert precondition["measured"] is True
    assert precondition["total"] == 0
    # No `why` on a passing run: the text is an instruction for acting on a
    # failure, and carrying it unconditionally would make every report read
    # as though something were wrong with it.
    assert "why" not in precondition


def test_store_pass_preemption_alone_invalidates_the_run():
    # The store pass is the one that matters most: it poisons the entries
    # the hit pass then reads, which is why pass 1 can stay byte-identical
    # to the baseline while pass 2 corrupts.
    precondition = preemption_precondition(7, 0)
    assert precondition["pass"] is False
    assert precondition["total"] == 7
    assert precondition["pass1_preemptions"] == 7
    assert precondition["pass2_preemptions"] == 0
    assert "resumed_req_ids" in precondition["why"]


def test_per_pass_split_is_preserved_not_just_the_total():
    precondition = preemption_precondition(3, 5)
    assert precondition["total"] == 8
    assert (
        precondition["pass1_preemptions"],
        precondition["pass2_preemptions"],
    ) == (3, 5)


def test_invalid_report_is_refused_rather_than_certified():
    report = {"preemption": preemption_precondition(4, 0)}
    with pytest.raises(ValueError, match="INVALID, not failing"):
        certify.require_valid_parity_run(report, pathlib.Path("parity.json"))


def test_clean_report_is_accepted():
    report = {"preemption": preemption_precondition(0, 0)}
    certify.require_valid_parity_run(report, pathlib.Path("parity.json"))


def test_report_predating_the_check_is_accepted_as_older_evidence():
    # Every recorded report from before this check exists carries no
    # `preemption` block. Refusing those would invalidate certificates that
    # are merely older, so they pass -- and `preemption_check` is what keeps
    # that from reading as a measured zero.
    certify.require_valid_parity_run({}, pathlib.Path("parity.json"))


def test_preemption_check_distinguishes_measured_zero_from_unmeasured():
    measured = certify.preemption_check({"preemption": preemption_precondition(0, 0)})
    unmeasured = certify.preemption_check({})
    invalid = certify.preemption_check({"preemption": preemption_precondition(1, 2)})
    assert "zero preemptions measured" in measured
    assert "not measured" in unmeasured
    assert "INVALID" in invalid
    assert measured != unmeasured != invalid


def test_stats_stay_on_or_the_counter_reads_zero_forever():
    # The offline LLM API disables stat logging, and with it vLLM's
    # `num_preemptions` counter -- so a precondition built on it would pass
    # vacuously on every run. This is the one kwarg the whole check rests on.
    assert _engine_kwargs(0, 0.0)["disable_log_stats"] is False


def test_unset_knobs_leave_the_historical_geometry_untouched():
    # Recorded reports were all produced at 0.6 with vLLM's own batch size.
    # Routing these knobs through a default would silently re-shape them.
    kwargs = _engine_kwargs(0, 0.0)
    assert kwargs["gpu_memory_utilization"] == 0.6
    assert "max_num_seqs" not in kwargs


def test_set_knobs_reach_the_engine():
    kwargs = _engine_kwargs(64, 0.88)
    assert kwargs["max_num_seqs"] == 64
    assert kwargs["gpu_memory_utilization"] == 0.88


@pytest.mark.parametrize("model_key", _SHAPED_KEYS)
def test_shaped_models_pass_their_cap_to_the_runner(model_key):
    cmd = certify.parity_command(model_key, 0, pathlib.Path("out.json"))
    spec = MODEL_SPECS[model_key]
    assert "--max-num-seqs" in cmd
    assert cmd[cmd.index("--max-num-seqs") + 1] == str(spec.mme_max_num_seqs)


@pytest.mark.parametrize(
    "model_key", [k for k in sorted(MODEL_SPECS) if k not in _SHAPED_KEYS]
)
def test_unshaped_models_pass_no_cap(model_key):
    # A flag emitted for every model would change the geometry of every
    # recorded parity report at once; the shaping is per model, on purpose.
    cmd = certify.parity_command(model_key, 0, pathlib.Path("out.json"))
    assert "--max-num-seqs" not in cmd
    assert "--gpu-memory-utilization" not in cmd


@pytest.mark.parametrize("model_key", _SHAPED_KEYS)
def test_shaped_models_disclaim_the_pressure_regime(model_key):
    # Avoiding the trigger costs coverage, and a certificate that stays
    # silent about it would claim a regime the run deliberately did not
    # enter.
    exclusions = certify.known_not_covered(MODEL_SPECS[model_key])
    assert certify.PREEMPTION_AVOIDED_NOT_COVERED in exclusions


@pytest.mark.parametrize(
    "model_key", [k for k in sorted(MODEL_SPECS) if k not in _SHAPED_KEYS]
)
def test_unshaped_models_do_not_disclaim_it(model_key):
    exclusions = certify.known_not_covered(MODEL_SPECS[model_key])
    assert certify.PREEMPTION_AVOIDED_NOT_COVERED not in exclusions
