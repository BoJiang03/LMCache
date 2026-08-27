# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the parity gate's flip-split semantics (CPU-only).

The gate separates verdict-to-verdict answer flips (budgeted by
``MAX_FLIP_FRACTION`` in count and by ``MAX_FLIP_ASYMMETRY_P`` in
direction) from ``''``<->verdict parse flips (bounded through the per-pass
parse-ratio deltas), and reports score deltas without gating them. Each
case here is a scenario the full MME runs actually produced; see
records/2026/08/26/8_ and records/2026/08/26/10_ for the measurements the
thresholds encode.
"""

# First Party (test-local)
from benchmark_parity import (
    MAX_PARSE_RATIO_DELTA,
    FlipCounts,
    MMAUBenchmark,
    MMEBenchmark,
    count_flips,
    flip_asymmetry_p,
    parity_gate,
)


def _report(**overrides) -> dict:
    """A minimal green split-schema report; overrides patch fields in.

    The baseline is 2374 questions (the full MME set), a perfect hit
    ratio, no flips of either kind, and stable parse ratios.
    """
    report = {
        "benchmark": "mme",
        "num_questions": 2374,
        "scores": {
            "baseline": {"total": 1968.78},
            "pass1_miss": {"total": 1968.78},
            "pass2_hit": {"total": 1968.78},
        },
        "flips_pass1_vs_baseline": 0,
        "flips_pass2_vs_pass1": 0,
        "answer_flips_pass1_vs_baseline": 0,
        "parse_flips_pass1_vs_baseline": 0,
        "answer_regressions_pass1_vs_baseline": 0,
        "answer_improvements_pass1_vs_baseline": 0,
        "answer_lateral_pass1_vs_baseline": 0,
        "answer_flips_pass2_vs_pass1": 0,
        "parse_flips_pass2_vs_pass1": 0,
        "answer_regressions_pass2_vs_pass1": 0,
        "answer_improvements_pass2_vs_pass1": 0,
        "answer_lateral_pass2_vs_pass1": 0,
        "pass2_lookup_hit_ratio": 0.98,
        "cache_granularity_tokens": 16,
        "pass2_achievable_hit_tokens": 1266912,
        "pass2_external_cached_tokens": 1268288,
        "baseline_answer_parse_ratio": 1.0,
        "pass1_answer_parse_ratio": 1.0,
        "pass2_answer_parse_ratio": 1.0,
    }
    report.update(overrides)
    return report


def test_green_report_passes():
    gate = parity_gate(_report())
    assert gate["pass"] is True


def test_answer_flips_over_budget_fail():
    # The qwen2-vl-2b case: 18 deterministic verdict flips against a
    # budget of 0.005 * 2374 = 11.87 stays red under the split gate.
    gate = parity_gate(_report(flips_pass2_vs_pass1=18, answer_flips_pass2_vs_pass1=18))
    assert gate["pass"] is False
    assert gate["answer_flips_pass2_vs_pass1"] == 18


def test_parse_flips_alone_do_not_fail():
    # The gemma-4-e4b case: 14 abstain-margin flips in both directions,
    # 1 verdict flip, parse ratio essentially unmoved. Red before the
    # split (15 > 11.87), green after -- and reproducibly so, since both
    # of its full runs had 1 verdict flip.
    gate = parity_gate(
        _report(
            flips_pass2_vs_pass1=15,
            answer_flips_pass2_vs_pass1=1,
            parse_flips_pass2_vs_pass1=14,
            baseline_answer_parse_ratio=0.896,
            pass1_answer_parse_ratio=0.896,
            pass2_answer_parse_ratio=0.896,
        ),
        min_parse_ratio=0.85,
    )
    assert gate["pass"] is True
    assert gate["parse_ratio_deltas"]["pass2_vs_pass1"] == 0.0


def test_one_sided_parse_collapse_fails():
    # What a real hit-path defect does: the 2026-08-21 KEY_NOT_READABLE
    # regression truncated pass-2 answers, moving the parse ratio ~0.4 in
    # one direction. The delta bound catches it even with zero verdict
    # flips counted.
    gate = parity_gate(
        _report(
            flips_pass2_vs_pass1=900,
            answer_flips_pass2_vs_pass1=0,
            parse_flips_pass2_vs_pass1=900,
            pass2_answer_parse_ratio=0.62,
        )
    )
    assert gate["pass"] is False
    delta = gate["parse_ratio_deltas"]["pass2_vs_pass1"]
    assert delta > MAX_PARSE_RATIO_DELTA


def test_cold_pass_parse_collapse_fails():
    # Same bound on the pass1-vs-baseline side, where cold-pass cache
    # poisoning (issue #3301) would surface.
    gate = parity_gate(
        _report(
            flips_pass1_vs_baseline=900,
            answer_flips_pass1_vs_baseline=0,
            parse_flips_pass1_vs_baseline=900,
            pass1_answer_parse_ratio=0.62,
            pass2_answer_parse_ratio=0.62,
        )
    )
    assert gate["pass"] is False


def test_score_delta_reported_not_gated():
    # A large total movement with in-budget flips passes; the delta stays
    # visible in the gate dict for diagnosis. Score deltas left the gate
    # because MME quantizes single borderline questions at ~7.5 points:
    # one identical 18-flip core measured 9.00 / 2.25 / 9.75 across three
    # runs against the old 10.0 budget.
    report = _report(
        answer_flips_pass2_vs_pass1=5,
        flips_pass2_vs_pass1=5,
    )
    report["scores"]["pass2_hit"] = {"total": 1998.78}
    gate = parity_gate(report)
    assert gate["pass"] is True
    assert gate["score_delta_pass2_vs_pass1"] == 30.0
    assert "max_score_delta" not in gate["thresholds"]


def test_pre_split_report_gates_combined_flips():
    # A report recorded before the split has only the combined counts;
    # they keep gating as before (over-failing is acceptable, letting a
    # defect through is not).
    old = _report(flips_pass2_vs_pass1=15)
    for key in (
        "answer_flips_pass1_vs_baseline",
        "parse_flips_pass1_vs_baseline",
        "answer_regressions_pass1_vs_baseline",
        "answer_improvements_pass1_vs_baseline",
        "answer_lateral_pass1_vs_baseline",
        "answer_flips_pass2_vs_pass1",
        "parse_flips_pass2_vs_pass1",
        "answer_regressions_pass2_vs_pass1",
        "answer_improvements_pass2_vs_pass1",
        "answer_lateral_pass2_vs_pass1",
        "pass1_answer_parse_ratio",
        "pass2_answer_parse_ratio",
    ):
        del old[key]
    gate = parity_gate(old)
    assert gate["pass"] is False
    assert gate["answer_flips_pass2_vs_pass1"] == 15
    assert gate["parse_ratio_deltas"] == {}
    assert gate["flip_asymmetry_p"] == {}


def test_hit_ratio_floor_still_binds():
    gate = parity_gate(_report(pass2_lookup_hit_ratio=0.5))
    assert gate["pass"] is False


def test_count_flips_classifies_both_kinds():
    bench = MMEBenchmark()
    items = [{"qid": str(i), "answer": "yes"} for i in range(4)]
    pass_x = ["Yes", "Yes", "maybe", "No"]
    pass_y = ["No", "Yes", "No", ""]
    counts = count_flips(bench, items, pass_x, pass_y)
    # Item 0 flips no->yes against an answer key of yes, so the pass under
    # test is the one that got it right: an improvement, not a regression.
    assert counts == FlipCounts(
        answer_flips=1,
        parse_flips=2,
        regressions=0,
        improvements=1,
        lateral=0,
    )
    assert counts.total == 3


def test_count_flips_directions_split_by_answer_key():
    bench = MMEBenchmark()
    items = [
        {"qid": "0", "answer": "yes"},
        {"qid": "1", "answer": "no"},
    ]
    # Both items flip; the answer key decides which way each one counts.
    counts = count_flips(bench, items, ["No", "Yes"], ["Yes", "No"])
    assert counts.answer_flips == 2
    assert (counts.regressions, counts.improvements, counts.lateral) == (2, 0, 0)


def test_count_flips_lateral_needs_more_than_two_verdicts():
    # MMAU offers up to four options, so a flip can move between two wrong
    # ones -- direction-free, and left to the count budget alone.
    bench = MMAUBenchmark()
    items = [{"qid": "0", "choices": ["a", "b", "c"], "answer_letter": "C"}]
    counts = count_flips(bench, items, ["A"], ["B"])
    assert counts.answer_flips == 1
    assert (counts.regressions, counts.improvements, counts.lateral) == (0, 0, 1)


def test_flip_asymmetry_p_is_the_exact_binomial_tail():
    assert flip_asymmetry_p(0, 0) == 1.0
    assert flip_asymmetry_p(1, 1) == 0.75
    assert flip_asymmetry_p(19, 0) == 0.5**19
    # Symmetric input can never be evidence of skew.
    assert flip_asymmetry_p(50, 50) > 0.5


def test_one_sided_flips_fail_inside_the_count_budget():
    # 11 flips sit under the default budget of 0.005 * 2374 = 11.87, so the
    # count alone passes them. All 11 leaning the same way is what a
    # corrupting cache looks like (p = 0.5**11), and that fails.
    gate = parity_gate(
        _report(
            flips_pass2_vs_pass1=11,
            answer_flips_pass2_vs_pass1=11,
            answer_regressions_pass2_vs_pass1=11,
            answer_improvements_pass2_vs_pass1=0,
        )
    )
    assert gate["pass"] is False
    assert gate["answer_flips_pass2_vs_pass1"] <= gate["max_flips"]
    assert gate["flip_asymmetry_p"]["pass2_vs_pass1"] == 0.5**11


def test_balanced_flips_pass_a_widened_budget():
    # The qwen2-vl-2b case on vLLM 0.27.1: 19 flips against a 0.01 budget,
    # near-evenly split, which is the engine's batch-shape numerics rather
    # than a defect.
    gate = parity_gate(
        _report(
            flips_pass2_vs_pass1=19,
            answer_flips_pass2_vs_pass1=19,
            answer_regressions_pass2_vs_pass1=10,
            answer_improvements_pass2_vs_pass1=9,
        ),
        max_flip_fraction=0.01,
    )
    assert gate["pass"] is True


def test_asymmetry_calibration_at_the_widened_budget():
    # What the widened budget still catches: of 19 flips, 15 one way fails
    # and 14 passes. Pins the calibration the threshold comment claims.
    def gate_for(regressions: int) -> dict:
        return parity_gate(
            _report(
                flips_pass2_vs_pass1=19,
                answer_flips_pass2_vs_pass1=19,
                answer_regressions_pass2_vs_pass1=regressions,
                answer_improvements_pass2_vs_pass1=19 - regressions,
            ),
            max_flip_fraction=0.01,
        )

    assert gate_for(15)["pass"] is False
    assert gate_for(14)["pass"] is True


def test_baseline_comparison_is_gated_on_direction_too():
    # pass1 vs the no-LMCache baseline exercises the store path; a one-sided
    # lean there is a defect the cold pass wrote, not a hit-path artifact.
    gate = parity_gate(
        _report(
            flips_pass1_vs_baseline=8,
            answer_flips_pass1_vs_baseline=8,
            answer_regressions_pass1_vs_baseline=8,
            answer_improvements_pass1_vs_baseline=0,
        )
    )
    assert gate["pass"] is False
    assert gate["flip_asymmetry_p"]["pass1_vs_baseline"] == 0.5**8


def test_lateral_flips_do_not_move_the_asymmetry():
    # Direction-free flips stay out of the binomial; they are bounded by the
    # count budget, which 5 flips of 2374 does not reach.
    gate = parity_gate(
        _report(
            flips_pass2_vs_pass1=5,
            answer_flips_pass2_vs_pass1=5,
            answer_regressions_pass2_vs_pass1=0,
            answer_improvements_pass2_vs_pass1=0,
            answer_lateral_pass2_vs_pass1=5,
        )
    )
    assert gate["pass"] is True
    assert gate["flip_asymmetry_p"]["pass2_vs_pass1"] == 1.0
