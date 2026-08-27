#!/usr/bin/env python3
"""Characterise a parity run's flips, and compare flip sets across runs.

A red flip gate says how many answers moved, not why. This tells the two
candidate causes apart from the data already on disk:

* KV corruption damages an IMAGE's cached keys, and MME asks two questions
  per image off the same KV, so a corrupt image flips BOTH of its
  questions and the score moves one way. That was the gemma-4-e4b failure
  of 2026-08-21 (1288/2374 flips, -920 score).
* Numeric / batching nondeterminism flips scattered single questions in
  both directions, leaves the parse ratio perfect, and moves categories up
  as well as down.

Passing two runs compares their flip index SETS, which is the decisive
control: the same indices flipping twice means a deterministic difference
in the cached path; a different set each time means nondeterminism.

Usage:
    flipstat.py <run.json> [<run2.json> ...]

Each <run>.json is a benchmark_parity report; its sibling
<run>.answers.json and <run>.baseline.json are read alongside it.
"""

# Standard
import collections
import json
import pathlib
import re
import sys


def parse_answer(text: str) -> str:
    """Reduce a generated answer to 'yes', 'no', or '' (unparseable).

    Mirrors MMEBenchmark.parse_answer in benchmark_parity.py; kept here so
    this script needs no vLLM import to run.

    Args:
        text: The model's generated text.

    Returns:
        'yes', 'no', or '' when neither can be read off.
    """
    lowered = text.strip().lower()
    if "<|begin_of_box|>" in lowered:
        lowered = lowered.rsplit("<|begin_of_box|>", 1)[1]
        lowered = lowered.split("<|end_of_box|>", 1)[0].strip()
    elif "</think>" in lowered:
        tail = lowered.rsplit("</think>", 1)[1]
        matches = re.findall(r"\b(yes|no)\b", tail)
        return matches[-1] if matches else ""
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return ""


def load_run(report_path: pathlib.Path) -> dict:
    """Read one parity run's three answer sets and its gate.

    Args:
        report_path: Path to a ``benchmark_parity.py`` report json.

    Returns:
        Keys ``name``, ``gate``, ``report``, and ``passes`` (a dict of
        pass name to the parsed yes/no verdict per question).
    """
    report = json.loads(report_path.read_text())
    stem = report_path.with_suffix("")
    answers = json.loads(pathlib.Path(f"{stem}.answers.json").read_text())
    baseline = json.loads(pathlib.Path(f"{stem}.baseline.json").read_text())
    raw = {"baseline": baseline, "pass1": answers["pass1"], "pass2": answers["pass2"]}
    return {
        "name": report_path.name,
        "gate": report.get("gate", {}),
        "report": report,
        "passes": {k: [parse_answer(t) for t in v] for k, v in raw.items()},
    }


def describe(run: dict) -> set[int]:
    """Print one run's flip structure and return its pass1->pass2 flip set.

    Args:
        run: A ``load_run`` result.

    Returns:
        The question indices whose verdict changed between pass1 and pass2.
    """
    passes = run["passes"]
    total = len(passes["pass1"])
    print(f"\n=== {run['name']}  ({total} questions) ===")
    gate = run["gate"]
    print(
        f"  gate pass={gate.get('pass')} "
        f"max_flips={gate.get('max_flips')} "
        f"score_delta_pass2_vs_pass1={gate.get('score_delta_pass2_vs_pass1')} "
        f"hit_coverage={gate.get('pass2_hit_coverage')}"
    )
    scores = run["report"].get("scores", {})
    for key in ("baseline", "pass1_miss", "pass2_hit"):
        if key in scores:
            print(f"  score {key:11} total={scores[key].get('total')}")
    names = ["baseline", "pass1", "pass2"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = [
                k for k in range(total) if passes[names[i]][k] != passes[names[j]][k]
            ]
            print(f"  {names[i]:8} vs {names[j]:8}: {len(diff):4} flips")
    flips = [k for k in range(total) if passes["pass1"][k] != passes["pass2"][k]]
    if not flips:
        return set()
    direction = collections.Counter(
        (passes["pass1"][k], passes["pass2"][k]) for k in flips
    )
    slots = collections.Counter(k // 2 for k in flips)
    both = sum(1 for v in slots.values() if v == 2)
    print(f"  direction: {dict(direction)}")
    print(
        f"  images touched: {len(slots)}; images with BOTH questions flipped: "
        f"{both}  <- corruption signature is a high count here"
    )
    print(
        f"  unparseable per pass: {[sum(1 for x in passes[n] if not x) for n in names]}"
    )
    print(f"  indices: {flips}")
    return set(flips)


def main() -> int:
    """Describe each run given, then compare their flip sets pairwise.

    Returns:
        0 always; this is a reporting tool with no pass/fail of its own.
    """
    if len(sys.argv) < 2:
        print(__doc__)
        return 0
    sets = [
        (pathlib.Path(a).name, describe(load_run(pathlib.Path(a))))
        for a in sys.argv[1:]
    ]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            (na, sa), (nb, sb) = sets[i], sets[j]
            shared = sa & sb
            union = sa | sb
            print(f"\n=== flip-set overlap: {na} vs {nb} ===")
            print(f"  {len(sa)} and {len(sb)} flips; shared {len(shared)}; ")
            print(
                f"  jaccard {len(shared) / len(union):.3f}" if union else "  both empty"
            )
            print(f"  shared indices: {sorted(shared)}")
            print(
                "  reading: a high overlap means a DETERMINISTIC difference in the "
                "cached path; a low one means nondeterminism the gate is sampling."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
