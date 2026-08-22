# SPDX-License-Identifier: Apache-2.0
"""Benchmark score-parity check for LMCache multimodal support.

The synthetic acceptance suite (test_mm_acceptance.py) proves cache-key
isolation; this script proves the cache HIT path does not degrade real
model quality. It scores a benchmark three ways:

1. ``baseline``  -- plain vLLM, no LMCache (run in a subprocess);
2. ``pass1``     -- LMCache engine, cold cache (miss path, fills the cache);
3. ``pass2``     -- same engine, same questions again (hit path: the KV for
   every prompt is restored from LMCache instead of being computed).

and reports per-run benchmark scores plus per-item answer flips. Any KV
corruption on the hit path shows up directly as pass2-vs-pass1 flips and a
score drop.

Two benchmarks are available, selected with ``--benchmark``:

``mme``
    lmms-lab/MME, 2374 yes/no questions over images. The original target.
``mmau``
    TwinkStart/MMAU test-mini, 1000 multiple-choice questions over audio.
    Audio is a lower-token-density modality (~13 tok/s on Qwen3-Omni versus
    ~768 tokens for one capped MME image), so prompts are shorter; at the
    suite's 16-token chunk they are still comfortably cacheable.

Everything outside the benchmark classes is modality-agnostic: the three
passes, the counters, the hit-coverage arithmetic and the gate are shared,
so a new benchmark supplies only items, conversations, a parser and a
scorer.

Usage (from tests/e2e_mm):

    CUDA_VISIBLE_DEVICES=0 python benchmark_parity.py \
        [--benchmark mme] [--model Qwen/Qwen2.5-VL-3B-Instruct] \
        [--limit 0] [--out mme_parity_report.json]

``--limit 0`` (default) runs the full benchmark. Requires GPU, model
weights, and the benchmark's dataset (downloaded via HF).

Passing ``--hybrid-block-tokens N`` (a Mamba/GDN model's vLLM unified block
size) moves passes 2 and 3 onto the MP deployment path with a cache server
started here: vLLM offers its hybrid KV cache manager only to connectors
that advertise support for it, which the in-process connector does not.

Exit code 0 = parity holds (see THRESHOLDS below), 1 = parity violated.
"""

# Standard
import abc
import argparse
import base64
import glob
import io
import json
import os
import pathlib
import re
import subprocess
import sys

# Pin THIS repo's lmcache package (see tests/e2e_mm/conftest.py).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

PERCEPTION_CATEGORIES = [
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
]
COGNITION_CATEGORIES = [
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
]

MME_KEY = "mme"
MMAU_KEY = "mmau"

# Bound the visual token count so the full benchmark fits the LMCache CPU
# cache and the context window (Qwen smart-resize: tokens <= max_pixels/28^2).
MAX_PIXELS = 768 * 28 * 28

# Parity thresholds: batched GPU inference is not bit-deterministic, so a
# tiny number of borderline answer flips between passes is tolerated.
# pass1-vs-baseline is gated too: cross-image contamination (issue #3301)
# poisons the cache on the COLD pass and then replays deterministically, so
# pass2-vs-pass1 alone cannot see it.
#
# The score-delta budget is NOT here: it lives on each Benchmark, because
# the totals are on different scales (MME 2800, MMAU 100) and were
# calibrated against different answer regimes.
MAX_FLIP_FRACTION = 0.005  # per-item answer flips, both comparisons
MIN_HIT_RATIO = 0.8  # pass2 lookup hit ratio (else parity is vacuous)
# Fraction of what pass 1 stored that pass 2 must actually LOAD back.
# Replaces the raw hit-ratio floor when the cache granularity is coarse: a
# Mamba/GDN hybrid caches at vLLM's unified block size (e.g. 544 tokens),
# so an 800-token MME prompt can never exceed a 0.68 raw ratio however
# perfect the hit path is. Coverage is granularity-free, and counting
# loaded (not merely held) tokens also closes the vacuity hole the raw
# ratio has whenever vLLM prefix caching is on.
MIN_HIT_COVERAGE = 0.95
# Fraction of baseline answers that must parse to a verdict. If the model
# does not answer within the 8-token budget (e.g. a thinking model emitting
# a reasoning preamble), every answer parses to '' on all three passes and
# the flip/score comparisons pass VACUOUSLY while measuring nothing.
# Measured satisfiable on both benchmarks: 1.0 for Qwen3-Omni on a
# 45-question MMAU sample.
MIN_PARSE_RATIO = 0.9


class Benchmark(abc.ABC):
    """One question-answering benchmark the parity harness can run.

    Subclasses own everything modality- and dataset-specific. The three
    passes, the LMCache counters, the hit-coverage arithmetic and the gate
    are shared, so the contract here is deliberately narrow: produce items,
    turn them into conversations, parse one answer, and score a whole run.

    Attributes:
        key: ``--benchmark`` value, also recorded in the report so a stored
            report can be re-gated against the right score scale.
        modality: vLLM ``limit_mm_per_prompt`` key for the medium each
            question carries ("image", "audio").
        score_scale: Maximum value ``scores()["total"]`` can take. Recorded
            for readers; ``max_score_delta`` is what the gate uses.
        max_score_delta: Largest ``|total|`` difference between two passes
            that still counts as parity, on this benchmark's own scale.
    """

    key: str
    modality: str
    score_scale: float
    max_score_delta: float

    @abc.abstractmethod
    def load_items(self, limit: int) -> list[dict]:
        """Load benchmark questions.

        Args:
            limit: Maximum questions to load; 0 means the whole benchmark.

        Returns:
            Item dicts. Keys are benchmark-specific apart from ``qid``, but
            every item must carry whatever ``conversations``,
            ``parse_answer`` and ``scores`` need.
        """

    @abc.abstractmethod
    def conversations(self, items: list[dict]) -> list[list[dict]]:
        """Build one single-turn chat conversation per question, in order."""

    @abc.abstractmethod
    def parse_answer(self, text: str, item: dict) -> str:
        """Extract the comparable verdict from one generated answer.

        Returns '' when the answer names no valid option. That is not an
        error -- it is what the parse-ratio gate measures, since a run of
        unparseable answers would otherwise compare '' against '' and pass
        vacuously.

        Args:
            text: The model's generated text for this question.
            item: The item the answer belongs to, for benchmarks whose
                valid answers depend on the question (MMAU's option list).
        """

    @abc.abstractmethod
    def scores(self, items: list[dict], answers: list[str]) -> dict:
        """Score a full run.

        Returns:
            A dict carrying at least ``total`` (the number the parity gate
            compares); other keys are benchmark-specific breakdowns.
        """

    def default_mm_processor_kwargs(self) -> dict:
        """Processor kwargs to use when the model spec supplies none."""
        return {}


class MMEBenchmark(Benchmark):
    """lmms-lab/MME: 2374 yes/no questions over 1097 images."""

    key = MME_KEY
    modality = "image"
    # 14 categories x (acc*100 + acc+*100): Perception 2000 + Cognition 800.
    score_scale = 2800.0
    # 10 points of 2800 = 0.36%, calibrated on Qwen2.5-VL against the
    # engine's own batch-shape nondeterminism.
    max_score_delta = 10.0

    def default_mm_processor_kwargs(self) -> dict:
        """Qwen-style pixel cap; see MAX_PIXELS."""
        return {"max_pixels": MAX_PIXELS}

    def load_items(self, limit: int) -> list[dict]:
        """Load MME questions: [{qid, image_uri, question, answer, category}].

        Args:
            limit: Maximum questions; 0 loads all 2374.

        Returns:
            Item dicts; two consecutive items share a ``qid`` (and image),
            which is what MME's acc+ metric scores.
        """
        # Third Party
        from datasets import load_dataset

        ds = load_dataset("lmms-lab/MME", split="test")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        items = []
        uri_cache: dict[str, str] = {}
        for row in ds:
            qid = row["question_id"]
            if qid not in uri_cache:
                buf = io.BytesIO()
                row["image"].convert("RGB").save(buf, format="PNG")
                uri_cache[qid] = "data:image/png;base64," + base64.b64encode(
                    buf.getvalue()
                ).decode("ascii")
            items.append(
                {
                    "qid": qid,
                    "image_uri": uri_cache[qid],
                    "question": row["question"],
                    "answer": row["answer"].strip().lower(),
                    "category": row["category"],
                }
            )
        return items

    def conversations(self, items: list[dict]) -> list[list[dict]]:
        """Build one single-turn conversation per benchmark question."""
        return [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": item["image_uri"]}},
                        {"type": "text", "text": item["question"]},
                    ],
                }
            ]
            for item in items
        ]

    def parse_answer(self, text: str, item: dict) -> str:
        """Extract the yes/no verdict from a model answer ('' if neither).

        Three answer shapes are recognized:

        - A boxed answer (GLM-style ``<|begin_of_box|>yes<|end_of_box|>``,
          possibly after a short preamble): the box content is the verdict.
        - Completed thinking without a box (``...</think> The code is not
          Python. So the answer is no.``): the verdict is the LAST standalone
          yes/no in the post-thinking answer text; models restate the
          question before concluding, so the conclusion comes last.
        - Otherwise: the answer must START with yes/no (Qwen-style direct
          answers). Substring search is deliberately avoided, because MME
          questions quote statements containing yes/no-adjacent words, so a
          match deeper in free text is not a verdict.

        A truncated answer (open box or unfinished thinking) parses to '',
        which is what the parse-ratio gate exists to catch.

        Args:
            text: The model's generated text.
            item: Unused; MME's valid answers are yes/no for every question.
        """
        lowered = text.strip().lower()
        if "<|begin_of_box|>" in lowered:
            # Last begin marker: a preamble may open a spurious unclosed box.
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

    def scores(self, items: list[dict], answers: list[str]) -> dict:
        """Standard MME scoring: per-category acc*100 + acc+*100, summed.

        acc  = per-question accuracy;
        acc+ = fraction of images whose BOTH questions are answered correctly.
        Perception sums 10 categories (max 2000), Cognition 4 (max 800).
        """
        by_cat: dict[str, dict[str, list]] = {}
        for item, answer in zip(items, answers, strict=True):
            cat = by_cat.setdefault(item["category"], {"correct": [], "by_image": {}})
            ok = self.parse_answer(answer, item) == item["answer"]
            cat["correct"].append(ok)
            cat["by_image"].setdefault(item["qid"], []).append(ok)

        per_category = {}
        for name, cat in by_cat.items():
            acc = sum(cat["correct"]) / len(cat["correct"])
            plus_flags = [all(v) for v in cat["by_image"].values()]
            acc_plus = sum(plus_flags) / len(plus_flags)
            per_category[name] = round(acc * 100 + acc_plus * 100, 2)

        perception = round(
            sum(per_category.get(c, 0.0) for c in PERCEPTION_CATEGORIES), 2
        )
        cognition = round(
            sum(per_category.get(c, 0.0) for c in COGNITION_CATEGORIES), 2
        )
        return {
            "perception": perception,
            "cognition": cognition,
            "total": round(perception + cognition, 2),
            "per_category": per_category,
        }


class MMAUBenchmark(Benchmark):
    """TwinkStart/MMAU test-mini: 1000 multiple-choice questions over audio.

    Audio is the point: it exercises a modality whose placeholder expansion,
    processor and encoder are entirely separate from the image path, and
    whose contribution to the LMCache cache key has never been certified.

    The dataset ships audio as WAV bytes inside the parquet shards, so no
    second fetch is needed, and rows carry a ``task`` (sound/speech/music)
    that accuracy is reported over: a measured 0.47-1.00 accuracy spread
    across those three on Qwen3-Omni means an aggregate-only score would
    average away a regression confined to one of them.
    """

    key = MMAU_KEY
    modality = "audio"
    # Mean per-task accuracy, as a percentage.
    score_scale = 100.0
    # Calibrated on Qwen3-Omni-30B over the full 1000 questions: baseline,
    # pass1 and pass2 returned byte-identical scores (66.90; music 70.06 /
    # sound 71.47 / speech 59.16) for a measured delta of 0.00 and 0 flips
    # on BOTH comparisons -- including baseline-vs-pass1, which crosses
    # processes and engine configs and is where MME's flip budget came
    # from. So the observed nondeterminism floor here is zero, and this
    # budget is headroom above it rather than a fitted value: one flipped
    # answer of 1000 moves the total by ~0.1, so 1.0 tolerates ~10
    # same-direction flips, keeping the flip gate the one that binds first
    # as it is for MME. MME's 0.36%-of-scale ratio was deliberately NOT
    # inherited: it was calibrated on a yes/no model, and four-way choice
    # over an audio encoder is a different numeric regime. One run bounds
    # the floor, it does not prove it is always zero.
    max_score_delta = 1.0

    # 970 rows offer four options, 20 offer five and 10 offer two. Five
    # letters cover every row, so none is dropped for its option count.
    LETTERS = "ABCDE"
    _SHARD_GLOBS = (
        "/raid/data/hub/datasets--TwinkStart--MMAU/snapshots/*/data/"
        "test_mini-*.parquet",
        "~/.cache/huggingface/hub/datasets--TwinkStart--MMAU/snapshots/*/"
        "data/test_mini-*.parquet",
    )
    _META_COLUMNS = ["id", "question", "choices", "answer", "task", "difficulty"]

    def _shards(self) -> list[str]:
        """Parquet shard paths, from whichever HF cache holds the dataset.

        Raises:
            RuntimeError: If no shard matches any known location.
        """
        for pattern in self._SHARD_GLOBS:
            found = sorted(glob.glob(os.path.expanduser(pattern)))
            if found:
                return found
        raise RuntimeError(
            "MMAU test-mini shards not found; expected a TwinkStart--MMAU "
            f"dataset directory in one of: {self._SHARD_GLOBS}"
        )

    def load_items(self, limit: int) -> list[dict]:
        """Read MMAU test-mini rows, audio inlined as a data URI.

        A truncated read in shard order would sample ONE task: the split
        holds 333 sound / 333 speech / 334 music, but stores them in long
        same-task runs (96, 333, 301, 48, 33, 189), so the first 40 rows
        are all 'sound'. That was measured, not assumed; a 40-item smoke
        read this way reported ``accuracy_by_task`` with a single key.

        Rows are therefore round-robined across tasks in shard order: still
        fully deterministic and unshuffled, but a prefix of any length
        covers all three tasks in near-equal proportion. Audio is read in a
        second pass, for the selected rows only, so a short run does not
        decode the whole 2.84 GB of WAV bytes.

        Args:
            limit: Maximum questions; 0 loads all 1000.

        Returns:
            Item dicts with qid, question, choices, answer_letter,
            answer_text, task, difficulty and audio_uri.
        """
        # Third Party
        import pyarrow.parquet as pq

        by_task: dict[str, list[tuple[str, int, dict]]] = {}
        for shard in self._shards():
            cols = pq.read_table(shard, columns=self._META_COLUMNS).to_pydict()
            for i in range(len(cols["id"])):
                choices = list(cols["choices"][i])
                answer = cols["answer"][i]
                if answer not in choices or not 2 <= len(choices) <= len(self.LETTERS):
                    # Genuinely malformed: no ground truth to score against.
                    continue
                task = cols["task"][i]
                by_task.setdefault(task, []).append(
                    (
                        shard,
                        i,
                        {
                            "qid": cols["id"][i],
                            "question": cols["question"][i],
                            "choices": choices,
                            "answer_letter": self.LETTERS[choices.index(answer)],
                            "answer_text": answer,
                            "task": task,
                            "difficulty": cols["difficulty"][i],
                        },
                    )
                )

        # Tasks in a fixed alphabetical order, so the selection never
        # depends on dict insertion order.
        queues = [by_task[task] for task in sorted(by_task)]
        selected: list[tuple[str, int, dict]] = []
        for rank in range(max((len(q) for q in queues), default=0)):
            for queue in queues:
                if rank < len(queue):
                    selected.append(queue[rank])
        if limit > 0:
            selected = selected[:limit]

        wanted: dict[str, set[int]] = {}
        for shard, i, _ in selected:
            wanted.setdefault(shard, set()).add(i)
        audio: dict[tuple[str, int], bytes] = {}
        for shard, rows in wanted.items():
            column = pq.read_table(shard, columns=["audio"]).to_pydict()["audio"]
            for i in rows:
                audio[(shard, i)] = column[i]["bytes"]

        items = []
        for shard, i, meta in selected:
            uri = base64.b64encode(audio[(shard, i)]).decode("ascii")
            items.append({**meta, "audio_uri": "data:audio/wav;base64," + uri})
        return items

    def _prompt_text(self, item: dict) -> str:
        """Render the question, its lettered options, and the instruction."""
        lines = [item["question"], ""]
        lines += [f"{self.LETTERS[i]}. {c}" for i, c in enumerate(item["choices"])]
        lines += ["", "Answer with the letter of the correct option only."]
        return "\n".join(lines)

    def conversations(self, items: list[dict]) -> list[list[dict]]:
        """Build one single-turn conversation per benchmark question."""
        return [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": item["audio_uri"]},
                        },
                        {"type": "text", "text": self._prompt_text(item)},
                    ],
                }
            ]
            for item in items
        ]

    def parse_answer(self, text: str, item: dict) -> str:
        """The chosen option letter, '' if the answer names none.

        Two shapes are accepted, in order: a leading letter (``B``, ``B.``,
        ``(B)``, ``Answer: B``), which is what the instruction asks for,
        and failing that an exact choice TEXT anywhere in the answer, since
        a model that ignores the instruction usually restates the option
        verbatim. Substring letter search is avoided on purpose, because
        'A' appears inside ordinary words and would manufacture verdicts
        out of prose.

        A letter past the end of THIS item's option list does not count:
        accepting 'C' on a two-choice question would score a hallucinated
        option as a real answer.

        Args:
            text: The model's generated text.
            item: The item, for its ``choices`` list.
        """
        stripped = text.strip()
        pattern = (
            rf"^\W*(?:answer\s*[:\-]?\s*)?\(?([A-{self.LETTERS[-1]}])\)?"
            r"(?:[.,:)\s]|$)"
        )
        match = re.match(pattern, stripped, re.I)
        choices = item["choices"]
        if match:
            letter = match.group(1).upper()
            return letter if self.LETTERS.index(letter) < len(choices) else ""
        lowered = stripped.lower()
        hits = [
            self.LETTERS[i]
            for i, choice in enumerate(choices)
            if choice.strip().lower() in lowered
        ]
        return hits[0] if len(hits) == 1 else ""

    def scores(self, items: list[dict], answers: list[str]) -> dict:
        """Per-task accuracy percentages, plus their mean as ``total``.

        The mean is over TASKS rather than over questions so the three
        weigh equally. The split is near-balanced (333/333/334) so the two
        agree closely, but a ``--limit`` prefix need not be exactly
        balanced, and a per-task mean stays interpretable when it is not.
        """
        by_task: dict[str, list[bool]] = {}
        for item, answer in zip(items, answers, strict=True):
            ok = self.parse_answer(answer, item) == item["answer_letter"]
            by_task.setdefault(item["task"], []).append(ok)
        per_task = {
            name: round(100.0 * sum(flags) / len(flags), 2)
            for name, flags in sorted(by_task.items())
        }
        overall = sum(per_task.values()) / len(per_task) if per_task else 0.0
        return {
            "total": round(overall, 2),
            "per_task": per_task,
            "n_by_task": {name: len(v) for name, v in sorted(by_task.items())},
        }


BENCHMARKS: dict[str, Benchmark] = {
    MME_KEY: MMEBenchmark(),
    MMAU_KEY: MMAUBenchmark(),
}


def achievable_hit_tokens(prompt_lengths: list[int], granularity: int) -> int:
    """Tokens a perfect cache could return for these prompts.

    LMCache stores whole chunks and never serves a prompt's final token,
    so a fully cached prompt of ``t`` tokens can return at most
    ``granularity * ((t - 1) // granularity)``. Summing that is the only
    fair denominator for a hit-coverage gate: a store-token total is
    dedup-sensitive (identical prefixes submitted together are stored once
    and hit once per request), and the raw prompt-token total is capped by
    granularity (at a 544-token unified block a 380-token prompt is not
    cacheable at all).

    Args:
        prompt_lengths: Lookup token count of each request, one entry per
            request.
        granularity: Cache chunk size in tokens.

    Returns:
        The maximum hit tokens this workload admits; 0 for no requests.
    """
    return sum(granularity * ((t - 1) // granularity) for t in prompt_lengths)


def parity_gate(
    report: dict, max_flip_fraction: float = 0.0, min_parse_ratio: float = 0.0
) -> dict:
    """Evaluate the parity thresholds against a report dict.

    Shared by this script's exit code and by ``certify.py`` when it ingests
    a previously recorded parity report.

    Args:
        report: A report as written by ``main`` (needs ``scores``,
            ``num_questions``, both flip counts and the hit ratio).
        max_flip_fraction: Per-model override of the flip budget
            (``ModelSpec.mme_max_flip_fraction``); 0 keeps the default
            ``MAX_FLIP_FRACTION``.
        min_parse_ratio: Per-model override of the baseline parse-rate
            floor (``ModelSpec.mme_min_parse_ratio``); 0 keeps
            ``MIN_PARSE_RATIO``. Only for models that abstain rather than
            truncate; see that field's docstring.

    Returns:
        Dict with ``pass`` (bool), the evaluated deltas, the flip budget,
        the hit criterion that applied, and the thresholds used.
        ``pass2_hit_coverage`` is None when the run provides no denominator
        for it (an in-process run has no per-request lookup lengths); that
        is "not measured", not a coverage of zero, and it fails a coverage
        gate rather than satisfying one.
    """
    # First Party (test-local)
    from harness import LMCACHE_TEST_CHUNK_SIZE

    scores = report["scores"]
    flip_fraction = max_flip_fraction or MAX_FLIP_FRACTION
    parse_floor = min_parse_ratio or MIN_PARSE_RATIO
    # Score scales differ per benchmark (MME's total runs to 2800, MMAU's to
    # 100), so the delta budget travels with the benchmark. Reports written
    # before this field existed are all MME, whose budget is unchanged.
    benchmark = BENCHMARKS[report.get("benchmark", MME_KEY)]
    max_score_delta = benchmark.max_score_delta
    max_flips = flip_fraction * report["num_questions"]
    delta_p2_p1 = abs(scores["pass2_hit"]["total"] - scores["pass1_miss"]["total"])
    delta_p1_base = abs(scores["pass1_miss"]["total"] - scores["baseline"]["total"])
    hit_ratio = report["pass2_lookup_hit_ratio"]
    # Reports recorded before the parse-ratio guard existed lack the field;
    # their high absolute MME scores already prove the answers parsed.
    parse_ratio = report.get("baseline_answer_parse_ratio", 1.0)
    # Coarse-granularity runs (hybrids) are gated on coverage; fine-grained
    # ones keep the raw floor, and reports predating the coverage fields
    # (granularity absent) keep it too.
    granularity = report.get("cache_granularity_tokens", LMCACHE_TEST_CHUNK_SIZE)
    achievable = report.get("pass2_achievable_hit_tokens", 0)
    # Tokens vLLM actually skipped on the connector's account, not what the
    # cache merely held: with vLLM prefix caching on (mandatory for
    # hybrids) a replay served out of GPU memory still reports a full
    # LMCache hit, so only this number proves the retrieve path ran.
    loaded = report.get("pass2_external_cached_tokens", 0)
    # None, not 0.0, when the denominator is missing: the per-request token
    # list comes from MPTransportCounters, which is installed only on the MP
    # path, so an in-process run has no denominator at all. Publishing 0.0
    # there reads as "the cache achieved nothing" for a run whose raw hit
    # ratio was 1.0 -- an unmeasured quantity must not look like a measured
    # zero.
    coverage = round(loaded / achievable, 4) if achievable else None
    if granularity > LMCACHE_TEST_CHUNK_SIZE:
        hit_criterion = "coverage"
        # An unmeasurable coverage cannot satisfy a coverage gate.
        hit_ok = coverage is not None and coverage >= MIN_HIT_COVERAGE
    else:
        hit_criterion = "raw_hit_ratio"
        hit_ok = hit_ratio >= MIN_HIT_RATIO
    ok = (
        report["flips_pass2_vs_pass1"] <= max_flips
        and report["flips_pass1_vs_baseline"] <= max_flips
        and delta_p2_p1 <= max_score_delta
        and delta_p1_base <= max_score_delta
        and hit_ok
        and parse_ratio >= parse_floor
    )
    return {
        "pass": ok,
        "max_flips": max_flips,
        "score_delta_pass2_vs_pass1": delta_p2_p1,
        "score_delta_pass1_vs_baseline": delta_p1_base,
        "baseline_answer_parse_ratio": parse_ratio,
        "hit_criterion": hit_criterion,
        "cache_granularity_tokens": granularity,
        "pass2_hit_coverage": coverage,
        "thresholds": {
            "max_flip_fraction": flip_fraction,
            "max_score_delta": max_score_delta,
            "min_hit_ratio": MIN_HIT_RATIO,
            "min_hit_coverage": MIN_HIT_COVERAGE,
            "min_parse_ratio": parse_floor,
        },
    }


def run_batch(
    llm,
    benchmark: Benchmark,
    items: list[dict],
    chat_template_kwargs: dict,
    max_tokens: int,
) -> list[str]:
    """Run all questions in one batched chat call, order-aligned.

    Args:
        llm: The vLLM engine.
        benchmark: The benchmark, which renders the conversations.
        items: Question dicts (see ``Benchmark.load_items``).
        chat_template_kwargs: Extra chat-template kwargs from the model spec
            (e.g. ``{"enable_thinking": False}``); empty dict for none.
        max_tokens: Decode budget per question; must be large enough for the
            model's verdict to land in the generated text (see
            ``ModelSpec.min_decode_tokens``).
    """
    # Third Party
    from vllm import SamplingParams

    outputs = llm.chat(
        benchmark.conversations(items),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0),
        chat_template_kwargs=dict(chat_template_kwargs) or None,
        use_tqdm=True,
    )
    return [out.outputs[0].text for out in outputs]


def engine_kwargs(
    model: str,
    benchmark: Benchmark,
    mm_processor_kwargs: dict,
    hybrid_block_tokens: int,
    hf_overrides: dict,
    hybrid_family: str,
    mm_encoder_attn_backend: str,
) -> dict:
    """Engine kwargs for both parity engines.

    Args:
        model: HuggingFace model id.
        benchmark: The benchmark, which supplies the modality to admit and
            the fallback processor kwargs.
        mm_processor_kwargs: Model-specific token cap from the spec
            (``ModelSpec.mme_mm_processor_kwargs``); empty falls back to
            the benchmark's own default.
        hybrid_block_tokens: LMCache chunk size for a multi-KV-group model
            (``ModelSpec.hybrid_block_tokens``), 0 for every other model.
            Non-zero adds the mandatory hybrid settings -- to BOTH engines,
            since ``align`` mode changes the numeric regime and a mismatched
            baseline would misattribute the difference to LMCache.
        hf_overrides: Config repairs from ``ModelSpec.hf_overrides``, also
            applied to BOTH engines: they change the model's geometry, so a
            baseline built without them is not comparable.
        hybrid_family: ``ModelSpec.hybrid_family`` value, which decides
            whether the align settings apply ('' = recurrent state, the
            historical assumption).
        mm_encoder_attn_backend: vLLM multimodal-encoder attention backend
            ('' = vLLM's own choice). Set for a model whose default encoder
            backend is broken on the pinned vLLM: Qwen3-Omni on 0.23.0
            passes its vision tower's ``cu_seqlens`` to the attention
            kernel without moving it to the device (fixed upstream in
            0.27.1 at ``qwen3_omni_moe_thinker.py:982``), which aborts
            profiling even for an audio-only run, and "TORCH_SDPA" avoids
            the affected kernel. Applied to BOTH engines: it changes how
            the encoder computes, so a baseline built without it is not
            comparable.
    """
    kwargs = dict(
        model=model,
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        enable_prefix_caching=False,
        limit_mm_per_prompt={benchmark.modality: 1},
        mm_processor_kwargs=(
            dict(mm_processor_kwargs) or benchmark.default_mm_processor_kwargs()
        ),
    )
    if mm_encoder_attn_backend:
        kwargs["mm_encoder_attn_backend"] = mm_encoder_attn_backend
    # First Party (test-local)
    from harness import hybrid_engine_kwargs
    from specs import HybridFamily

    family = (
        HybridFamily(hybrid_family) if hybrid_family else HybridFamily.RECURRENT_STATE
    )
    kwargs.update(hybrid_engine_kwargs(hybrid_block_tokens, family))
    if hf_overrides:
        kwargs["hf_overrides"] = dict(hf_overrides)
    return kwargs


def run_baseline(
    model: str,
    benchmark: Benchmark,
    limit: int,
    out_path: str,
    chat_template_kwargs: dict,
    mm_processor_kwargs: dict,
    max_tokens: int,
    hybrid_block_tokens: int,
    hf_overrides: dict,
    hybrid_family: str,
    mm_encoder_attn_backend: str,
) -> None:
    """Subprocess role: plain vLLM answers for every question."""
    # Third Party
    from vllm import LLM

    items = benchmark.load_items(limit)
    llm = LLM(
        **engine_kwargs(
            model,
            benchmark,
            mm_processor_kwargs,
            hybrid_block_tokens,
            hf_overrides,
            hybrid_family,
            mm_encoder_attn_backend,
        )
    )
    answers = run_batch(llm, benchmark, items, chat_template_kwargs, max_tokens)
    with open(out_path, "w") as f:
        json.dump(answers, f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default=MME_KEY,
        choices=sorted(BENCHMARKS),
        help="which benchmark to score; 'mme' is images, 'mmau' is audio",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--limit", type=int, default=0, help="0 = full benchmark")
    parser.add_argument("--out", default="mme_parity_report.json")
    parser.add_argument("--role", choices=["main", "baseline"], default="main")
    parser.add_argument("--baseline-out", default="")
    parser.add_argument(
        "--chat-template-kwargs",
        default="",
        help="JSON object of extra chat-template kwargs from the model spec "
        "(e.g. '{\"enable_thinking\": false}'); empty = none",
    )
    parser.add_argument(
        "--mm-processor-kwargs",
        default="",
        help="JSON object with the model-specific per-image token cap "
        "(ModelSpec.mme_mm_processor_kwargs); empty = Qwen-style max_pixels",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="decode budget per question; raise for models whose answer "
        "lands after a preamble (ModelSpec.min_decode_tokens)",
    )
    parser.add_argument(
        "--max-flip-fraction",
        type=float,
        default=0.0,
        help="per-model flip-budget override for the gate "
        "(ModelSpec.mme_max_flip_fraction); 0 = the default",
    )
    parser.add_argument(
        "--max-local-cpu-gb",
        type=float,
        default=0.0,
        help="LMCache local-CPU capacity override "
        "(ModelSpec.mme_max_local_cpu_gb); 0 = the 40 GB default. Must "
        "hold the full benchmark's KV or the pass-2 LRU scan evicts every "
        "entry before its revisit and the hit-ratio gate fails at ~0",
    )
    parser.add_argument(
        "--hybrid-block-tokens",
        type=int,
        default=0,
        help="vLLM unified block size of a Mamba/GDN hybrid model "
        "(ModelSpec.hybrid_block_tokens); 0 = not a hybrid. Non-zero moves "
        "the LMCache pass onto the MP deployment path -- the only one vLLM "
        "offers its hybrid KV cache manager to -- with a cache server "
        "started here at that block size",
    )
    parser.add_argument(
        "--min-parse-ratio",
        type=float,
        default=0.0,
        help="Baseline parse-rate floor override "
        "(ModelSpec.mme_min_parse_ratio); 0 = the 0.9 default. Only for a "
        "model that ABSTAINS rather than truncates",
    )
    parser.add_argument(
        "--hf-overrides",
        default="",
        help="JSON object of vLLM hf_overrides from the model spec "
        "(ModelSpec.hf_overrides), applied to BOTH parity engines; empty = "
        "none. Gemma 4 needs it to see its full-attention head dims",
    )
    parser.add_argument(
        "--hybrid-family",
        default="",
        choices=["", "recurrent_state", "sliding_window"],
        help="ModelSpec.hybrid_family value, which decides whether the "
        "mamba align settings apply; empty = recurrent_state (the only "
        "family that existed before sliding-window hybrids)",
    )
    parser.add_argument(
        "--mm-encoder-attn-backend",
        default="",
        help="vLLM multimodal-encoder attention backend, applied to BOTH "
        "engines; empty = vLLM's own choice. Qwen3-Omni needs TORCH_SDPA on "
        "vLLM 0.23.0, whose vision tower otherwise aborts profiling even "
        "for an audio-only run (see engine_kwargs)",
    )
    args = parser.parse_args()
    benchmark = BENCHMARKS[args.benchmark]
    chat_template_kwargs: dict = (
        json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else {}
    )
    mm_processor_kwargs: dict = (
        json.loads(args.mm_processor_kwargs) if args.mm_processor_kwargs else {}
    )

    # First Party (test-local)
    from harness import (
        LMCACHE_TEST_CHUNK_SIZE,
        configure_environment,
        VllmPrefillCounters,
        cumulative_lookup_stats,
        cumulative_stored_tokens,
        reset_vllm_prefix_cache,
    )

    hf_overrides: dict = json.loads(args.hf_overrides) if args.hf_overrides else {}

    configure_environment(args.max_local_cpu_gb or 40.0)

    if args.role == "baseline":
        run_baseline(
            args.model,
            benchmark,
            args.limit,
            args.baseline_out,
            chat_template_kwargs,
            mm_processor_kwargs,
            args.max_tokens,
            args.hybrid_block_tokens,
            hf_overrides,
            args.hybrid_family,
            args.mm_encoder_attn_backend,
        )
        return 0

    items = benchmark.load_items(args.limit)
    print(f"[parity] {len(items)} {benchmark.key.upper()} questions loaded")

    baseline_path = pathlib.Path(args.out).with_suffix(".baseline.json")
    proc = subprocess.run(
        [
            sys.executable,
            __file__,
            "--role",
            "baseline",
            "--benchmark",
            args.benchmark,
            "--model",
            args.model,
            "--limit",
            str(args.limit),
            "--baseline-out",
            str(baseline_path),
            "--chat-template-kwargs",
            args.chat_template_kwargs,
            "--mm-processor-kwargs",
            args.mm_processor_kwargs,
            "--max-tokens",
            str(args.max_tokens),
            "--hybrid-block-tokens",
            str(args.hybrid_block_tokens),
            "--hf-overrides",
            args.hf_overrides,
            "--hybrid-family",
            args.hybrid_family,
            "--mm-encoder-attn-backend",
            args.mm_encoder_attn_backend,
        ],
        timeout=7200,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"baseline subprocess failed with {proc.returncode}")
    answers_base = json.loads(baseline_path.read_text())

    # Third Party
    from vllm import LLM
    from vllm.config import KVTransferConfig

    # A hybrid model can only be served over the MP transport, so the
    # LMCache pass needs its own cache server (chunked at the unified block
    # size, one object group per KV cache group) and reads its lookup
    # counters off the MP adapters instead of the in-process monitor.
    server = None
    counters = None
    monitor = None
    prefill = VllmPrefillCounters()
    prefill.install()
    if args.hybrid_block_tokens:
        # First Party (test-local)
        from harness import (
            MPTransportCounters,
            mp_kv_transfer_config,
            start_mp_cache_server,
        )

        server = start_mp_cache_server(
            zmq_port=26000 + (os.getpid() % 1000),
            http_port=27000 + (os.getpid() % 1000),
            chunk_size=args.hybrid_block_tokens,
            log_path=pathlib.Path(args.out).with_suffix(".mp_server.log"),
            l1_size_gb=args.max_local_cpu_gb or 40.0,
            separate_object_groups=True,
        )
        kv_transfer_config = mp_kv_transfer_config(server.zmq_port)
        counters = MPTransportCounters()
        counters.install()
    else:
        kv_transfer_config = KVTransferConfig(
            kv_connector="LMCacheConnectorV1", kv_role="kv_both"
        )

    llm = LLM(
        kv_transfer_config=kv_transfer_config,
        **engine_kwargs(
            args.model,
            benchmark,
            mm_processor_kwargs,
            args.hybrid_block_tokens,
            hf_overrides,
            args.hybrid_family,
            args.mm_encoder_attn_backend,
        ),
    )
    if counters is None:
        # First Party
        from lmcache.observability import LMCStatsMonitor

        monitor = LMCStatsMonitor.GetOrCreate()

    def lookup_stats() -> tuple[int, int]:
        """Cumulative (lookup_tokens, lookup_hits) on whichever transport."""
        if counters is not None:
            return (counters.lookup_tokens, counters.lookup_hits)
        return cumulative_lookup_stats(monitor)

    def stored_tokens() -> int:
        """Cumulative store-requested tokens on whichever transport."""
        if counters is not None:
            return counters.stored_tokens
        return cumulative_stored_tokens(monitor)

    def reset_local_prefix_cache() -> None:
        """Drop vLLM's own prefix cache so LMCache is the only hit source.

        Only hybrid models run with vLLM prefix caching enabled (``align``
        mode requires it); see ``harness.reset_vllm_prefix_cache``.
        """
        if args.hybrid_block_tokens:
            reset_vllm_prefix_cache(llm)

    # A server outlives a crashed run otherwise, holding the engine's IPC-
    # wrapped KV memory (tens of GB of GPU) until killed by hand.
    try:
        reset_local_prefix_cache()
        stored_before = stored_tokens()
        answers_p1 = run_batch(
            llm, benchmark, items, chat_template_kwargs, args.max_tokens
        )
        stored_p1 = stored_tokens() - stored_before
        reset_local_prefix_cache()
        t0, h0 = lookup_stats()
        local0, external0 = prefill.local_cached, prefill.external_cached
        lookups0 = len(counters.lookup_request_tokens) if counters else 0
        answers_p2 = run_batch(
            llm, benchmark, items, chat_template_kwargs, args.max_tokens
        )
        t1, h1 = lookup_stats()
        local_p2 = prefill.local_cached - local0
        external_p2 = prefill.external_cached - external0
        # Only the MP counters expose per-request lookup lengths, so an
        # in-process run cannot form the coverage denominator. Report that
        # as absent rather than as a zero it could be mistaken for.
        achievable_p2 = (
            achievable_hit_tokens(
                counters.lookup_request_tokens[lookups0:],
                args.hybrid_block_tokens or LMCACHE_TEST_CHUNK_SIZE,
            )
            if counters
            else None
        )
        hit_ratio = (h1 - h0) / max(1, t1 - t0)
    finally:
        if server is not None:
            server.process.terminate()

    scores = {
        "baseline": benchmark.scores(items, answers_base),
        "pass1_miss": benchmark.scores(items, answers_p1),
        "pass2_hit": benchmark.scores(items, answers_p2),
    }
    flips_p1_base = sum(
        benchmark.parse_answer(a, item) != benchmark.parse_answer(b, item)
        for a, b, item in zip(answers_p1, answers_base, items, strict=True)
    )
    flips_p2_p1 = sum(
        benchmark.parse_answer(a, item) != benchmark.parse_answer(b, item)
        for a, b, item in zip(answers_p2, answers_p1, items, strict=True)
    )
    report = {
        "model": args.model,
        "benchmark": benchmark.key,
        "deployment_path": "mp" if args.hybrid_block_tokens else "in_process",
        "num_questions": len(items),
        "scores": scores,
        "flips_pass1_vs_baseline": flips_p1_base,
        "flips_pass2_vs_pass1": flips_p2_p1,
        "pass2_lookup_hit_ratio": round(hit_ratio, 4),
        "cache_granularity_tokens": (
            args.hybrid_block_tokens or LMCACHE_TEST_CHUNK_SIZE
        ),
        # Store-request total, for diagnosis only: it is dedup-sensitive
        # (MME asks two questions per image, so the shared prefix is stored
        # once and hit twice) and cannot serve as a coverage denominator.
        "pass1_stored_tokens": stored_p1,
        "pass2_lookup_hit_tokens": h1 - h0,
        "pass2_lookup_tokens": t1 - t0,
        "pass2_achievable_hit_tokens": achievable_p2,
        # vLLM's own split of who served pass 2's prefill tokens. The
        # LMCache counter above says what the cache held; this says what
        # was loaded from it, and is what the coverage gate uses.
        "pass2_external_cached_tokens": external_p2,
        "pass2_local_cached_tokens": local_p2,
        "baseline_answer_parse_ratio": round(
            sum(
                1
                for a, item in zip(answers_base, items, strict=True)
                if benchmark.parse_answer(a, item)
            )
            / max(1, len(items)),
            4,
        ),
    }
    gate = parity_gate(report, args.max_flip_fraction, args.min_parse_ratio)
    report["gate"] = gate
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    # Raw per-pass answers, so a flip overshoot can be analyzed post-hoc
    # (which questions, which categories, which direction) without paying
    # for a multi-hour rerun. The baseline answers are already on disk.
    answers_path = pathlib.Path(args.out).with_suffix(".answers.json")
    with open(answers_path, "w") as f:
        json.dump({"pass1": answers_p1, "pass2": answers_p2}, f)

    coverage_text = (
        "n/a (in-process: no per-request denominator)"
        if gate["pass2_hit_coverage"] is None
        else f"{gate['pass2_hit_coverage']:.3f}"
    )
    print(json.dumps(report, indent=2))
    print(
        f"[parity] hit_ratio={hit_ratio:.3f} "
        f"hit_coverage={coverage_text} "
        f"(gated on {gate['hit_criterion']}) "
        f"score_delta(pass2-pass1)={gate['score_delta_pass2_vs_pass1']:.2f} "
        f"score_delta(pass1-baseline)="
        f"{gate['score_delta_pass1_vs_baseline']:.2f} "
        f"flips(pass2 vs pass1)={flips_p2_p1}/{len(items)} "
        f"flips(pass1 vs baseline)={flips_p1_base}/{len(items)} "
        f"=> {'PASS' if gate['pass'] else 'FAIL'}"
    )
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    code = main()
    # vLLM/NCCL teardown can hang for a long time while holding all GPU
    # memory; the report is already on disk, so exit hard.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
