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

The LMCache passes run on the multi-process deployment -- the only one this
suite drives -- against a cache server started here. Passing
``--hybrid-block-tokens N`` (a Mamba/GDN model's vLLM unified block size)
additionally chunks that server at the unified block size and gives each KV
cache group its own cache objects.

Exit code 0 = parity holds (see THRESHOLDS below), 1 = parity violated.
"""

# Standard
import abc
import argparse
import base64
import dataclasses
import glob
import io
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time

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
# Score deltas are reported but NOT gated. MME's per-category scores are
# quantized at ~7.5 points per borderline question, so a 10.0-point budget
# handed single marginal answers the verdict: the same deterministic
# 18-question flip core measured 9.00 / 2.25 / 9.75 across three identical
# qwen2-vl-2b runs (records/2026/08/26/8_). The flip SET is the stable,
# gateable quantity; a concentration of flips in one category is diagnosed
# from the per-category table the report already carries.
MAX_FLIP_FRACTION = 0.005  # verdict-to-verdict answer flips, both comparisons
# Answer flips are bounded by COUNT above and by DIRECTION here, because the
# count alone cannot separate the two things that move a verdict. Engine
# noise is two-sided: a question sitting within one bf16 quantum of the
# yes/no boundary is a coin flip, so regressions and improvements arrive in
# roughly equal numbers -- qwen2-vl-2b on vLLM 0.27.1 flips 19 of 2374 and
# moves the MME total by 2.25 of 1968.78, with the per-category table gaining
# and losing in turn (records/2026/08/26/10_). A KV defect is one-sided: it
# only degrades, which is how the stream-ordering corruption was recognized
# before it was located. This threshold is the exact one-sided binomial tail
# for the observed regression share against a fair coin; a run fails when
# chance explains the skew with probability below it. At 19 flips that takes
# 15 or more leaning one way, so a widened per-model count budget buys
# tolerance for noise without also buying tolerance for a defect.
MAX_FLIP_ASYMMETRY_P = 0.01
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
# Stores are submitted when a request finishes but commit on the MP server
# asynchronously. A full benchmark hides that tail (pass 1 keeps running
# minutes past its last submit), but a --limit smoke run reaches pass 2
# while pass 1's stores are still in flight and reads a misleadingly low
# hit ratio: measured 0.14 at --limit 40 on a tree whose full run measures
# 0.98. Sized on the 110-item probe runs, where 5s reliably yielded full
# pass-2 hits.
STORE_COMMIT_GRACE_S = 5.0
# Parse flips (a verdict on one side, '' on the other) are budgeted apart
# from answer flips: they measure how many answers sit on the model's own
# abstain/answer margin, not whether the cache changed a verdict.
# gemma-4-e4b is the measured case (records/2026/08/26/8_): at parse ratio
# 0.896 -- 239 hard refusals that no decode budget resolves, see
# ModelSpec.mme_min_parse_ratio -- its two full MME runs flipped 4 and 14
# answers ''<->verdict in BOTH directions (net parse-ratio movement 0.0008
# and 0.0000) while flipping 1 and 1 actual verdicts. Counting parse flips
# against MAX_FLIP_FRACTION made the gate verdict an unreproducible coin
# toss (PASS then FAIL, flip-set jaccard 0.11). A real hit-path defect
# moves parseability in ONE direction instead: the 2026-08-21
# KEY_NOT_READABLE regression truncated enough pass-2 answers to move the
# parse ratio by ~0.4. So the gate bounds the parse-ratio DELTA between
# passes; 0.02 sits an order of magnitude above the measured marginality
# noise and an order below the measured defect.
MAX_PARSE_RATIO_DELTA = 0.02


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
            for readers; score deltas are reported, not gated (see the
            threshold block above ``MAX_FLIP_FRACTION``).
    """

    key: str
    modality: str
    score_scale: float

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
    def ground_truth(self, item: dict) -> str:
        """The correct verdict for one question, in ``parse_answer`` terms.

        Lets ``count_flips`` label a flip as a regression or an improvement,
        so the gate can tell one-sided corruption from two-sided numeric
        noise. The value must be comparable with what ``parse_answer``
        returns for the same item, since ``scores`` already compares the
        two directly.

        Args:
            item: The item whose answer key is wanted.
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

    def ground_truth(self, item: dict) -> str:
        """MME's yes/no answer key, as the dataset stores it.

        Args:
            item: The item whose answer key is wanted.
        """
        return item["answer"]

    def scores(self, items: list[dict], answers: list[str]) -> dict:
        """Standard MME scoring: per-category acc*100 + acc+*100, summed.

        acc  = per-question accuracy;
        acc+ = fraction of images whose BOTH questions are answered correctly.
        Perception sums 10 categories (max 2000), Cognition 4 (max 800).
        """
        by_cat: dict[str, dict[str, list]] = {}
        for item, answer in zip(items, answers, strict=True):
            cat = by_cat.setdefault(item["category"], {"correct": [], "by_image": {}})
            ok = self.parse_answer(answer, item) == self.ground_truth(item)
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
    # Mean per-task accuracy, as a percentage. The measured nondeterminism
    # floor on this benchmark is zero: Qwen3-Omni-30B over the full 1000
    # questions returned byte-identical scores (66.90; music 70.06 / sound
    # 71.47 / speech 59.16) with 0 flips on BOTH comparisons -- including
    # baseline-vs-pass1, which crosses processes and engine configs.
    score_scale = 100.0

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

    def ground_truth(self, item: dict) -> str:
        """The correct option letter for this question.

        Args:
            item: The item whose answer key is wanted.
        """
        return item["answer_letter"]

    def scores(self, items: list[dict], answers: list[str]) -> dict:
        """Per-task accuracy percentages, plus their mean as ``total``.

        The mean is over TASKS rather than over questions so the three
        weigh equally. The split is near-balanced (333/333/334) so the two
        agree closely, but a ``--limit`` prefix need not be exactly
        balanced, and a per-task mean stays interpretable when it is not.
        """
        by_task: dict[str, list[bool]] = {}
        for item, answer in zip(items, answers, strict=True):
            ok = self.parse_answer(answer, item) == self.ground_truth(item)
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


@dataclasses.dataclass(frozen=True)
class FlipCounts:
    """Tally of changed verdicts between two aligned answer passes.

    One pass is under test and the other is the reference it is compared
    against; that asymmetry is what makes ``regressions`` meaningful, and
    ``count_flips`` fixes which is which.

    ``regressions``, ``improvements`` and ``lateral`` partition
    ``answer_flips``.

    Attributes:
        answer_flips: Questions where both passes parsed to a verdict and
            the verdicts differ -- the only kind of flip a KV defect must
            produce, and what ``MAX_FLIP_FRACTION`` budgets.
        parse_flips: Questions where exactly one pass parsed to a verdict.
            These measure abstain/answer marginality and are bounded via
            the parse-ratio delta instead (``MAX_PARSE_RATIO_DELTA``).
        regressions: Answer flips where the reference pass was right and
            the pass under test is wrong. Their share of the directional
            flips is what ``MAX_FLIP_ASYMMETRY_P`` bounds.
        improvements: Answer flips the other way round: the reference was
            wrong and the pass under test is right.
        lateral: Answer flips between two wrong verdicts, which carry no
            direction and so stay bounded by the count budget alone.
            Unreachable on a two-verdict benchmark like MME, where a
            changed verdict always crosses the answer key; reachable on
            MMAU, whose questions offer up to four options.
    """

    answer_flips: int
    parse_flips: int
    regressions: int
    improvements: int
    lateral: int

    @property
    def total(self) -> int:
        """Every flipped question; what the pre-split gate counted."""
        return self.answer_flips + self.parse_flips


def count_flips(
    benchmark: Benchmark,
    items: list[dict],
    answers_x: list[str],
    answers_y: list[str],
) -> FlipCounts:
    """Classify every changed verdict between two aligned passes.

    Args:
        benchmark: Supplies ``parse_answer`` and ``ground_truth``.
        items: The questions, aligned with both answer lists.
        answers_x: Generated answers of the pass UNDER TEST, in item order
            (the hit pass, or the cold pass when the reference is the
            no-LMCache baseline).
        answers_y: Generated answers of the REFERENCE pass, same order.
            A flip away from a correct reference verdict is the regression
            that ``FlipCounts.regressions`` counts.

    Returns:
        The flip tally; every count is zero when the passes agree on every
        question.
    """
    answer_flips = 0
    parse_flips = 0
    regressions = 0
    improvements = 0
    lateral = 0
    for a, b, item in zip(answers_x, answers_y, items, strict=True):
        verdict_a = benchmark.parse_answer(a, item)
        verdict_b = benchmark.parse_answer(b, item)
        if verdict_a == verdict_b:
            continue
        if not (verdict_a and verdict_b):
            parse_flips += 1
            continue
        answer_flips += 1
        truth = benchmark.ground_truth(item)
        if verdict_b == truth:
            regressions += 1
        elif verdict_a == truth:
            improvements += 1
        else:
            lateral += 1
    return FlipCounts(
        answer_flips=answer_flips,
        parse_flips=parse_flips,
        regressions=regressions,
        improvements=improvements,
        lateral=lateral,
    )


def flip_asymmetry_p(regressions: int, improvements: int) -> float:
    """Probability that two-sided noise alone produces this regression skew.

    The exact one-sided binomial tail ``P(X >= regressions)`` for X over
    ``regressions + improvements`` fair coin flips. A small value means the
    flips lean toward "was right, now wrong" further than chance accounts
    for, which is a corrupting cache rather than the engine's batch-shape
    numerics. The test is one-sided on purpose: an excess of improvements
    is not a corruption signature.

    The fair coin is the right null even for an accurate model, which is
    not obvious: one would expect a flip to break a correct answer as
    often as the model is correct, so that a model at 85% should regress
    on 85% of its flips with nothing wrong. Measured on vLLM 0.27.1
    (2026-08-27) across four full MME runs, it does not -- qwen2.5-vl-3b
    at 84.9% accuracy flips 13 to 11, qwen2-vl-2b at 83.2% flips 10 to 8,
    and mistral-small-3.1-24b at 84.5% flips 4 to 4. Flips land on the
    items the model is genuinely undecided about, and those are balanced
    whatever the overall accuracy is. See records/2026/08/27/2_.

    Args:
        regressions: Flips where the reference pass was correct and the
            pass under test is not.
        improvements: Flips the other way round.

    Returns:
        A probability in (0, 1]; 1.0 when there are no directional flips,
        which reads as "no evidence of skew", not as "no skew".
    """
    n = regressions + improvements
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(regressions, n + 1))
    return tail / float(2**n)


def answer_parse_ratio(
    benchmark: Benchmark, items: list[dict], answers: list[str]
) -> float:
    """Fraction of one pass's answers that parse to a verdict.

    Args:
        benchmark: Supplies ``parse_answer``.
        items: The questions, aligned with the answers.
        answers: Generated answers of one pass, in item order.

    Returns:
        The ratio in [0, 1], rounded to 4 places; 0.0 for no questions.
    """
    if not items:
        return 0.0
    parsed = sum(
        1
        for a, item in zip(answers, items, strict=True)
        if benchmark.parse_answer(a, item)
    )
    return round(parsed / len(items), 4)


def lateral_text(counts: FlipCounts) -> str:
    """Summary-line fragment naming direction-free flips, empty when none.

    Args:
        counts: The tally whose ``lateral`` count is to be rendered.

    Returns:
        ``"/=N"`` for N direction-free flips, or ``""`` when the benchmark
        produced none (always the case for MME).
    """
    return f"/={counts.lateral}" if counts.lateral else ""


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
        the hit criterion that applied, and the thresholds used. Score
        deltas appear for observability but do not gate (see the threshold
        block above ``MAX_FLIP_FRACTION``). ``flip_asymmetry_p`` carries one
        entry per comparison whose direction counts the report records, and
        is empty for a report predating them. ``pass2_hit_coverage`` is None
        when the report provides no denominator for it (one recorded
        before the coverage fields existed); that is "not measured", not a
        coverage of zero, and it fails a coverage gate rather than
        satisfying one.
    """
    # First Party (test-local)
    from harness import LMCACHE_TEST_CHUNK_SIZE

    scores = report["scores"]
    flip_fraction = max_flip_fraction or MAX_FLIP_FRACTION
    parse_floor = min_parse_ratio or MIN_PARSE_RATIO
    max_flips = flip_fraction * report["num_questions"]
    delta_p2_p1 = abs(scores["pass2_hit"]["total"] - scores["pass1_miss"]["total"])
    delta_p1_base = abs(scores["pass1_miss"]["total"] - scores["baseline"]["total"])
    # Reports recorded before the answer/parse split carry only the
    # combined flip counts; gating those combined counts is the pre-split
    # behavior and can only over-fail (the combined count includes parse
    # flips), never let a defect through.
    answer_flips_p2_p1 = report.get(
        "answer_flips_pass2_vs_pass1", report["flips_pass2_vs_pass1"]
    )
    answer_flips_p1_base = report.get(
        "answer_flips_pass1_vs_baseline", report["flips_pass1_vs_baseline"]
    )
    # Parse-flip movement is bounded through the per-pass parse ratios; a
    # report without them (pre-split) contributes no deltas and is bounded
    # by its combined flip counts above.
    parse_ratio_deltas: dict[str, float] = {}
    for delta_key, ratio_a, ratio_b in (
        ("pass2_vs_pass1", "pass2_answer_parse_ratio", "pass1_answer_parse_ratio"),
        (
            "pass1_vs_baseline",
            "pass1_answer_parse_ratio",
            "baseline_answer_parse_ratio",
        ),
    ):
        if ratio_a in report and ratio_b in report:
            parse_ratio_deltas[delta_key] = round(
                abs(report[ratio_a] - report[ratio_b]), 4
            )
    parse_stable = all(
        delta <= MAX_PARSE_RATIO_DELTA for delta in parse_ratio_deltas.values()
    )
    # Which way the answer flips lean. A report recorded before the
    # direction counters existed contributes no entry and keeps gating on
    # count alone, which is the pre-direction behavior: that can only
    # over-fail relative to the count budget, never let a defect through
    # that the budget would have caught.
    asymmetry_p: dict[str, float] = {}
    for delta_key in ("pass2_vs_pass1", "pass1_vs_baseline"):
        regressions = report.get(f"answer_regressions_{delta_key}")
        improvements = report.get(f"answer_improvements_{delta_key}")
        if regressions is None or improvements is None:
            continue
        # Kept at full precision: rounding before the comparison could lift
        # a tail that sits just under the threshold up onto it.
        asymmetry_p[delta_key] = flip_asymmetry_p(regressions, improvements)
    flips_two_sided = all(
        probability >= MAX_FLIP_ASYMMETRY_P for probability in asymmetry_p.values()
    )
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
    # None, not 0.0, when the denominator is missing -- a report recorded
    # before the coverage fields existed, or one from the removed
    # in-process path, which had no per-request lookup lengths. Publishing
    # 0.0 there reads as "the cache achieved nothing" for a run whose raw
    # hit ratio was 1.0 -- an unmeasured quantity must not look like a
    # measured zero.
    coverage = round(loaded / achievable, 4) if achievable else None
    if granularity > LMCACHE_TEST_CHUNK_SIZE:
        hit_criterion = "coverage"
        # An unmeasurable coverage cannot satisfy a coverage gate.
        hit_ok = coverage is not None and coverage >= MIN_HIT_COVERAGE
    else:
        hit_criterion = "raw_hit_ratio"
        hit_ok = hit_ratio >= MIN_HIT_RATIO
    ok = (
        answer_flips_p2_p1 <= max_flips
        and answer_flips_p1_base <= max_flips
        and flips_two_sided
        and parse_stable
        and hit_ok
        and parse_ratio >= parse_floor
    )
    return {
        "pass": ok,
        "max_flips": max_flips,
        "answer_flips_pass2_vs_pass1": answer_flips_p2_p1,
        "answer_flips_pass1_vs_baseline": answer_flips_p1_base,
        "flip_asymmetry_p": asymmetry_p,
        "parse_ratio_deltas": parse_ratio_deltas,
        "score_delta_pass2_vs_pass1": delta_p2_p1,
        "score_delta_pass1_vs_baseline": delta_p1_base,
        "baseline_answer_parse_ratio": parse_ratio,
        "hit_criterion": hit_criterion,
        "cache_granularity_tokens": granularity,
        "pass2_hit_coverage": coverage,
        "thresholds": {
            "max_flip_fraction": flip_fraction,
            "max_flip_asymmetry_p": MAX_FLIP_ASYMMETRY_P,
            "max_parse_ratio_delta": MAX_PARSE_RATIO_DELTA,
            "min_hit_ratio": MIN_HIT_RATIO,
            "min_hit_coverage": MIN_HIT_COVERAGE,
            "min_parse_ratio": parse_floor,
        },
    }


def run_batch(
    llm,
    benchmark: Benchmark,
    items: list[dict],
    chat_template: str,
    chat_template_kwargs: dict,
    max_tokens: int,
) -> list[str]:
    """Run all questions in one batched chat call, order-aligned.

    Args:
        llm: The vLLM engine.
        benchmark: The benchmark, which renders the conversations.
        items: Question dicts (see ``Benchmark.load_items``).
        chat_template: Chat template from the model spec, for a model whose
            repo ships none; empty string uses the model's own.
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
        chat_template=chat_template or None,
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
    trust_remote_code: bool,
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
        trust_remote_code: ``ModelSpec.trust_remote_code`` -- whether this
            model's repo config can only be read by executing repo code
            (Molmo 2). Applied to BOTH engines: without it the engine that
            has it would be the only one that starts, leaving nothing to
            compare against.
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
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
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
    chat_template: str,
    chat_template_kwargs: dict,
    mm_processor_kwargs: dict,
    max_tokens: int,
    hybrid_block_tokens: int,
    hf_overrides: dict,
    hybrid_family: str,
    mm_encoder_attn_backend: str,
    trust_remote_code: bool,
) -> None:
    """Subprocess role: plain vLLM answers for every question."""
    # Third Party
    from vllm import LLM

    # This subprocess writes to the parent's stdout, so its lines carry a
    # distinct tag -- otherwise the two streams interleave under one prefix
    # and there is no telling which process is where.
    #
    # It reloads the dataset the parent already loaded, because the items
    # (base64 data URIs for 1097 MME images) do not cross the process
    # boundary. That is another 12-13 minutes of silent CPU before this
    # role touches a GPU, and until these lines existed the log showed
    # nothing at all for it.
    print(f"[parity:baseline] loading {benchmark.key.upper()} items (limit={limit})")
    load_started = time.monotonic()
    items = benchmark.load_items(limit)
    print(
        f"[parity:baseline] {len(items)} items loaded in "
        f"{time.monotonic() - load_started:.1f}s; building the plain-vLLM engine"
    )
    llm = LLM(
        **engine_kwargs(
            model,
            benchmark,
            mm_processor_kwargs,
            hybrid_block_tokens,
            hf_overrides,
            hybrid_family,
            mm_encoder_attn_backend,
            trust_remote_code,
        )
    )
    answers = run_batch(
        llm, benchmark, items, chat_template, chat_template_kwargs, max_tokens
    )
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
        "--chat-template",
        default="",
        help="Jinja chat template from the model spec, for a model whose "
        "repo ships none; empty = use the model's own",
    )
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
        help="MP cache server L1 capacity override "
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
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="pass vLLM's trust_remote_code to BOTH engines "
        "(ModelSpec.trust_remote_code). Only for a repo whose config cannot "
        "be read without it -- Molmo 2 ships auto_map and transformers 5.15 "
        "refuses the config outright, though vLLM implements the model "
        "natively and never runs the repo's modeling code",
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
        reset_vllm_prefix_cache,
    )

    hf_overrides: dict = json.loads(args.hf_overrides) if args.hf_overrides else {}

    configure_environment()

    if args.role == "baseline":
        run_baseline(
            args.model,
            benchmark,
            args.limit,
            args.baseline_out,
            args.chat_template,
            chat_template_kwargs,
            mm_processor_kwargs,
            args.max_tokens,
            args.hybrid_block_tokens,
            hf_overrides,
            args.hybrid_family,
            args.mm_encoder_attn_backend,
            args.trust_remote_code,
        )
        return 0

    # Progress markers, not decoration. Loading the full MME set takes
    # 12-13 minutes (measured on 2374 questions, records/2026/08/26/7_),
    # and until this run printed something a stalled log looked exactly
    # like a live one -- which is how a healthy run got killed at the
    # 14-minute mark for "hanging".
    print(f"[parity] loading {args.benchmark.upper()} items (limit={args.limit})")
    load_started = time.monotonic()
    items = benchmark.load_items(args.limit)
    print(
        f"[parity] {len(items)} {benchmark.key.upper()} questions loaded "
        f"in {time.monotonic() - load_started:.1f}s"
    )

    baseline_path = pathlib.Path(args.out).with_suffix(".baseline.json")
    # A store_true flag cannot be forwarded as an empty string like the
    # others, so it is appended only when set.
    trust_flag = ["--trust-remote-code"] if args.trust_remote_code else []
    print(
        f"[parity] spawning the baseline subprocess -> {baseline_path} "
        f"(its output is inherited, so it lands in this same stream)"
    )
    baseline_started = time.monotonic()
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
            "--chat-template",
            args.chat_template,
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
        ]
        + trust_flag,
        timeout=7200,
    )
    print(
        f"[parity] baseline subprocess exited {proc.returncode} after "
        f"{time.monotonic() - baseline_started:.1f}s"
    )
    if proc.returncode != 0:
        raise RuntimeError(f"baseline subprocess failed with {proc.returncode}")
    answers_base = json.loads(baseline_path.read_text())

    # Third Party
    from vllm import LLM

    # First Party (test-local)
    from harness import (
        MPTransportCounters,
        mp_kv_transfer_config,
        start_mp_cache_server,
    )

    # The LMCache pass runs on the multi-process deployment, the only one
    # this suite drives (see the harness module docstring), so it needs its
    # own cache server and reads its lookup counters off the MP adapters. A
    # hybrid additionally chunks at the unified block size and gives each
    # KV cache group its own objects.
    prefill = VllmPrefillCounters()
    prefill.install()
    server = start_mp_cache_server(
        zmq_port=26000 + (os.getpid() % 1000),
        http_port=27000 + (os.getpid() % 1000),
        chunk_size=args.hybrid_block_tokens or LMCACHE_TEST_CHUNK_SIZE,
        log_path=pathlib.Path(args.out).with_suffix(".mp_server.log"),
        l1_size_gb=args.max_local_cpu_gb or 40.0,
        separate_object_groups=bool(args.hybrid_block_tokens),
    )
    kv_transfer_config = mp_kv_transfer_config(server.zmq_port)
    counters = MPTransportCounters()
    counters.install()

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
            args.trust_remote_code,
        ),
    )

    def lookup_stats() -> tuple[int, int]:
        """Cumulative (lookup_tokens, lookup_hits) since engine start."""
        return (counters.lookup_tokens, counters.lookup_hits)

    def stored_tokens() -> int:
        """Cumulative tokens submitted to the MP server for storage."""
        return counters.stored_tokens

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
            llm,
            benchmark,
            items,
            args.chat_template,
            chat_template_kwargs,
            args.max_tokens,
        )
        stored_p1 = stored_tokens() - stored_before
        # See STORE_COMMIT_GRACE_S: without this, a --limit run's pass 2
        # laps pass 1's in-flight stores and under-reports the hit ratio.
        time.sleep(STORE_COMMIT_GRACE_S)
        reset_local_prefix_cache()
        t0, h0 = lookup_stats()
        local0, external0 = prefill.local_cached, prefill.external_cached
        lookups0 = len(counters.lookup_request_tokens)
        answers_p2 = run_batch(
            llm,
            benchmark,
            items,
            args.chat_template,
            chat_template_kwargs,
            args.max_tokens,
        )
        t1, h1 = lookup_stats()
        local_p2 = prefill.local_cached - local0
        external_p2 = prefill.external_cached - external0
        achievable_p2 = achievable_hit_tokens(
            counters.lookup_request_tokens[lookups0:],
            args.hybrid_block_tokens or LMCACHE_TEST_CHUNK_SIZE,
        )
        hit_ratio = (h1 - h0) / max(1, t1 - t0)
    finally:
        server.process.terminate()

    scores = {
        "baseline": benchmark.scores(items, answers_base),
        "pass1_miss": benchmark.scores(items, answers_p1),
        "pass2_hit": benchmark.scores(items, answers_p2),
    }
    flips_p1_base = count_flips(benchmark, items, answers_p1, answers_base)
    flips_p2_p1 = count_flips(benchmark, items, answers_p2, answers_p1)
    report = {
        "model": args.model,
        "benchmark": benchmark.key,
        "deployment_path": "mp",
        "num_questions": len(items),
        "scores": scores,
        "flips_pass1_vs_baseline": flips_p1_base.total,
        "flips_pass2_vs_pass1": flips_p2_p1.total,
        "answer_flips_pass1_vs_baseline": flips_p1_base.answer_flips,
        "parse_flips_pass1_vs_baseline": flips_p1_base.parse_flips,
        "answer_regressions_pass1_vs_baseline": flips_p1_base.regressions,
        "answer_improvements_pass1_vs_baseline": flips_p1_base.improvements,
        "answer_lateral_pass1_vs_baseline": flips_p1_base.lateral,
        "answer_flips_pass2_vs_pass1": flips_p2_p1.answer_flips,
        "parse_flips_pass2_vs_pass1": flips_p2_p1.parse_flips,
        "answer_regressions_pass2_vs_pass1": flips_p2_p1.regressions,
        "answer_improvements_pass2_vs_pass1": flips_p2_p1.improvements,
        "answer_lateral_pass2_vs_pass1": flips_p2_p1.lateral,
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
        "baseline_answer_parse_ratio": answer_parse_ratio(
            benchmark, items, answers_base
        ),
        "pass1_answer_parse_ratio": answer_parse_ratio(benchmark, items, answers_p1),
        "pass2_answer_parse_ratio": answer_parse_ratio(benchmark, items, answers_p2),
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
        "n/a (report carries no per-request denominator)"
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
        f"answer_flips(pass2 vs pass1)="
        f"{flips_p2_p1.answer_flips}/{len(items)} "
        f"(-{flips_p2_p1.regressions}/+{flips_p2_p1.improvements}"
        f"{lateral_text(flips_p2_p1)}, "
        f"p={gate['flip_asymmetry_p']['pass2_vs_pass1']:.4f}, "
        f"+{flips_p2_p1.parse_flips} parse) "
        f"answer_flips(pass1 vs baseline)="
        f"{flips_p1_base.answer_flips}/{len(items)} "
        f"(-{flips_p1_base.regressions}/+{flips_p1_base.improvements}"
        f"{lateral_text(flips_p1_base)}, "
        f"p={gate['flip_asymmetry_p']['pass1_vs_baseline']:.4f}, "
        f"+{flips_p1_base.parse_flips} parse) "
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
