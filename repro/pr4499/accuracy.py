#!/usr/bin/env python3
"""Layer 2b, third regime: correctness of the retrieved KV, and the store cut.

Two questions, both of them correctness questions, measured in one run:

1. **Did lazy really store less, without losing the hits it needs?** The
   store ledger says how many chunks each policy admitted; the external hit
   rate says how many of them were actually read back. A policy that stores
   less and hits at the same order is doing its job; one that stores less and
   hits far less has merely done less work.

2. **Is what came back out of the cache the right KV?** A wrong offset, a
   chunk filed under the wrong position, or a prefix rebuilt to the wrong
   length all return *correct bytes* attached to the *wrong tokens*. Layer 1's
   S6/S19 compare stored bytes and cannot see any of it. The layer-2 oracle
   (#18) compares tokens, but on a 0.6B model, 8 hand-written prompts and 32
   output tokens. This file asks the same question where a wrong answer has
   somewhere to hide: a real task, a real score, 8B, ~3100-token prompts and
   ~200 output tokens, with the whole prefill served out of L1.

The task is GSM8K, 20-shot, greedy. Two properties make it the right probe:

- It is **scored**, so "the tokens are fine" stops being a judgement call. A
  corrupted prefix does not produce a plausible wrong answer, it produces a
  collapsed score.
- Each question gets its **own** 20 exemplars, drawn from the train split with
  a per-question seed. That is what makes the prompts long and mostly unique,
  so pass 2 has ~1900 tokens per request to retrieve rather than one shared
  prefix that the GPU would keep resident anyway.

Structure: two passes over the same 120 prompts against one engine.

    pass 1   cold. Nothing is cached; every prompt is computed from scratch.
             This is the reference score *within* the config.
    pass 2   identical prompts. The GPU pool (2048 blocks = 32768 tokens, ~10
             requests) has turned over many times, so vLLM's own prefix cache
             is empty and the prefill can only come from L1.

That gives two independent comparisons, and the first is the stronger one:

- **within a config, pass 1 vs pass 2.** Same engine, same prompts, same
  sampler; the only difference is that pass 2's KV was fetched instead of
  computed. Any drop is retrieval corruption, and it cannot be blamed on a
  configuration difference because there isn't one.
- **across configs, off vs eager vs lazy.** `off` never stores, so its pass 2
  is a recompute reference.

Vacuity guards, printed with every run and checked in `table()`:

- pass 2 `apc` (vLLM's own prefix-cache hit rate) must be near 0. If the GPU
  served the prefill, nothing was retrieved and a passing score proves
  nothing. This is the trap the first regime fell into (record 1 section 4).
- pass 2 `ext` must be substantially above 0 for eager and lazy. Same reason
  from the other side.
- L1 must stay below the eviction watermark. Displacement is record 2's
  regime; here it would only add noise to a correctness reading.

On exact token identity across configs: it is reported but it is *not* the
pass criterion. Retrieval changes how many tokens remain to be prefilled,
which changes the prefill split, which perturbs the logits in the last
bits -- so a greedy decode of 200 tokens may legitimately diverge without any
bug. The score is the criterion; identity is a diagnostic, and a divergence
position is printed so a real corruption (diverges immediately, then rambles)
can be told apart from a numerical one (diverges late, stays coherent).

Usage:
    python accuracy.py run <off|eager|lazy> [rep] [n_questions] [concurrency]
                          [l1_gb]
    python accuracy.py table
"""

# Standard
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
import random
import re
import statistics
import sys
import threading
import time
import urllib.request

# First Party
import driver
from driver import (
    LOGDIR,
    cache_object_count,
    grep_final_counters,
    grep_tracebacks,
    grep_warnings,
    mode_lines,
    server_status,
    teardown,
)
from workload import MODEL, TP_SIZE, _metrics, start_engine, start_server_sized

CONFIGS = ("off", "eager", "lazy")

#: GSM8K, dumped from the HuggingFace datasets cache to jsonl so this harness
#: needs neither `datasets` nor a network round trip. Fields: question, answer.
_GSM8K = driver.BASE / "gsm8k"

#: Test questions scored per pass. Measured with the Qwen3-8B tokenizer, not
#: estimated: these prompts run 2451-4013 tokens, median 3120, so 120 of them
#: are 55.4 GB of KV. That is the number L1_GB has to clear.
N_QUESTIONS = 120

#: Exemplars prepended to each question, and the fact that each question gets
#: a *different* draw. 20 exemplars is ~3000 tokens -- 12 LMCache chunks --
#: which is what gives pass 2 something substantial to retrieve. A single
#: shared few-shot prefix would be a dozen chunks for the whole run, would
#: stay resident in the GPU cache, and pass 2 would then measure nothing.
N_SHOTS = 20

#: Output budget. GSM8K chains of thought run 60-200 tokens; the stop sequence
#: below normally ends generation well inside this.
MAX_TOKENS = 320

#: Requests in flight. 4 matches the first regime, so latency is comparable
#: across records.
#:
#: It also sets the floor on how sharp the identity comparison can be. At 4,
#: a request's decode step is batched with whatever else is in flight, and
#: batch composition varies run to run, so a greedy decode is not reproducible
#: even with no cache involved at all: `off` resending the same 120 prompts
#: reproduced only 100 of its own completions byte for byte. Run at 1 to take
#: that source out and leave only the prefill-split difference that retrieval
#: itself causes.
CONCURRENCY = 4

#: KV pool in 16-token blocks. 2048 blocks = 32768 tokens ~ 10 of these
#: requests; the working set is 120 of them, so by the time pass 2 reaches a
#: prompt its GPU blocks are long gone. Verified by the pass-2 `apc` guard.
POOL_BLOCKS = 2048

#: L1 budget in GB, overridable per run. The host has 2 TB, so every size here
#: is a deliberate constraint and not a resource limit.
#:
#: The corpus is 55.4 GB by token count and 51 GB as actually stored (chunks
#: are shared between prompts), against a 4.9 GB pool. Three sizes matter:
#:
#: 128 GB   corpus at 0.40, far under the 0.8 eviction watermark. Nothing is
#:          ever displaced, so the score reads pure retrieval correctness.
#:          Record 4's size, kept as the reference.
#:  68 GB   corpus at 0.75, the tightest budget that still clears the
#:          watermark, and it satisfies pool 4.9 < corpus 55.4 < pool + L1
#:          72.9. This is the size the scores are read at.
#:  52 GB   corpus at 0.98, deliberately over-subscribed. Both passes walk the
#:          questions in the same order, and LRU under a cyclic scan evicts
#:          precisely what the next pass is about to ask for, so the external
#:          hit rate can collapse to near zero -- for *both* connector
#:          configs, since gate 1 reasons about GPU blocks and knows nothing
#:          of the L1 watermark. Measured as a control: if it collapses, the
#:          squeeze destroys the signal rather than sharpening it, and the
#:          `ext` guard in table() marks those rows unusable.
L1_GB = 68

_PASSES = ("cold", "cached")

#: Answers end with a `#### <number>` line in GSM8K, both in the gold answers
#: and -- because every exemplar shows it -- in the model's continuation.
_FINAL = re.compile(r"####\s*(-?[\d,]*\.?\d+)")

#: Calculator annotations in the gold rationales, stripped so the exemplars
#: read as plain arithmetic. Standard practice for this dataset.
_CALC = re.compile(r"<<[^>]*>>")

#: Any number, for the lenient extraction: the last one in the completion.
_NUMBER = re.compile(r"-?[\d,]*\.?\d+")

_PROMPT_TEMPLATE = "Question: {q}\nAnswer: {a}\n\n"

#: Ends generation once the model starts inventing the next question, which is
#: what a 20-shot completion prompt naturally leads it to do.
STOP = ["\nQuestion:", "\n\nQuestion:"]


def _load(split: str) -> list[dict[str, str]]:
    """Read one GSM8K split from the local jsonl dump.

    Args:
        split: "train" or "test".

    Returns:
        The split's rows, each with "question" and "answer" keys.

    Raises:
        FileNotFoundError: if the dump is missing; see _GSM8K.
    """
    path = _GSM8K / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; dump the datasets cache first")
    # split("\n"), not splitlines(): a few GSM8K rows carry a U+2028 line
    # separator, which json escapes but str.splitlines() still breaks on.
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def gold(answer: str) -> str:
    """The reference number from a GSM8K gold answer.

    Args:
        answer: The dataset's answer field, ending in a `#### N` line.

    Returns:
        The number as a normalised string.

    Raises:
        ValueError: if the answer carries no `####` line.
    """
    found = _FINAL.search(answer)
    if not found:
        raise ValueError(f"no #### line in {answer!r}")
    return _normalise(found.group(1))


def _normalise(number: str) -> str:
    """Strip thousands separators and a trailing `.0` from a number string."""
    text = number.replace(",", "").rstrip(".")
    if text.endswith(".0"):
        text = text[:-2]
    return text


def extract_strict(completion: str) -> str:
    """The model's answer, read from its `#### N` line.

    Args:
        completion: The generated text.

    Returns:
        The normalised number, or "" if the model never emitted a `####` line.
    """
    found = _FINAL.findall(completion)
    return _normalise(found[-1]) if found else ""


def extract_lenient(completion: str) -> str:
    """The model's answer, read as the last number anywhere in the text.

    Reported alongside the strict form because a formatting slip and a wrong
    computation are different failures, and only the second one would be
    caused by a corrupted prefix.

    Args:
        completion: The generated text.

    Returns:
        The normalised number, or "" if the text holds no number.
    """
    found = _NUMBER.findall(completion)
    return _normalise(found[-1]) if found else ""


def prompts() -> tuple[list[str], list[str]]:
    """Build the prompt set and its gold answers.

    Each of the first N_QUESTIONS test questions gets its own N_SHOTS
    exemplars, drawn without replacement from the train split under
    `random.Random(index)`. The draw is seeded per question index, so every
    config sees byte-identical prompts and the comparison across configs is
    over the same task.

    Returns:
        (prompts in question order, gold answers in the same order).
    """
    train = _load("train")
    test = _load("test")[:N_QUESTIONS]
    built: list[str] = []
    golds: list[str] = []
    for index, row in enumerate(test):
        shots = random.Random(index).sample(train, N_SHOTS)
        text = "".join(
            _PROMPT_TEMPLATE.format(q=s["question"], a=_CALC.sub("", s["answer"]))
            for s in shots
        )
        built.append(text + f"Question: {row['question']}\nAnswer:")
        golds.append(gold(row["answer"]))
    return built, golds


def generate(prompt: str) -> tuple[float, str]:
    """Stream one greedy completion, timing the first token.

    Args:
        prompt: The prompt text.

    Returns:
        (time to first token in ms, the generated text).

    Raises:
        RuntimeError: if the stream carried no token at all.
    """
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "seed": 0,
            "stop": STOP,
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{driver.VLLM_PORT}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    first = 0.0
    parts: list[str] = []
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if not first:
                first = time.time()
            parts.append(json.loads(line[6:])["choices"][0]["text"])
    if not first:
        raise RuntimeError("stream produced no tokens")
    return (first - start) * 1000.0, "".join(parts)


def _hit(counters: dict[str, float]) -> float:
    """External hit rate over a pass's counter deltas, nan if unqueried."""
    queries = counters.get("ext_queries", 0.0)
    return counters["ext_hits"] / queries if queries else float("nan")


def _apc(counters: dict[str, float]) -> float:
    """vLLM prefix-cache hit rate over a pass's counter deltas."""
    queries = counters.get("apc_queries", 0.0)
    return counters["apc_hits"] / queries if queries else float("nan")


def _covered(counters: dict[str, float]) -> float:
    """Fraction of queried prompt tokens served by *any* cache.

    This, not the external hit rate, is the parity metric for "lazy matched as
    much KV as eager". Gate 1 declines a store precisely when the GPU will
    serve the reuse itself, so a lower external rate is the intended outcome
    and says nothing on its own; what has to hold is that the two policies
    cover the same share of prompt tokens between them.

    vLLM counts both pairs in tokens, and the external lookup only sees what
    the GPU cache missed, so GPU hits and external hits sum without
    double-counting against the GPU cache's query total.

    Args:
        counters: A pass's counter deltas.

    Returns:
        (apc_hits + ext_hits) / apc_queries, nan if nothing was queried.
    """
    queries = counters.get("apc_queries", 0.0)
    if not queries:
        return float("nan")
    return (counters["apc_hits"] + counters.get("ext_hits", 0.0)) / queries


#: Seconds between L1 samples. The eviction loop itself ticks once a second,
#: so anything coarser than this can miss a whole trim cycle; anything finer
#: only adds `/status` round trips to a server the run is also measuring.
L1_POLL_SECONDS = 1.0

#: The eviction controller's INFO line for a cycle that actually ran. Counting
#: it in the server log is the only record of how often L1 was trimmed -- the
#: status document reports the policy and the watermark but no event count.
_EVICTION_LINE = "above watermark"


class L1Sampler:
    """Poll the MP server's L1 state on a thread for the length of a pass.

    The status document is a snapshot, so reading it once after a pass says
    where L1 ended up and nothing about where it went. Under a budget that
    sits over the eviction watermark that is the whole quantity of interest:
    the fill climbs, the eviction loop trims it, and the resulting sawtooth is
    what decides which chunks are still there to be hit on pass 2.

    A failed poll is dropped rather than raised. The server is under
    measurement and a refused or slow `/status` is not a reason to lose the
    pass that is running against it.
    """

    def __init__(self) -> None:
        """Create an idle sampler."""
        self._samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._failures = 0

    def _loop(self) -> None:
        """Sample until stopped, one poll per L1_POLL_SECONDS."""
        start = time.time()
        while not self._stop.is_set():
            try:
                l1 = server_status()["storage_manager"]["l1_manager"]
                self._samples.append(
                    {
                        "t": time.time() - start,
                        "ratio": float(l1["memory_usage_ratio"]),
                        "used_gb": float(l1["memory_used_bytes"]) / (1 << 30),
                        "objects": float(l1["total_object_count"]),
                    }
                )
            except Exception:
                self._failures += 1
            self._stop.wait(L1_POLL_SECONDS)

    def start(self) -> None:
        """Begin sampling."""
        self._thread.start()

    def stop(self) -> dict:
        """Stop sampling and summarise what L1 did.

        Returns:
            A dict with the peak and final fill, the trough after the deepest
            trim, the object-count range, the raw series, and the number of
            polls that failed.
        """
        self._stop.set()
        self._thread.join(timeout=30.0)
        if not self._samples:
            return {"samples": [], "poll_failures": self._failures}
        ratios = [s["ratio"] for s in self._samples]
        objects = [s["objects"] for s in self._samples]
        return {
            "peak_ratio": max(ratios),
            "min_ratio": min(ratios),
            "final_ratio": ratios[-1],
            "mean_ratio": statistics.fmean(ratios),
            "peak_used_gb": max(s["used_gb"] for s in self._samples),
            "final_used_gb": self._samples[-1]["used_gb"],
            "peak_objects": int(max(objects)),
            "final_objects": int(objects[-1]),
            "n_samples": len(self._samples),
            "poll_failures": self._failures,
            "samples": self._samples,
        }


def eviction_cycles(tag: str) -> int:
    """Count L1 eviction cycles the server has logged so far in this run.

    Args:
        tag: The run tag, which names the server log.

    Returns:
        Number of cycles that found L1 over the watermark and trimmed it, or
        0 if the log is not there yet.
    """
    log = LOGDIR / f"{tag}_server.log"
    if not log.exists():
        return 0
    return sum(1 for line in log.read_text(errors="replace").splitlines()
               if _EVICTION_LINE in line)


def run_pass(name: str, texts: list[str], golds: list[str], tag: str) -> dict:
    """Send one pass, score it, and read the cache counters around it.

    Args:
        name: Pass name, for the log line and the result key.
        texts: Prompts, in order.
        golds: Gold answers, in the same order.
        tag: The run tag, so the pass can read its server log.

    Returns:
        A dict with the pass's scores, per-question completions and TTFTs,
        TTFT percentiles, vLLM counter deltas, and the LMCache object count
        and L1 trace over the pass.
    """
    before = _metrics()
    objects_before = cache_object_count()
    evictions_before = eviction_cycles(tag)
    sampler = L1Sampler()
    sampler.start()
    start = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(generate, texts))
    elapsed = time.time() - start
    # Stores are asynchronous; without this the object count reads the middle
    # of the drain rather than its end.
    time.sleep(10)
    after = _metrics()
    l1_trace = sampler.stop()
    l1 = server_status()["storage_manager"]["l1_manager"]
    # In question order, so a per-question comparison across configs pairs the
    # same prompt on both sides; the percentiles below sort their own copy.
    per_question_ttft = [ttft for ttft, _ in results]
    ttfts = sorted(per_question_ttft)
    completions = [text for _, text in results]
    strict = [extract_strict(text) for text in completions]
    lenient = [extract_lenient(text) for text in completions]
    out = {
        "requests": len(texts),
        "seconds": elapsed,
        "score_strict": sum(a == b for a, b in zip(strict, golds)) / len(golds),
        "score_lenient": sum(a == b for a, b in zip(lenient, golds)) / len(golds),
        "no_final_line": sum(1 for s in strict if not s),
        "ttft_p50": ttfts[len(ttfts) // 2],
        "ttft_p90": ttfts[int(len(ttfts) * 0.9)],
        "ttft_p10": ttfts[int(len(ttfts) * 0.1)],
        "ttft_mean": sum(ttfts) / len(ttfts),
        "ttft_ms": per_question_ttft,
        "mean_chars": sum(len(t) for t in completions) / len(completions),
        "counters": {k: after[k] - before[k] for k in before},
        "objects_before": objects_before,
        "objects_after": cache_object_count(),
        "l1_usage_ratio": float(l1["memory_usage_ratio"]),
        "l1_total_gb": float(l1["memory_total_bytes"]) / (1 << 30),
        "l1_locked_after": {
            "write_locked": int(l1["write_locked_count"]),
            "read_locked": int(l1["read_locked_count"]),
            "temporary": int(l1["temporary_count"]),
        },
        "l1_trace": l1_trace,
        "eviction_cycles": eviction_cycles(tag) - evictions_before,
        "completions": completions,
        "answers": strict,
    }
    print(
        f"[ac] {name}: {len(texts)} q in {elapsed:.0f}s "
        f"strict {out['score_strict']:.3f} lenient {out['score_lenient']:.3f} "
        f"ttft p50 {out['ttft_p50']:.0f}ms "
        f"ext {_hit(out['counters']):.3f} apc {_apc(out['counters']):.3f} "
        f"obj {objects_before}->{out['objects_after']} "
        f"l1 {l1_trace.get('peak_ratio', float('nan')):.3f} peak / "
        f"{out['l1_usage_ratio']:.3f} end, {out['eviction_cycles']} evictions"
    )
    return out


def eviction_settings() -> dict[str, float]:
    """Return the MP server's live L1 eviction thresholds.

    Recorded per run because they decide whether displacement (record 2's
    regime) is in play, and this regime needs it not to be.

    Returns:
        The controller's policy name, watermark and ratio.
    """
    status = server_status()["storage_manager"]["l1_eviction_controller"]
    return {
        "policy": status["eviction_policy"],
        "trigger_watermark": float(status["trigger_watermark"]),
        "eviction_ratio": float(status["eviction_ratio"]),
    }


def run(
    config: str,
    rep: int = 0,
    n_questions: int = 0,
    concurrency: int = 0,
    l1_gb: int = 0,
) -> dict:
    """One measurement point: boot, run both passes, collect, tear down.

    Args:
        config: One of CONFIGS. `off` gets no connector and is the recompute
            reference.
        rep: Repeat index, so a cell can be measured more than once.
        n_questions: Override N_QUESTIONS; 0 keeps it.
        concurrency: Override CONCURRENCY; 0 keeps it. 1 makes the identity
            comparison sharp -- see the note on CONCURRENCY.
        l1_gb: Override L1_GB; 0 keeps it. See L1_GB for the sizes that mean
            something against this corpus.

    Returns:
        The result document, also written to logs/ac_<tag>.json.

    Raises:
        KeyError: if config is unknown.
    """
    global N_QUESTIONS, CONCURRENCY, L1_GB
    if config not in CONFIGS:
        raise KeyError(f"unknown config {config!r}; known: {CONFIGS}")
    if n_questions:
        N_QUESTIONS = n_questions
    if concurrency:
        CONCURRENCY = concurrency
    if l1_gb:
        L1_GB = l1_gb
    suffix = "" if CONCURRENCY == 4 else f"_c{CONCURRENCY}"
    # The L1 budget joins the tag only when it is not the 128 GB the first
    # round used, so those results keep the names they were recorded under.
    budget = "" if L1_GB == 128 else f"_l{L1_GB}"
    tag = f"A_{config}_n{N_QUESTIONS}{suffix}{budget}_{rep}"
    texts, golds = prompts()
    result: dict = {
        "config": config, "tag": tag, "rep": rep, "model": MODEL,
        "n_questions": N_QUESTIONS, "n_shots": N_SHOTS, "l1_gb": L1_GB,
        "pool_blocks": POOL_BLOCKS, "concurrency": CONCURRENCY,
        "tensor_parallel_size": TP_SIZE, "max_tokens": MAX_TOKENS,
    }
    print(f"[ac] === {tag}")
    server = start_server_sized(tag, L1_GB)
    eviction = eviction_settings()
    print(f"[ac] l1 eviction: {eviction}")
    try:
        engine = start_engine(tag, config, POOL_BLOCKS)
    except Exception:
        teardown([server])
        raise
    try:
        result["prompt_tokens"] = [driver.prompt_tokens(t) for t in texts[:3]]
        result["passes"] = {
            name: run_pass(name, texts, golds, tag) for name in _PASSES
        }
    finally:
        teardown([engine, server])
    result["rc_engine"] = engine.returncode
    result["rc_server"] = server.returncode
    result["eviction"] = eviction
    result["mode_lines"] = mode_lines(tag)
    result["ledger"] = grep_final_counters(tag) or {}
    result["tracebacks"] = grep_tracebacks(tag)
    result["warnings"] = [
        w for w in grep_warnings(tag) if driver._WARN_NO_SESSION not in w
    ]
    out = LOGDIR / f"ac_{tag}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"[ac] {tag}: wrote {out}")
    return result


def _divergence(a: str, b: str) -> int:
    """First character index at which two completions differ, -1 if equal."""
    if a == b:
        return -1
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return min(len(a), len(b))


def _agreement(left: dict, right: dict) -> tuple[str, str]:
    """Compare two passes question by question.

    Args:
        left: A pass document, used as the reference.
        right: The pass document to compare against it.

    Returns:
        (identical completions as "k/n", identical extracted answers as
        "k/n").
    """
    a, b = left["completions"], right["completions"]
    same_text = sum(1 for x, y in zip(a, b) if x == y)
    same_answer = sum(
        1 for x, y in zip(left["answers"], right["answers"]) if x == y
    )
    return f"{same_text}/{len(a)}", f"{same_answer}/{len(a)}"


def _l1_table(docs: list[dict]) -> None:
    """Print what L1 did during each pass.

    Args:
        docs: Result documents, in the order they should print.
    """
    print(
        "\nL1 over each pass, sampled once a second. Under a budget above the "
        "eviction\nwatermark the fill sawtooths, so the peak and the trough "
        "say more than the end.\n"
    )
    print(
        f"{'tag':26} {'pass':7} {'budget':>8} {'peak':>6} {'min':>6} "
        f"{'end':>6} {'peak GB':>8} {'objects':>8} {'evictions':>10}"
    )
    for doc in docs:
        for name in _PASSES:
            p = doc["passes"].get(name)
            if not p:
                continue
            trace = p.get("l1_trace", {})
            if not trace.get("n_samples"):
                print(
                    f"{doc['tag']:26} {name:7} {doc['l1_gb']:>7d}G "
                    f"{'-':>6} {'-':>6} {p['l1_usage_ratio']:>6.3f} "
                    f"{'-':>8} {p['objects_after']:>8d} {'-':>10}"
                )
                continue
            print(
                f"{doc['tag']:26} {name:7} {doc['l1_gb']:>7d}G "
                f"{trace['peak_ratio']:>6.3f} {trace['min_ratio']:>6.3f} "
                f"{trace['final_ratio']:>6.3f} {trace['peak_used_gb']:>8.1f} "
                f"{trace['peak_objects']:>8d} {p['eviction_cycles']:>10d}"
            )


def _sign_test_p(deltas: list[float]) -> float:
    """Two-sided sign-test p for "the paired differences are centred on zero".

    The pairing is what makes this the right test: both configs answer the
    same 120 questions, so each question contributes one difference and the
    question-to-question spread -- which dwarfs the effect -- cancels. Only
    the signs are used, so a few outsized differences cannot carry the
    verdict.

    Args:
        deltas: Paired differences; exact zeros are dropped, as the test
            requires.

    Returns:
        The probability of a split at least this lopsided under the null,
        or 1.0 if every pair tied.
    """
    signs = [d for d in deltas if d != 0.0]
    if not signs:
        return 1.0
    n = len(signs)
    positive = sum(1 for d in signs if d > 0)
    # Exact binomial: sum the tail at least as far from n/2 as the observed
    # count, on both sides. n is 120 here, so this is cheap.
    def choose(k: int) -> float:
        result = 1.0
        for i in range(k):
            result = result * (n - i) / (i + 1)
        return result

    observed = abs(positive - n / 2)
    total = 2.0**n
    tail = sum(
        choose(k) for k in range(n + 1) if abs(k - n / 2) >= observed
    )
    return min(1.0, tail / total)


def _ttft_table(docs: list[dict]) -> None:
    """Compare per-question TTFT across configs, question by question.

    Each connector run is compared against the `off` run that shares its
    question count, concurrency, L1 budget and repeat index -- the same
    matching the score comparison uses.

    Args:
        docs: Result documents.
    """
    references = {
        (d["n_questions"], d["concurrency"], d["l1_gb"], d["rep"]): d
        for d in docs
        if d["config"] == "off"
    }
    print(
        "\nTTFT on the `cached` pass, paired by question against the matching "
        "`off` run.\nThe pairing removes the prompt-length spread, which is "
        "far larger than the effect;\n`p` is a two-sided sign test on the 120 "
        "paired differences.\n"
    )
    print(
        f"{'tag':26} {'against':26} {'p50 off':>8} {'p50':>8} {'d p50':>8} "
        f"{'d mean':>8} {'faster':>8} {'p':>7}"
    )
    for doc in docs:
        cached = doc["passes"].get("cached")
        reference = references.get(
            (doc["n_questions"], doc["concurrency"], doc["l1_gb"], doc["rep"])
        )
        if not cached or not reference or doc["tag"] == reference["tag"]:
            continue
        base = reference["passes"].get("cached")
        if not base or "ttft_ms" not in cached or "ttft_ms" not in base:
            continue
        pairs = list(zip(base["ttft_ms"], cached["ttft_ms"], strict=True))
        deltas = [b - a for a, b in pairs]
        faster = sum(1 for d in deltas if d < 0)
        print(
            f"{doc['tag']:26} {reference['tag']:26} "
            f"{statistics.median(base['ttft_ms']):>8.1f} "
            f"{statistics.median(cached['ttft_ms']):>8.1f} "
            f"{statistics.median(deltas):>+7.1f}ms "
            f"{statistics.fmean(deltas):>+7.1f}ms "
            f"{f'{faster}/{len(deltas)}':>8} {_sign_test_p(deltas):>7.3f}"
        )


def table() -> int:
    """Print every saved result: scores, the two agreements, and the guards.

    Returns:
        Process exit status; always 0. Like every other layer-2b table this
        one asserts nothing -- the pass/fail authority is #18's oracle.
    """
    docs = [
        json.loads(path.read_text()) for path in sorted(LOGDIR.glob("ac_A_*.json"))
    ]
    if not docs:
        print("no results")
        return 0
    print(
        "\nGSM8K 20-shot, greedy. `cold` computes every prefill; `cached` "
        "resends the same\nprompts once the GPU pool has turned over, so its "
        "prefill comes from L1.\n"
    )
    header = (
        f"{'tag':26} {'pass':7} {'strict':>7} {'lenient':>8} {'ttft p50':>9} "
        f"{'ext':>6} {'apc':>6} {'covered':>8} {'objects':>8} {'l1 end':>7} "
        f"{'no ####':>8}"
    )
    print(header)
    for doc in docs:
        for name in _PASSES:
            p = doc["passes"].get(name)
            if not p:
                continue
            print(
                f"{doc['tag']:26} {name:7} {p['score_strict']:7.3f} "
                f"{p['score_lenient']:8.3f} {p['ttft_p50']:9.1f} "
                f"{_hit(p['counters']):6.3f} {_apc(p['counters']):6.3f} "
                f"{_covered(p['counters']):8.3f} "
                f"{p['objects_after']:8d} {p['l1_usage_ratio']:7.3f} "
                f"{p['no_final_line']:8d}"
            )
    _l1_table(docs)
    _ttft_table(docs)
    print("\nWithin a config: cached vs cold (same engine, same prompts; the")
    print("only difference is that the KV was fetched instead of computed).\n")
    print(f"{'tag':22} {'strict delta':>13} {'same text':>11} {'same answer':>12}")
    for doc in docs:
        cold, cached = doc["passes"].get("cold"), doc["passes"].get("cached")
        if not cold or not cached:
            continue
        text, answer = _agreement(cold, cached)
        delta = cached["score_strict"] - cold["score_strict"]
        print(f"{doc['tag']:22} {delta:+13.3f} {text:>11} {answer:>12}")
    print(
        "\nAcross repeats of one config (`cached` pass): the noise floor. Two "
        "runs of the\nsame config on the same prompts differ only by batch "
        "composition and engine\nstate, so whatever they disagree on cannot be "
        "attributed to a policy.\n"
    )
    print(f"{'config n conc':22} {'strict':>13} {'same text':>11} {'same answer':>12}")
    groups: dict[tuple[str, int, int], list[dict]] = {}
    for doc in docs:
        if doc["passes"].get("cached"):
            key = (doc["config"], doc["n_questions"], doc["concurrency"])
            groups.setdefault(key, []).append(doc)
    for (config, questions, conc), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda d: d["rep"])
        first, second = members[0]["passes"]["cached"], members[1]["passes"]["cached"]
        text, answer = _agreement(first, second)
        scores = f"{first['score_strict']:.3f}/{second['score_strict']:.3f}"
        print(f"{f'{config} n{questions} c{conc}':22} {scores:>13} {text:>11} {answer:>12}")
    # Each connector run is compared against the `off` run that shares its
    # question count, concurrency, L1 budget and repeat index -- comparing a
    # serial run against a concurrent reference, or a squeezed L1 against a
    # roomy one, would fold two differences into one number.
    references = {
        (d["n_questions"], d["concurrency"], d["l1_gb"], d["rep"]): d
        for d in docs
        if d["config"] == "off" and d["passes"].get("cached")
    }
    print(
        "\nAcross configs, `cached` pass against the matching `off` run "
        "(recompute\nreference). Exact text identity is a diagnostic, not the "
        "criterion -- see the\nmodule docstring.\n"
    )
    print(
        f"{'tag':26} {'against':26} {'strict delta':>13} {'same text':>11} "
        f"{'same answer':>12} {'divergence p50':>15}"
    )
    for doc in docs:
        cached = doc["passes"].get("cached")
        reference = references.get(
            (doc["n_questions"], doc["concurrency"], doc["l1_gb"], doc["rep"])
        )
        if not cached or not reference or doc["tag"] == reference["tag"]:
            continue
        base = reference["passes"]["cached"]
        text, answer = _agreement(base, cached)
        positions = sorted(
            _divergence(x, y)
            for x, y in zip(base["completions"], cached["completions"])
            if x != y
        )
        median = positions[len(positions) // 2] if positions else -1
        delta = cached["score_strict"] - base["score_strict"]
        print(
            f"{doc['tag']:22} {reference['tag']:22} {delta:+13.3f} {text:>11} "
            f"{answer:>12} {median:15d}"
        )
    print("\nStore ledgers:")
    for doc in docs:
        if doc["ledger"]:
            print(f"  {doc['tag']:22} {doc['ledger']}")
    print("\nGuards (a violated guard makes that row's score meaningless):")
    for doc in docs:
        cached = doc["passes"].get("cached")
        if not cached:
            continue
        apc, ext = _apc(cached["counters"]), _hit(cached["counters"])
        notes = []
        if apc > 0.05:
            notes.append(f"apc {apc:.3f} > 0.05: the GPU served the prefill")
        if doc["config"] != "off" and not ext > 0.5:
            notes.append(f"ext {ext:.3f} <= 0.5: little came from L1")
        if cached["l1_usage_ratio"] >= doc["eviction"]["trigger_watermark"]:
            notes.append(f"l1 {cached['l1_usage_ratio']:.3f} at the watermark")
        if doc["tracebacks"]:
            notes.append(f"{len(doc['tracebacks'])} tracebacks")
        print(f"  {doc['tag']:22} {'; '.join(notes) if notes else 'clean'}")
    return 0


def main() -> int:
    """Dispatch the command line. Returns the process exit status."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    command = sys.argv[1]
    if command == "run":
        run(
            sys.argv[2],
            int(sys.argv[3]) if len(sys.argv) > 3 else 0,
            int(sys.argv[4]) if len(sys.argv) > 4 else 0,
            int(sys.argv[5]) if len(sys.argv) > 5 else 0,
            int(sys.argv[6]) if len(sys.argv) > 6 else 0,
        )
        return 0
    if command == "table":
        return table()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
