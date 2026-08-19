#!/usr/bin/env python3
"""Long documents with a hot set the GPU keeps resident, and a cold tail.

This is the regime the whole feature is aimed at, and no earlier one has it.
Every workload measured so far has been uniform: each prompt is as likely to
be reused as any other, so "the GPU will serve this reuse itself" is either
true for everything or false for everything, and gate 1 has no decision to
make. Here the access distribution is skewed on purpose:

    hot    3 documents, 75% of the requests. The sequence is a fixed rotation
           -- each hot document, then one cold document, repeat -- so two
           requests for the same hot document are always four apart with
           exactly one new document in between. Its blocks therefore never
           approach the free queue's head and the GPU serves every reuse.
           Storing them to L1 buys nothing.
    cold  11 documents, 25% of the requests. Evicted from the GPU long before
           they come back, so L1 is the only thing that can serve them.

Gate 1's claim is exactly that it can tell those two apart. Eager cannot: it
writes both.

Sizes, from 144 KiB per token (36 layers * 8 KV heads * 128 dims * 2 tensors
* 2 bytes) and 20,000-token documents at 2.75 GiB each:

    GPU pool       20.0 GiB   9102 blocks   7.3 documents
    hot set         8.2 GiB   3 documents
    cold set       30.2 GiB   11 documents
    distinct       38.5 GiB
    L1             40.0 GiB   watermark 0.8 -> 32 GiB usable

The pool has to hold the hot set, the 2 in-flight requests and the one cold
document that separates two rotations: 3 + 2 + 1 = 6 documents against the
7.3 it holds. `residency_budget` checks that before every run, because a hot
set the pool cannot keep is not a hot set and the measurement would be
vacuous.

The two policies land on opposite sides of the L1 watermark: storing the cold
set alone is 0.755 of L1 and never triggers eviction, storing everything is
0.96 and triggers it continuously.

What this measures, in the order the claims should be read:

1. `covered` must match. Declining a store is only free if the GPU really
   does serve the reuse; if lazy's coverage drops, gate 1 was wrong.
2. Bytes written to L1, and the object count. This is the saving, and it
   should be about the size of the hot set.
3. L1 fill and eviction cycles. Lazy should stay under the watermark and
   never evict; eager should cross it and evict.

A prediction was stated before the first run, so that a null result would not
be read as a failure: eager's *hit rate* need not collapse, because L1 evicts
by last access and eager's hot entries are written once and then never read,
which makes them their own first victims -- self-correcting waste.

**That prediction was wrong, and it is left here because being wrong about it
is the finding.** Across three repeats eager's external hit rate was 0.000,
0.000 and 0.000 while lazy's was 0.641, 0.669 and 0.778. Eager does not merely
waste capacity, it drives itself into a thrash: it enters the query phase with
a smaller surviving cache, misses, re-stores the whole document, evicts more
(14-15 cycles against lazy's 3-6), and misses again. Lazy's deferral makes L1
fill more slowly, so it crosses the watermark later, keeps more of the warm-up,
hits, and therefore has nothing to re-store. The two policies settle into
different attractors from the same workload.

The consequence is worth stating plainly: in this regime eager is *worse than
running no external cache at all* -- 41-43s for the query phase against off's
39s -- while lazy takes 28-31s.

Usage:
    python longdoc.py run <off|eager|lazy> [rep] [l1_gb]
    python longdoc.py table
"""

# Standard
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

# First Party
import driver
from driver import (
    LOGDIR,
    cache_object_count,
    grep_final_counters,
    grep_ledgers,
    grep_tracebacks,
    grep_warnings,
    mode_lines,
    server_status,
    teardown,
    wait_for,
)
from accuracy import CONFIGS, L1Sampler, eviction_cycles
from workload import MODEL, TP_SIZE, _CONNECTOR_CONFIGS, _metrics, start_server_sized

#: KV bytes per token for this model: 36 layers * 8 KV heads * 128 dims * 2
#: tensors * 2 bytes. Every size below is derived from it.
KV_BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2

#: Document length in tokens. The prompts are a repeated single-token word, so
#: the word count is the token count to within the prefix; the run records the
#: engine's own count for the first document as a check.
DOCUMENT_TOKENS = 20000

#: Documents whose reuse the GPU can serve by itself, and documents whose
#: reuse it cannot. See the module docstring for how the sizes were chosen.
HOT_DOCUMENTS = 3
COLD_DOCUMENTS = 11

#: Requests in the query phase. 120 leaves each cold document about three
#: requests, which is what makes its retrieval measurable at all.
QUERY_REQUESTS = 120

#: Cold requests are spaced this many hot requests apart, which fixes the hot
#: share at HOT_DOCUMENTS / (HOT_DOCUMENTS + 1) = 75%.
HOT_RUN_LENGTH = HOT_DOCUMENTS

#: Generated tokens per request. Short on purpose: this regime is about the
#: prefill and what serves it.
OUTPUT_LEN = 32

#: Requests in flight. 2, because the pool has to hold the hot set, the
#: in-flight requests and one cold document at once; see residency_budget.
INFLIGHT = 2

#: GPU pool in 16-token blocks: 20 GiB.
POOL_BLOCKS = (20 << 30) // KV_BYTES_PER_TOKEN // 16

#: L1 budget in GiB. Chosen so the cold set alone stays under the 0.8
#: watermark and the whole distinct set does not.
L1_GB = 40

#: Context ceiling: one document plus its generation plus slack.
MAX_MODEL_LEN = 22528

_PHASES = ("warmup", "query")

#: Ledger fields that report a current level rather than a running total, and
#: so must not be differenced when a phase's ledger delta is taken.
_LEDGER_GAUGES = frozenset({"pending"})


def documents() -> list[str]:
    """Build the document set: hot first, then cold.

    Each document is its own index followed by a run of a single repeated
    word, so no document is a prefix of another and the token count is
    predictable.

    Returns:
        HOT_DOCUMENTS + COLD_DOCUMENTS prompts, hot ones first.
    """
    return [
        f"{index} " + " ".join(["hi"] * DOCUMENT_TOKENS)
        for index in range(HOT_DOCUMENTS + COLD_DOCUMENTS)
    ]


def residency_budget() -> tuple[float, float]:
    """What the hot set needs from the GPU pool, against what the pool holds.

    A hot document stays resident only if, between two of its own requests,
    the pool is not asked for more than it can hold alongside it. Over one
    cycle of the sequence the pool has to carry the whole hot set, the
    in-flight requests, and the single cold document that separates two
    cycles.

    A random access sequence cannot promise this: its gap between two
    requests for the same hot document has a tail, and one long gap evicts
    the document the experiment depends on. That is why the sequence below is
    a fixed rotation rather than a draw.

    Returns:
        (documents the sequence needs resident, documents the pool holds).
    """
    needed = HOT_DOCUMENTS + INFLIGHT + 1
    held = POOL_BLOCKS * 16 / DOCUMENT_TOKENS
    return needed, held


def query_sequence() -> list[int]:
    """The query phase's document indices, in send order.

    A fixed rotation: every hot document once, then one cold document, and
    repeat. Two requests for the same hot document are therefore always
    HOT_DOCUMENTS + 1 apart with exactly one new document in between, which
    is what `residency_budget` checks against the pool.

    Returns:
        QUERY_REQUESTS document indices.

    Raises:
        ValueError: if the pool cannot hold what the rotation needs resident,
            which would make the hot set cold and the measurement vacuous.
    """
    needed, held = residency_budget()
    if needed > held:
        raise ValueError(
            f"the rotation needs {needed:.1f} documents resident but the pool "
            f"holds {held:.1f}; lower HOT_DOCUMENTS or INFLIGHT"
        )
    sequence: list[int] = []
    cold_cursor = 0
    while len(sequence) < QUERY_REQUESTS:
        sequence.extend(range(HOT_RUN_LENGTH))
        sequence.append(HOT_DOCUMENTS + cold_cursor % COLD_DOCUMENTS)
        cold_cursor += 1
    return sequence[:QUERY_REQUESTS]


def start_engine(scenario: str, config: str) -> subprocess.Popen:
    """Start `vllm serve` sized for this regime.

    Separate from `workload.start_engine` because that one fixes the context
    at 8192 tokens, which is under half of one document here.

    Args:
        scenario: Names the engine's log file.
        config: One of CONFIGS. `off` gets no KV connector.

    Returns:
        The running engine process.

    Raises:
        RuntimeError: if the engine does not come up inside the timeout.
    """
    driver._running_model = MODEL
    log = open(LOGDIR / f"{scenario}_vllm.log", "w")
    env = dict(
        os.environ,
        PYTHONPATH=driver.REPO,
        CUDA_VISIBLE_DEVICES=driver.GPU,
        VLLM_SERVER_DEV_MODE="1",
        CPATH=driver.CPATH,
        PYTHONFAULTHANDLER="1",
    )
    cmd = [
        driver.VLLM, "serve", MODEL,
        "--port", str(driver.VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.75",
        "--num-gpu-blocks-override", str(POOL_BLOCKS),
    ]
    if TP_SIZE > 1:
        cmd += ["--tensor-parallel-size", str(TP_SIZE)]
    if config != "off":
        kv_cfg = {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://127.0.0.1",
                "lmcache.mp.port": driver.MP_PORT,
                **_CONNECTOR_CONFIGS[config],
            },
        }
        cmd += ["--kv-transfer-config", json.dumps(kv_cfg)]
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=driver.REPO
    )
    try:
        wait_for(f"http://127.0.0.1:{driver.VLLM_PORT}/health", 900, "vllm")
    except Exception:
        teardown([proc])
        raise
    return proc


def generate(prompt: str) -> tuple[float, int]:
    """Stream one greedy completion, timing the first token.

    Args:
        prompt: The document text.

    Returns:
        (time to first token in ms, generated character count).

    Raises:
        RuntimeError: if the stream carried no token at all.
    """
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": OUTPUT_LEN,
            "temperature": 0,
            "seed": 0,
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
    chars = 0
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if not first:
                first = time.time()
            chars += len(json.loads(line[6:])["choices"][0]["text"])
    if not first:
        raise RuntimeError("stream produced no tokens")
    return (first - start) * 1000.0, chars


def _stats(values: list[float]) -> dict[str, float]:
    """Mean and percentiles of a TTFT list in ms, empty dict if there are none."""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": statistics.median(ordered),
        "p90_ms": ordered[int(len(ordered) * 0.9)],
        "max_ms": ordered[-1],
    }


def run_phase(name: str, texts: list[str], indices: list[int], tag: str) -> dict:
    """Send one phase's requests and collect everything measured around it.

    Args:
        name: Phase name, one of _PHASES.
        texts: The document set.
        indices: Document indices to send, in order.
        tag: The run tag, so the phase can read its server and engine logs.

    Returns:
        A dict with per-request TTFTs split by hot and cold, the vLLM counter
        deltas, the L1 trace, the eviction cycles and the ledger delta.
    """
    before = _metrics()
    objects_before = cache_object_count()
    evictions_before = eviction_cycles(tag)
    ledger_before = grep_final_counters(tag) or {}
    sampler = L1Sampler()
    sampler.start()
    start = time.time()
    with ThreadPoolExecutor(max_workers=INFLIGHT) as pool:
        results = list(pool.map(generate, (texts[i] for i in indices)))
    elapsed = time.time() - start
    # Stores are asynchronous; without this the object count and the L1 fill
    # read the middle of the drain rather than its end.
    time.sleep(15)
    after = _metrics()
    trace = sampler.stop()
    l1 = server_status()["storage_manager"]["l1_manager"]
    ledger_after = grep_final_counters(tag) or {}
    ttfts = [ttft for ttft, _ in results]
    hot = [t for t, i in zip(ttfts, indices, strict=True) if i < HOT_DOCUMENTS]
    cold = [t for t, i in zip(ttfts, indices, strict=True) if i >= HOT_DOCUMENTS]
    out = {
        "requests": len(indices),
        "seconds": elapsed,
        "ttft_ms": ttfts,
        "indices": indices,
        "all": _stats(ttfts),
        "hot": _stats(hot),
        "cold": _stats(cold),
        "counters": {k: after[k] - before[k] for k in before},
        "objects_before": objects_before,
        "objects_after": cache_object_count(),
        "l1_usage_ratio": float(l1["memory_usage_ratio"]),
        "l1_used_gb": float(l1["memory_used_bytes"]) / (1 << 30),
        "l1_trace": trace,
        "eviction_cycles": eviction_cycles(tag) - evictions_before,
        # Deltas for the cumulative counters; `pending` is a gauge -- the
        # queue depth right now -- so differencing it produces nonsense (a
        # phase that drained two ops would report -2 "pending"). It is carried
        # through as the absolute reading it is.
        "ledger": {
            key: (value if key in _LEDGER_GAUGES else value - ledger_before.get(key, 0))
            for key, value in ledger_after.items()
        },
    }
    print(
        f"[ld] {name}: {len(indices)} req in {elapsed:.0f}s  "
        f"ttft hot {out['hot'].get('p50_ms', float('nan')):.0f}ms "
        f"cold {out['cold'].get('p50_ms', float('nan')):.0f}ms  "
        f"ext {_hit(out['counters']):.3f} apc {_apc(out['counters']):.3f} "
        f"cov {_covered(out['counters']):.3f}  "
        f"obj {objects_before}->{out['objects_after']} "
        f"l1 {trace.get('peak_ratio', float('nan')):.3f} peak / "
        f"{out['l1_usage_ratio']:.3f} end, {out['eviction_cycles']} evictions"
    )
    return out


def run(config: str, rep: int = 0, l1_gb: int = 0) -> dict:
    """One measurement point: boot, warm up, query, collect, tear down.

    Args:
        config: One of CONFIGS.
        rep: Repeat index.
        l1_gb: Override L1_GB; 0 keeps it. Raising it above the distinct
            working set takes eviction out entirely, which is the control
            that says whether a difference between policies came from
            capacity pressure or from somewhere else.

    Returns:
        The result document, also written to logs/ld_<tag>.json.

    Raises:
        KeyError: if config is unknown.
    """
    global L1_GB
    if config not in CONFIGS:
        raise KeyError(f"unknown config {config!r}; known: {CONFIGS}")
    if l1_gb:
        L1_GB = l1_gb
    # The L1 budget joins the tag only when it is not the 40 GiB the first
    # round used, so those results keep the names they were recorded under.
    budget = "" if L1_GB == 40 else f"_l{L1_GB}"
    tag = f"L_{config}_h{HOT_DOCUMENTS}c{COLD_DOCUMENTS}{budget}_{rep}"
    texts = documents()
    per_document_gib = DOCUMENT_TOKENS * KV_BYTES_PER_TOKEN / (1 << 30)
    result: dict = {
        "config": config, "tag": tag, "rep": rep, "model": MODEL,
        "hot_documents": HOT_DOCUMENTS, "cold_documents": COLD_DOCUMENTS,
        "document_tokens": DOCUMENT_TOKENS, "query_requests": QUERY_REQUESTS,
        "hot_run_length": HOT_RUN_LENGTH, "inflight": INFLIGHT, "output_len": OUTPUT_LEN,
        "pool_blocks": POOL_BLOCKS, "l1_gb": L1_GB,
        "tensor_parallel_size": TP_SIZE,
        "pool_gib": POOL_BLOCKS * 16 * KV_BYTES_PER_TOKEN / (1 << 30),
        "hot_gib": HOT_DOCUMENTS * per_document_gib,
        "cold_gib": COLD_DOCUMENTS * per_document_gib,
    }
    print(
        f"[ld] === {tag}: hot {result['hot_gib']:.1f} GiB + cold "
        f"{result['cold_gib']:.1f} GiB through a {result['pool_gib']:.1f} GiB "
        f"pool and {L1_GB} GiB of L1"
    )
    server = start_server_sized(tag, L1_GB)
    try:
        engine = start_engine(tag, config)
    except Exception:
        teardown([server])
        raise
    try:
        result["prompt_tokens"] = driver.prompt_tokens(texts[0])
        result["phases"] = {
            "warmup": run_phase("warmup", texts, list(range(len(texts))), tag),
            "query": run_phase("query", texts, query_sequence(), tag),
        }
    finally:
        teardown([engine, server])
    result["rc_engine"] = engine.returncode
    result["rc_server"] = server.returncode
    result["mode_lines"] = mode_lines(tag)
    result["ledger"] = grep_final_counters(tag) or {}
    result["ledger_lines"] = len(grep_ledgers(tag))
    result["tracebacks"] = grep_tracebacks(tag)
    result["warnings"] = [
        w for w in grep_warnings(tag) if not driver._is_sessionless_warning(w)
    ]
    out = LOGDIR / f"ld_{tag}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"[ld] {tag}: wrote {out}  ledger {result['ledger']}")
    return result


def _hit(counters: dict[str, float]) -> float:
    """External hit rate over a phase's counter deltas, nan if unqueried."""
    queries = counters.get("ext_queries", 0.0)
    return counters["ext_hits"] / queries if queries else float("nan")


def _apc(counters: dict[str, float]) -> float:
    """vLLM prefix-cache hit rate over a phase's counter deltas."""
    queries = counters.get("apc_queries", 0.0)
    return counters["apc_hits"] / queries if queries else float("nan")


def _covered(counters: dict[str, float]) -> float:
    """Fraction of queried prompt tokens served by any cache, nan if none."""
    queries = counters.get("apc_queries", 0.0)
    if not queries:
        return float("nan")
    return (counters["apc_hits"] + counters.get("ext_hits", 0.0)) / queries


def table() -> int:
    """Print every saved result. Returns the process exit status."""
    docs = [
        json.loads(path.read_text()) for path in sorted(LOGDIR.glob("ld_L_*.json"))
    ]
    if not docs:
        print("no results")
        return 0
    first = docs[0]
    print(
        f"\n{first['hot_documents']} hot + {first['cold_documents']} cold "
        f"documents of {first['document_tokens']} tokens "
        f"({first['hot_gib']:.1f} + {first['cold_gib']:.1f} GiB) through a "
        f"{first['pool_gib']:.1f} GiB pool and {first['l1_gb']} GiB of L1.\n"
        f"Warm-up sends each document once; the query phase sends "
        f"{first['query_requests']} requests, {first['query_requests'] * HOT_DOCUMENTS // (HOT_DOCUMENTS + 1)} of them "
        f"to the hot set.\n"
    )
    print(
        f"{'tag':26} {'phase':7} {'ttft hot':>9} {'ttft cold':>10} "
        f"{'ext':>6} {'apc':>6} {'covered':>8}"
    )
    for doc in docs:
        for name in _PHASES:
            p = doc["phases"][name]
            print(
                f"{doc['tag']:26} {name:7} "
                f"{p['hot'].get('p50_ms', float('nan')):>8.0f}ms "
                f"{p['cold'].get('p50_ms', float('nan')):>9.0f}ms "
                f"{_hit(p['counters']):>6.3f} {_apc(p['counters']):>6.3f} "
                f"{_covered(p['counters']):>8.3f}"
            )
    print(
        "\nWhat each policy wrote, and what that cost L1. The saving should be "
        "about\nthe size of the hot set; the eviction column is where the "
        "watermark crossing shows.\n"
    )
    print(
        f"{'tag':26} {'phase':7} {'objects':>8} {'l1 GB':>7} {'peak':>6} "
        f"{'end':>6} {'evictions':>10}"
    )
    for doc in docs:
        for name in _PHASES:
            p = doc["phases"][name]
            trace = p["l1_trace"]
            print(
                f"{doc['tag']:26} {name:7} {p['objects_after']:>8d} "
                f"{p['l1_used_gb']:>7.1f} "
                f"{trace.get('peak_ratio', float('nan')):>6.3f} "
                f"{p['l1_usage_ratio']:>6.3f} {p['eviction_cycles']:>10d}"
            )
    print("\nStore ledgers, per phase:")
    for doc in docs:
        for name in _PHASES:
            ledger = doc["phases"][name]["ledger"]
            if ledger:
                print(f"  {doc['tag']:26} {name:7} {ledger}")
    print("\nTracebacks and warnings:")
    for doc in docs:
        notes = []
        if doc["tracebacks"]:
            notes.append(f"{len(doc['tracebacks'])} tracebacks")
        if doc["warnings"]:
            notes.append(f"{len(doc['warnings'])} warnings")
        print(f"  {doc['tag']:26} {'; '.join(notes) if notes else 'clean'}")
    return 0


def main() -> int:
    """Dispatch the command line. Returns the process exit status."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if sys.argv[1] == "run":
        run(
            sys.argv[2],
            int(sys.argv[3]) if len(sys.argv) > 3 else 0,
            int(sys.argv[4]) if len(sys.argv) > 4 else 0,
        )
        return 0
    if sys.argv[1] == "table":
        return table()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
