#!/usr/bin/env python3
"""Layer-2b workload harness for lazy offloading (#36).

Usage:
    workload.py run <off|eager|lazy> <reuse|noreuse> [rate]
    workload.py table                    tabulate every saved result
    workload.py matrix [rate]            every config x every workload

Layer 1 and the layer-2 oracle (#18) answer "does it behave", on a 0.6B
model with a synthetic corpus and no timing claim at all. This file answers
"is it worth enabling", which needs three things they deliberately do not
have: a model whose KV is large enough for offload to save real work, a
request stream rather than a scripted sequence, and CUDA graphs left on.

Why an 8B model. Qwen3-8B carries 36 layers x 8 KV heads x 128 dims, so
147456 bytes -- 144 KB -- of KV per token, against roughly 0.6 KB for
Qwen3-0.6B. Fetching a cached 2048-token prefix moves ~295 MB over the bus
in single-digit milliseconds; recomputing it is ~33 TFLOPs of prefill. The
gap between those two numbers is the entire value proposition, and at 0.6B
it is far inside the noise -- which is why the oracle measures no time.

Why not `driver.start_vllm`. Every layer-1 scenario runs `--enforce-eager`
to keep the engine's numerics and shutdown deterministic. That disables
CUDA graphs, which inflates decode latency and would make every TPOT here
a measurement of the harness rather than of the feature. This file starts
its own engine on the compiled path.

Two workloads, because "worth enabling" has two halves:

- `reuse`: many requests over a few long shared prefixes, against a pool
  too small to hold them all. This is where a KV cache can pay: a prefix
  evicted between two uses is either refetched or recomputed.
- `noreuse`: distinct prompts of the same length. Nothing can ever hit, so
  every store is pure overhead. This is the regression side, and it is the
  half a benchmark chosen to flatter the feature would omit.
"""

import json
import math
import os
import subprocess
import sys
import time

import driver
from driver import (
    cache_object_count,
    complete,
    grep_final_counters,
    grep_retrieved,
    grep_tracebacks,
    grep_warnings,
    long_prompt,
    mode_lines,
    server_status,
    teardown,
    wait_for,
)

LOGDIR = driver.LOGDIR
MODEL = "Qwen/Qwen3-8B"
TP_SIZE = int(os.environ.get("SMOKE_TP", "1"))
if TP_SIZE < 1:
    raise ValueError("SMOKE_TP must be at least 1")

#: L1 (host memory) budget for the MP server. At 144 KB/token this is about
#: 450k tokens, comfortably more than any working set below; the host has
#: 2 TB. Sized generously on purpose -- an L1 that evicts would put a
#: second, unmeasured eviction policy inside the measurement.
L1_GB = 64

#: Engine context length. Longest request below is 2048 + 256 prompt plus
#: 128 output.
MAX_MODEL_LEN = 8192

#: KV pool, in 16-token blocks, and the client-side concurrency ceiling.
#: These two are one decision: the pool has to be too small to keep the
#: shared prefixes resident, yet large enough for the requests in flight, or
#: the run measures the scheduler recovering from preemption instead of the
#: connector.
#:
#: The arithmetic, at 2432 tokens per request (2048 prefix + 256 suffix +
#: 128 output) and 152 blocks each:
#:
#:   in flight    4 x 152  =  608 blocks
#:   pool                  = 1024 blocks (16384 tokens, 2.25 GB)
#:   left for cache        =  416 blocks = 6656 tokens = ~3 of the 8 prefixes
#:
#: So five of eight prefixes are out of the GPU cache at any moment, and
#: most revisits miss it. That miss is the whole experiment: `off` has to
#: recompute 2048 tokens, the connector configs can fetch them.
#:
#: Without `--num-gpu-blocks-override` the engine sizes the pool from free
#: memory -- tens of thousands of blocks on an H200 -- nothing is ever
#: evicted, and all three configs measure the same thing. The first version
#: of this file used 1536 blocks, which is *larger* than the 16384-token
#: prefix working set and had exactly that defect.
#:
#: That reasoning is right for measuring the *retrieval* side and backwards
#: for measuring gate 1. Gate 1 stores a chunk iff its GPU copy will be
#: evicted; its value is the stores it *skips*, i.e. case 1 of the decision
#: model ("GPU serves every reuse, the copy is worth 0"). A pool below the
#: working set evicts nearly everything, leaves gate 1 almost nothing to
#: skip, and drives lazy to eager's store count -- measured: 116 vs 128
#: objects at 1024 blocks. So the pool is an *axis*, not a constant:
#:
#:   1024 blocks -- below the 1024-block prefix working set; near-total
#:                  eviction. Gate 1 has no room. Expect lazy == eager.
#:   2048 blocks -- prefixes (1024) + in-flight (608) fit with headroom.
#:                  Expect lazy to skip most stores.
#:   4096 blocks -- comfortably resident. Expect lazy objects ~ 0 and lazy
#:                  latency ~ `off`, while eager still pays every store.
#:
#: 0 means "no override": let the engine size the pool itself.
POOL_BLOCKS = 1024
MAX_CONCURRENCY = 4

#: Requests per benchmark run, and the Poisson arrival rate they are sent
#: at. 64 requests at 2/s is a ~35 s window once concurrency is accounted
#: for: long enough for percentiles to mean something, short enough that
#: six of these plus six engine boots fit in a sitting.
NUM_PROMPTS = 64
DEFAULT_RATE = "2"

#: Shared-prefix workload geometry. Eight prefixes over 64 requests is 8
#: requests per prefix, so each prefix is revisited seven times after its
#: first, cold use.
_PREFIXES = 8
_PREFIX_LEN = 2048
_SUFFIX_LEN = 256
_OUTPUT_LEN = 128

#: `random` with a fixed seed generates the same prompts in every process,
#: so running it twice against one engine is a reuse workload whose reuse
#: distance is the whole first pass -- 64 requests, 147456 tokens, which is
#: 20 GB of KV against a 2.25 GB pool. Nothing survives in the GPU cache, so
#: pass 2 is the clean question: fetch the prefix, or recompute it?
#:
#: This exists because `prefix_repetition` cannot answer that. It emits its
#: requests grouped by prefix (`for _ in range(num_prefixes): for _ in
#: range(prompts_per_prefix)`, no shuffle), so every reuse is adjacent to its
#: producer and vLLM's own prefix cache always still holds it -- measured APC
#: hit 0.64 in all three configs. A workload where the GPU cache already wins
#: cannot show what an external cache is for.
WORKLOADS = {
    "reuse_far": [
        "--dataset-name", "random",
        "--random-input-len", str(_PREFIX_LEN + _SUFFIX_LEN),
        "--random-output-len", str(_OUTPUT_LEN),
        "--random-prefix-len", "0",
    ],
    "reuse": [
        "--dataset-name", "prefix_repetition",
        "--prefix-repetition-num-prefixes", str(_PREFIXES),
        "--prefix-repetition-prefix-len", str(_PREFIX_LEN),
        "--prefix-repetition-suffix-len", str(_SUFFIX_LEN),
        "--prefix-repetition-output-len", str(_OUTPUT_LEN),
    ],
    "noreuse": [
        "--dataset-name", "random",
        "--random-input-len", str(_PREFIX_LEN + _SUFFIX_LEN),
        "--random-output-len", str(_OUTPUT_LEN),
        "--random-prefix-len", "0",
    ],
}

#: Identical benchmark passes per workload. Only the last pass is compared:
#: for `reuse_far` the first pass is there to populate the cache, and its own
#: latencies are a cold-start measurement of no interest.
_PASSES = {"reuse_far": 2}

#: Connector settings per config; `off` is absent and gets no
#: `--kv-transfer-config` at all.
_CONNECTOR_CONFIGS = {
    "eager": {},
    "lazy": {
        "lmcache.mp.lazy_offload": True,
        "lmcache.mp.lazy_offload_policy": "EVICTION_AWARE",
        "lmcache.mp.lazy_offload_horizon_steps": float(
            os.environ.get("SMOKE_HORIZON", "2.0")
        ),
    },
}

#: Gate 1's lookahead, in scheduler steps, and the axis every layer-2 run
#: before this one left at its default. The policy emits an operation when a
#: block of it sits within `ceil(rate x horizon)` of the free-queue head, so
#: this is the whole lazy-vs-eager tradeoff in one number: larger drains
#: earlier (fewer `dropped_evicted`, coverage closer to eager, less filtered)
#: and smaller drains later. Matches LazyOffloadPolicyConfig.horizon_steps,
#: so passing it explicitly at this value reproduces every earlier run.
DEFAULT_HORIZON = float(os.environ.get("SMOKE_HORIZON", "2.0"))

CONFIGS = ("off", "eager", "lazy")

#: vLLM counters read before and after each run. Preemptions are here
#: because they change the meaning of every latency number: a pool tight
#: enough to preempt is measuring the scheduler's recovery, not the
#: connector's cost.
_METRICS = {
    "apc_queries": "vllm:prefix_cache_queries_total",
    "apc_hits": "vllm:prefix_cache_hits_total",
    "ext_queries": "vllm:external_prefix_cache_queries_total",
    "ext_hits": "vllm:external_prefix_cache_hits_total",
    "preemptions": "vllm:num_preemptions_total",
}

#: Not measured. NaN so an absent counter can never satisfy a comparison.
_UNMEASURED = math.nan


def start_server_sized(scenario: str, l1_gb: int) -> subprocess.Popen:
    """Start the LMCache MP server with an explicit L1 budget.

    `driver.start_server` fixes L1 at 8 GB, which is under two minutes of
    this workload's stores at 144 KB/token.

    Args:
        scenario: Names the server's log file.
        l1_gb: L1 (host memory) budget in GB.

    Returns:
        The running server process.
    """
    log = open(LOGDIR / f"{scenario}_server.log", "w")
    env = dict(os.environ, PYTHONPATH=driver.REPO, CUDA_VISIBLE_DEVICES=driver.GPU)
    proc = subprocess.Popen(
        [
            driver.PY, "-m", "lmcache.v1.multiprocess.http_server",
            "--host", "127.0.0.1", "--port", str(driver.MP_PORT),
            "--http-host", "127.0.0.1", "--http-port", str(driver.HTTP_PORT),
            "--l1-size-gb", str(l1_gb), "--eviction-policy", "LRU",
            "--script-allowed-imports", "hashlib",
            "--max-workers", "4",
        ],
        stdout=log, stderr=subprocess.STDOUT, env=env, cwd=driver.REPO,
    )
    wait_for(f"http://127.0.0.1:{driver.HTTP_PORT}/healthcheck", 60, "mp-server")
    return proc


def start_engine(
    scenario: str,
    config: str,
    pool: int = POOL_BLOCKS,
    horizon: float = DEFAULT_HORIZON,
) -> subprocess.Popen:
    """Start `vllm serve` for the 8B model on the compiled (graph) path.

    Args:
        scenario: Names the engine's log file.
        config: One of CONFIGS. `off` gets no KV connector.
        pool: KV blocks to force via `--num-gpu-blocks-override`. 0 leaves the
            engine to size the pool from free memory, which on this card means
            no eviction at all. See POOL_BLOCKS for why this is an axis.
        horizon: Gate 1's lookahead in scheduler steps; see DEFAULT_HORIZON.
            Passed only for `lazy`, the sole config with a policy to tune.

    Returns:
        The running engine process.
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
        "--gpu-memory-utilization", "0.5",
    ]
    if pool:
        cmd += ["--num-gpu-blocks-override", str(pool)]
    if TP_SIZE > 1:
        cmd += ["--tensor-parallel-size", str(TP_SIZE)]
    if config != "off":
        extra = dict(_CONNECTOR_CONFIGS[config])
        if config == "lazy":
            extra["lmcache.mp.lazy_offload_horizon_steps"] = horizon
        kv_cfg = {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://127.0.0.1",
                "lmcache.mp.port": driver.MP_PORT,
                **extra,
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


def _metrics() -> dict[str, float]:
    """Every counter in `_METRICS`, NaN for any the engine does not expose."""
    out = {}
    for key, name in _METRICS.items():
        try:
            out[key] = driver.vllm_metric(name)
        except RuntimeError:
            out[key] = _UNMEASURED
    return out


def run_bench(tag: str, workload: str, rate: str, concurrency: int) -> dict:
    """Run `vllm bench serve` against the live engine and return its report.

    Args:
        tag: Names the result and log files.
        workload: A key of WORKLOADS.
        rate: Requests per second, or "inf".
        concurrency: Client-side in-flight ceiling.

    Returns:
        The parsed benchmark JSON.

    Raises:
        RuntimeError: if the benchmark exits non-zero or writes no report.
    """
    out = LOGDIR / f"bench_{tag}.json"
    if out.exists():
        out.unlink()
    cmd = [
        driver.VLLM, "bench", "serve",
        "--backend", "vllm",
        "--model", MODEL,
        "--host", "127.0.0.1", "--port", str(driver.VLLM_PORT),
        "--num-prompts", str(NUM_PROMPTS),
        "--request-rate", rate,
        "--max-concurrency", str(concurrency),
        # Every request generates exactly _OUTPUT_LEN tokens, so decode work
        # is identical across configs. Without this an early EOS shortens a
        # request and moves the throughput numbers for reasons that have
        # nothing to do with the KV path.
        "--ignore-eos",
        "--seed", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--save-result", "--result-filename", str(out),
        "--disable-tqdm",
        *WORKLOADS[workload],
    ]
    log = LOGDIR / f"bench_{tag}.log"
    with open(log, "w") as fh:
        rc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=driver.REPO,
            env=dict(os.environ, PYTHONPATH=driver.REPO),
        ).returncode
    if rc != 0:
        raise RuntimeError(f"bench serve rc={rc}, see {log}")
    if not out.exists():
        raise RuntimeError(f"bench serve wrote no report, see {log}")
    return json.loads(out.read_text())


#: Benchmark fields carried into the comparison table.
_REPORT_KEYS = (
    "duration", "completed", "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_token_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p99_itl_ms",
    "mean_e2el_ms", "median_e2el_ms", "p99_e2el_ms",
)


def run(
    config: str,
    workload: str,
    rate: str,
    concurrency: int = MAX_CONCURRENCY,
    rep: int = 0,
    pool: int = POOL_BLOCKS,
    horizon: float = DEFAULT_HORIZON,
) -> dict:
    """One measurement point: boot, benchmark, collect, tear down.

    Args:
        config: One of CONFIGS.
        workload: A key of WORKLOADS.
        rate: Requests per second, or "inf".
        concurrency: Client-side in-flight ceiling. At 1 the client sends one
            request at a time, so TTFT carries no queueing delay and reads
            the prefill-or-fetch cost directly -- which is the quantity this
            file is actually about. At 4 it reads what a loaded server does
            with it, queueing included.
        rep: Repeat index. Distinguishes the result files of identical
            configurations so a cell can be measured more than once; a single
            run of a latency benchmark has no error bar, and the differences
            at stake here are tens of percent on 64 samples.
        pool: KV blocks forced on the engine, 0 for no override. This is the
            axis that decides whether gate 1 has anything to skip; see
            POOL_BLOCKS.
        horizon: Gate 1's lookahead in scheduler steps; see DEFAULT_HORIZON.
            Ignored by `off` and `eager`, which have no policy.

    Returns:
        The result document, also written to logs/wl_<tag>.json.

    Raises:
        KeyError: if config or workload is unknown.
    """
    if config not in CONFIGS:
        raise KeyError(f"unknown config {config!r}; known: {CONFIGS}")
    if workload not in WORKLOADS:
        raise KeyError(f"unknown workload {workload!r}; known: {tuple(WORKLOADS)}")
    passes = _PASSES.get(workload, 1)
    # `b<pool>` and not `p<pool>`: `_p<n>` is already the per-pass suffix of
    # the bench result files. Runs made before the pool became an axis have
    # no `b` field in their tag and recorded pool_blocks=1024; table() cells
    # by the recorded field, not the tag, so they still group with new 1024s.
    # The horizon suffix is omitted at the default so every run made before
    # the horizon became an axis keeps its filename; table() cells by the
    # recorded field, which those runs get from the .get() default.
    horizon_suffix = "" if horizon == DEFAULT_HORIZON else f"_h{horizon:g}"
    tag = (
        f"W_{config}_{workload}_r{rate}_c{concurrency}_b{pool}"
        f"{horizon_suffix}_{rep}"
    )
    result: dict = {
        "config": config, "workload": workload, "rate": rate, "tag": tag,
        "model": MODEL, "pool_blocks": pool, "horizon": horizon,
        "max_concurrency": concurrency, "rep": rep, "num_prompts": NUM_PROMPTS,
    }
    print(f"[wl] === {tag}")
    server = start_server_sized(tag, L1_GB)
    try:
        engine = start_engine(tag, config, pool, horizon)
    except Exception:
        teardown([server])
        raise
    try:
        reports = []
        retrieves_before_last = 0
        for p in range(passes):
            if p == passes - 1 and config != "off":
                # Retrievals are counted for the compared pass only; pass 1
                # of a two-pass workload retrieves whatever the previous
                # engine left, which is nothing, but the slice keeps the
                # reading honest if that ever changes.
                retrieves_before_last = len(grep_retrieved(tag))
            # Each pass sends the identical prompt set (fixed --seed), so
            # metrics are snapshotted per pass: pass 1's queries and hits are
            # the cold pass and must not be summed into pass 2's reading.
            before = _metrics()
            reports.append({
                k: v
                for k, v in run_bench(
                    f"{tag}_p{p}", workload, rate, concurrency
                ).items()
                if k in _REPORT_KEYS
            })
            if p + 1 < passes:
                time.sleep(10)  # let the pass's stores land before reusing
        result["reports"] = reports
        result["report"] = reports[-1]
        result["passes"] = passes
        time.sleep(8)  # let async stores land
        after = _metrics()
        result["metrics"] = {k: after[k] - before[k] for k in before}
        result["objects"] = cache_object_count()
        l1 = server_status()["storage_manager"]["l1_manager"]
        result["l1_used_bytes"] = int(l1["memory_used_bytes"])
        result["l1_usage_ratio"] = float(l1["memory_usage_ratio"])
        retrieved = (
            grep_retrieved(tag)[retrieves_before_last:] if config != "off" else []
        )
        result["retrieved_tokens"] = sum(retrieved)
        result["retrieves"] = len(retrieved)
        if config != "off":
            # Drive one more step so held ops and the ledger line come out.
            complete(long_prompt("flush", 2), 4)
            time.sleep(3)
    finally:
        teardown([engine, server])
    result["rc_engine"] = engine.returncode
    result["rc_server"] = server.returncode
    result["mode_lines"] = mode_lines(tag)
    result["ledger"] = grep_final_counters(tag) or {}
    result["tracebacks"] = grep_tracebacks(tag)
    result["warnings"] = [
        w for w in grep_warnings(tag) if driver._WARN_NO_SESSION not in w
    ]
    out = LOGDIR / f"wl_{tag}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[wl] {tag}: wrote {out}")
    return result


def _hit_rate(m: dict[str, float]) -> float:
    """External-cache hit rate over the run, or NaN if nothing was queried."""
    q = m.get("ext_queries", _UNMEASURED)
    if not q or q != q:  # zero or NaN
        return _UNMEASURED
    return m["ext_hits"] / q


def _median(values: list[float]) -> float:
    """Median of a non-empty list."""
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def table() -> int:
    """Print every saved result, grouped into cells and aggregated over reps.

    A cell is (workload, concurrency, pool, config). Its reps are reduced by
    median, and the spread across reps is printed next to it -- a delta
    between two cells means nothing unless it is larger than the spread
    within them.

    Two delta blocks are printed, and they answer different questions:

    - **lazy vs eager** is this PR's question. Both configs run the same
      connector, the same retrieval path and the same corpus; the only
      difference is which chunks get stored and when, so a delta there is
      attributable to the policy. The claims under test are "never worse
      than eager anywhere" and "better than eager where gate 1 has room".
    - **vs off** is the connector's question, not the policy's. Whether
      fetching beats recomputing is gate 3 (`min_prefix_tokens`, 0 in every
      run here), so a loss against `off` is not chargeable to lazy offload.
      It is printed because it calibrates the other block: it is what one
      store plus one lookup costs.

    Returns:
        0 always; this is a report, not a verdict. The oracle (#18) owns the
        pass/fail assertions -- a latency number has no true value to assert
        against, and a threshold invented here would be a number this
        machine happened to produce.
    """
    rows = sorted(LOGDIR.glob("wl_W_*.json"))
    if not rows:
        print("[wl] no results in", LOGDIR)
        return 0
    results = [json.loads(p.read_text()) for p in rows]
    cells: dict[tuple[str, int, int, float, str], list[dict]] = {}
    for r in results:
        cells.setdefault(
            (
                r["workload"], r["max_concurrency"],
                int(r.get("pool_blocks", POOL_BLOCKS)),
                float(r.get("horizon", DEFAULT_HORIZON)), r["config"],
            ),
            [],
        ).append(r)

    hdr = (
        f"{'workload':9s} {'conc':>4s} {'pool':>5s} {'horiz':>5s} {'cfg':6s} "
        f"{'n':>2s} "
        f"{'TTFT p50':>9s} {'spread':>13s} {'TTFT p99':>9s} {'TPOT p50':>9s} "
        f"{'out tok/s':>10s} {'apc hit':>8s} {'ext hit':>8s} "
        f"{'objects':>8s} {'preempt':>8s} {'emitted':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(cells):
        workload, conc, pool, horizon, config = key
        reps = cells[key]
        t50 = [r["report"]["median_ttft_ms"] for r in reps]
        m = reps[0]["metrics"]
        apc = (
            m["apc_hits"] / m["apc_queries"] if m.get("apc_queries") else _UNMEASURED
        )
        print(
            f"{workload:9s} {conc:4d} {pool:5d} "
            f"{(f'{horizon:g}' if config == 'lazy' else '-'):>5s} "
            f"{config:6s} {len(reps):2d} "
            f"{_median(t50):9.1f} "
            f"{f'{min(t50):.1f}-{max(t50):.1f}':>13s} "
            f"{_median([r['report']['p99_ttft_ms'] for r in reps]):9.1f} "
            f"{_median([r['report']['median_tpot_ms'] for r in reps]):9.2f} "
            f"{_median([r['report']['output_throughput'] for r in reps]):10.1f} "
            f"{apc:8.3f} {_hit_rate(m):8.3f} "
            f"{_median([float(r['objects']) for r in reps]):8.0f} "
            f"{_median([r['metrics']['preemptions'] for r in reps]):8.0f} "
            f"{reps[0]['ledger'].get('emitted', '-'):>8}"
        )

    shapes = sorted({(w, c, p) for w, c, p, _h, _cfg in cells})
    # eager has no policy, so it exists at one horizon only and every lazy
    # horizon is compared against that same eager cell.
    horizons = sorted({h for _w, _c, _p, h, cfg in cells if cfg == "lazy"})
    print()
    print("# lazy vs eager -- this PR's claim (same connector, same retrieval)")
    for workload, conc, pool in shapes:
        base = [
            r for h in horizons for r in cells.get((workload, conc, pool, h, "eager"), [])
        ] + cells.get((workload, conc, pool, DEFAULT_HORIZON, "eager"), [])
        base = list({r["tag"]: r for r in base}.values())
        for horizon in horizons:
            reps = cells.get((workload, conc, pool, horizon, "lazy"), [])
            shape = f"[{workload} c{conc} pool{pool} h{horizon:g}]"
            if not base or not reps:
                continue
            print(
                f"{shape} lazy vs eager: "
                f"TTFT p50 {_cell_pct(reps, base, 'median_ttft_ms')}, "
                f"TTFT p99 {_cell_pct(reps, base, 'p99_ttft_ms')}, "
                f"out tok/s {_cell_pct(reps, base, 'output_throughput')}, "
                f"objects {_median([float(r['objects']) for r in reps]):.0f}"
                f" vs {_median([float(r['objects']) for r in base]):.0f}, "
                f"overlap={_overlaps(reps, base, 'median_ttft_ms')}"
            )

    # Against the no-connector baseline: the connector's cost, not the
    # policy's. See the docstring.
    print()
    print("# vs off -- the connector's cost, gate 3's question, not this PR's")
    for workload, conc, pool in shapes:
        base = cells.get((workload, conc, pool, DEFAULT_HORIZON, "off"), [])
        if not base:
            print(f"[{workload} c{conc} pool{pool}] no `off` baseline")
            continue
        for key in sorted(cells):
            if key[:3] != (workload, conc, pool) or key[4] == "off":
                continue
            horizon, config = key[3], key[4]
            reps = cells[key]
            label = f"{config}" + (f" h{horizon:g}" if config == "lazy" else "")
            print(
                f"[{workload} c{conc} pool{pool}] {label:12s} vs off: "
                f"TTFT p50 {_cell_pct(reps, base, 'median_ttft_ms')}, "
                f"TTFT p99 {_cell_pct(reps, base, 'p99_ttft_ms')}, "
                f"TPOT p50 {_cell_pct(reps, base, 'median_tpot_ms')}, "
                f"out tok/s {_cell_pct(reps, base, 'output_throughput')}"
            )
    return 0


def _overlaps(reps: list[dict], base: list[dict], field: str) -> str:
    """Whether two cells' per-rep ranges of `field` intersect.

    A "yes" means the cells are not separated by this measurement, whatever
    their medians say -- the check that retracted this file's first
    lazy-vs-eager conclusion.

    Args:
        reps: One cell's result documents.
        base: The other cell's result documents.
        field: A key of a result's `report` sub-document.

    Returns:
        "yes", "no", or "n=1" when either cell has a single rep and therefore
        no range to compare.
    """
    a = [r["report"][field] for r in reps]
    b = [r["report"][field] for r in base]
    if len(a) < 2 or len(b) < 2:
        return "n=1"
    return "yes" if min(a) <= max(b) and min(b) <= max(a) else "no"


def _cell_pct(reps: list[dict], base: list[dict], field: str) -> str:
    """Median-of-reps change of `field` against the baseline cell."""
    return _pct(
        _median([r["report"][field] for r in reps]),
        _median([r["report"][field] for r in base]),
    )


def _pct(value: float, base: float) -> str:
    """`value` as a signed percentage change from `base`."""
    if not base:
        return "n/a"
    return f"{(value - base) / base * 100:+.1f}%"


def main() -> int:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "table"
    if cmd == "run":
        rate = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_RATE
        conc = int(sys.argv[5]) if len(sys.argv) > 5 else MAX_CONCURRENCY
        rep = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        pool = int(sys.argv[7]) if len(sys.argv) > 7 else POOL_BLOCKS
        horizon = float(sys.argv[8]) if len(sys.argv) > 8 else DEFAULT_HORIZON
        run(sys.argv[2], sys.argv[3], rate, conc, rep, pool, horizon)
        return 0
    if cmd == "matrix":
        rate = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_RATE
        for workload in WORKLOADS:
            for config in CONFIGS:
                run(config, workload, rate)
        return table()
    if cmd == "table":
        return table()
    raise SystemExit(
        f"usage: {sys.argv[0]} "
        "run <config> <workload> [rate] [conc] [rep] [pool] | matrix | table"
    )


if __name__ == "__main__":
    sys.exit(main())
