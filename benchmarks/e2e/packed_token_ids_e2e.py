#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end A/B for the connector's ``token_ids`` transmission cost.

The microbenchmark in ``benchmarks/microbenchmark`` measures the CPU each
hop burns on token ids in isolation. This script asks the question that
actually matters: does removing that CPU move TTFT on a real vLLM server?

It runs one vLLM + LMCache-server pair per configuration, against a fixed
set of long prompts, and reports TTFT for a **second** pass over the same
prompts -- the pass that hits the cache, so its TTFT is gated by the
retrieve path rather than by prefill compute.

A configuration is just a checkout, which is how the two points are reached:

``baseline``
    a checkout from before this change: token ids as a list of ints.
``packed``
    this branch: token ids as a big-endian ``uint32`` buffer.

Prompts must be long for this to show anything. The per-token decode the
change removes is proportional to context length, and below roughly 16k
tokens packing is a small pessimization, so a short-prompt run measures
noise. The cost also multiplies by TP rank, since every worker decodes the
op and every rank sends its own key to the server.

Prompts are sent as explicit token id lists through the OpenAI completions
API, so the two passes see byte-identical token sequences and the cache hit
is exact -- no tokenizer round trip in between to perturb it.

Besides TTFT the script samples CPU time of the LMCache server process tree
and of the vLLM process tree, which is where the saving should show up even
when it is too small to move latency.

Usage::

    python benchmarks/e2e/packed_token_ids_e2e.py run \\
        --label packed --checkout /raid/bo/delta_token_ids_pr
    python benchmarks/e2e/packed_token_ids_e2e.py report results/*.json
"""

# Standard
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import argparse
import asyncio
import json
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import time

# Third Party
import aiohttp
import psutil

DEFAULT_MODEL = (
    "/raid/jiayi-data/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
VENV_BIN = Path("/home/bo/LMCache-worktrees/mtp/.venv/bin")

STARTUP_TIMEOUT_S = 1800.0
"""vLLM with TP=8 loads weights and captures graphs before it serves."""


@dataclass
class RunSpec:
    """One configuration to measure."""

    label: str
    checkout: Path
    model: str
    tensor_parallel_size: int
    input_len: int
    output_len: int
    num_prompts: int
    concurrency: int
    lmcache_port: int
    vllm_port: int
    l1_size_gb: int
    max_model_len: int
    gpu_memory_utilization: float
    kv_cache_dtype: str
    vllm_extra_args: list[str]
    warm_passes: int


@dataclass
class PassStats:
    """Latency of one pass over the prompt set."""

    ttft_ms: dict[str, float] = field(default_factory=dict)
    total_s: float = 0.0
    completed: int = 0
    failed: int = 0
    samples: list[float] = field(default_factory=list)
    """Raw per-request TTFTs, kept so repeated passes can be pooled."""


@dataclass
class RunResult:
    """Everything one configuration produced."""

    label: str
    spec: dict[str, Any]
    cold: PassStats
    warm: PassStats
    lmcache_server_cpu_s: float
    vllm_cpu_s: float
    main_loop_cpu_s: float
    """CPU seconds burned by the server's busiest single thread (see
    :func:`busiest_thread_delta`); over the measured wall time it gives that
    thread's utilization, i.e. how close the run came to the serialization
    limit this change relieves."""
    measured_wall_s: float
    """Wall time the CPU counters span, for turning them into utilizations."""


####
# Prompt generation
####


def model_vocab_size(model: str) -> int:
    """Read a model's vocabulary size from its ``config.json``.

    Prompts are sent as raw token ids, so an id past the embedding table is
    not a bad prompt but an out-of-bounds gather: vLLM does not validate it
    and the failure surfaces as a device-side assert that takes the engine
    core down. The bound therefore has to come from the model being served,
    not from a constant.

    Args:
        model: Path to a local HuggingFace checkout.

    Returns:
        The exclusive upper bound on token ids.

    Raises:
        ValueError: If the checkout has no readable ``vocab_size``.
    """
    config_path = Path(model) / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except OSError as exc:
        raise ValueError(f"cannot read {config_path}: {exc}") from exc
    vocab_size = config.get("vocab_size", 0)
    if not isinstance(vocab_size, int) or vocab_size <= 1:
        raise ValueError(f"{config_path} has no usable vocab_size: {vocab_size!r}")
    return vocab_size


def make_prompts(
    num_prompts: int, input_len: int, vocab_size: int, seed: int
) -> list[list[int]]:
    """Build deterministic, mutually distinct token id prompts.

    Each prompt gets its own leading marker so no two share a prefix, which
    keeps vLLM's own prefix cache from serving one prompt out of another and
    masking what LMCache is doing.

    Args:
        num_prompts: How many prompts to build.
        input_len: Tokens per prompt.
        vocab_size: Exclusive upper bound on token ids.
        seed: Seed for the generator.

    Returns:
        One token id list per prompt.
    """
    rng = random.Random(seed)
    prompts: list[list[int]] = []
    for index in range(num_prompts):
        body = [rng.randrange(1, vocab_size) for _ in range(input_len - 1)]
        prompts.append([1 + index % (vocab_size - 1), *body])
    return prompts


####
# Serving
####


def died(process: subprocess.Popen[bytes], name: str, log: Path) -> RuntimeError:
    """Build the error for a server that exited during startup.

    Args:
        process: The exited process.
        name: Human-readable name for the message.
        log: Its log file, whose tail is quoted.

    Returns:
        The error to raise, carrying the last lines of the log -- a startup
        failure is almost always a bad argument or a bad config, and the
        reason is in there.
    """
    try:
        tail = "".join(log.read_text(errors="replace").splitlines(keepends=True)[-20:])
    except OSError:
        tail = "(log unreadable)"
    return RuntimeError(
        f"{name} exited with code {process.returncode} during startup\n{tail}"
    )


FATAL_STARTUP_MARKERS = (
    "Engine core initialization failed",
    "EngineCore failed to start",
    "WorkerProc failed to start",
)
"""Log lines that mean a server will never become healthy.

``died()`` only catches a server whose *own* process exits. vLLM's API server
outlives its engine core: when the core dies during startup the parent keeps
running and keeps answering nothing, so a liveness check alone waits out the
full startup timeout. These markers are the engine's own report that it gave
up.
"""


def startup_failed(log: Path) -> str:
    """Return the fatal line from ``log``, or an empty string if there is none.

    Args:
        log: The server's log file.

    Returns:
        The first matching fatal line, stripped; empty when the log shows no
        fatal marker or cannot be read.
    """
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        if any(marker in line for marker in FATAL_STARTUP_MARKERS):
            return line.strip()
    return ""


def wait_for_port(
    port: int, deadline: float, process: subprocess.Popen[bytes], name: str, log: Path
) -> None:
    """Block until ``port`` accepts connections or ``process`` dies.

    Args:
        port: TCP port on localhost.
        deadline: ``time.monotonic()`` value to give up at.
        process: The server being waited on.
        name: Human-readable name for error messages.
        log: The server's log file.

    Raises:
        RuntimeError: If the process exited before the port opened.
        TimeoutError: If nothing is listening by the deadline.
    """
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            probe.settimeout(1.0)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        if process.poll() is not None:
            raise died(process, name, log)
        time.sleep(1.0)
    raise TimeoutError(f"nothing listening on port {port}")


async def wait_for_health(
    url: str,
    deadline: float,
    process: subprocess.Popen[bytes],
    name: str,
    log: Path,
) -> None:
    """Poll ``url`` until it answers 200 or ``process`` dies.

    Args:
        url: Health endpoint to poll.
        deadline: ``time.monotonic()`` value to give up at.
        process: The server being waited on.
        name: Human-readable name for error messages.
        log: The server's log file.

    Raises:
        RuntimeError: If the process exited before answering.
        TimeoutError: If the endpoint never became healthy.
    """
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status == 200:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            if process.poll() is not None:
                raise died(process, name, log)
            fatal = startup_failed(log)
            if fatal:
                raise RuntimeError(
                    f"{name} failed to start (process still alive): {fatal}\n"
                    + "".join(
                        log.read_text(errors="replace").splitlines(keepends=True)[-20:]
                    )
                )
            await asyncio.sleep(2.0)
    raise TimeoutError(f"{url} never became healthy")


def launch_lmcache_server(spec: RunSpec, log_dir: Path) -> subprocess.Popen[bytes]:
    """Start the LMCache MP server out of ``spec.checkout``.

    Args:
        spec: The configuration being measured.
        log_dir: Directory to write the server log into.

    Returns:
        The running process.
    """
    env = dict(os.environ, PYTHONPATH=str(spec.checkout))
    log = (log_dir / f"{spec.label}-lmcache-server.log").open("wb")
    return subprocess.Popen(
        [
            str(VENV_BIN / "lmcache"),
            "server",
            "--port",
            str(spec.lmcache_port),
            "--l1-size-gb",
            str(spec.l1_size_gb),
            "--eviction-policy",
            "LRU",
        ],
        cwd=str(spec.checkout),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def launch_vllm(spec: RunSpec, log_dir: Path) -> subprocess.Popen[bytes]:
    """Start vLLM with the LMCache MP connector out of ``spec.checkout``.

    Args:
        spec: The configuration being measured.
        log_dir: Directory to write the server log into.

    Returns:
        The running process.
    """
    extra: dict[str, Any] = {"lmcache.mp.port": spec.lmcache_port}
    kv_config = json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra,
        }
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(spec.checkout),
        VLLM_ALLOW_LONG_MAX_MODEL_LEN="1",
        VLLM_SERVER_DEV_MODE="1",
    )
    log = (log_dir / f"{spec.label}-vllm.log").open("wb")
    return subprocess.Popen(
        [
            str(VENV_BIN / "vllm"),
            "serve",
            spec.model,
            "--served-model-name",
            "bench",
            "--port",
            str(spec.vllm_port),
            "--tensor-parallel-size",
            str(spec.tensor_parallel_size),
            "--max-model-len",
            str(spec.max_model_len),
            "--gpu-memory-utilization",
            str(spec.gpu_memory_utilization),
            "--kv-cache-dtype",
            spec.kv_cache_dtype,
            "--disable-uvicorn-access-log",
            *spec.vllm_extra_args,
            "--kv-transfer-config",
            kv_config,
        ],
        cwd=str(spec.checkout),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def thread_cpu_seconds(pid: int) -> dict[str, float]:
    """Return per-thread CPU seconds for ``pid``, keyed by thread id.

    Threads are identified by id, not by name: CPython 3.10 does not push
    ``threading.Thread(name=...)`` down to the OS, so every entry under
    ``/proc/<pid>/task/*/comm`` reads ``python`` and a name-based lookup
    silently matches nothing.

    Args:
        pid: The process to sample.

    Returns:
        ``{tid: user + system CPU seconds}``; empty if the process is gone.
    """
    try:
        task_dir = Path(f"/proc/{pid}/task")
        tids = [entry.name for entry in task_dir.iterdir()]
    except OSError:
        return {}
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    out: dict[str, float] = {}
    for tid in tids:
        try:
            # utime and stime are fields 14 and 15 of /proc/.../stat, counted
            # from after comm -- which can itself contain spaces, so the
            # split starts past its closing parenthesis.
            stat = (task_dir / tid / "stat").read_text()
            fields = stat[stat.rindex(") ") + 2 :].split()
            out[tid] = (int(fields[11]) + int(fields[12])) / ticks_per_second
        except (OSError, ValueError, IndexError):
            continue
    return out


def busiest_thread_delta(before: dict[str, float], after: dict[str, float]) -> float:
    """Return the largest CPU-second increase of any single thread.

    The LMCache server decodes every request's msgpack payload on one thread:
    ``BlockingRequestHandler.__call__`` calls ``unwrap_request_payloads``
    before handing work to its executor. That thread is therefore a
    serialization point, and whether a run can show this change in latency
    rather than only in total CPU depends on how close it came to using a
    full core. Taking the maximum over threads finds it without needing to
    know which one it is.

    Args:
        before: Sample taken before the measured window.
        after: Sample taken after it.

    Returns:
        The busiest thread's CPU seconds over the window, 0.0 if unknown.
    """
    if not after:
        return 0.0
    deltas = (used - before.get(tid, 0.0) for tid, used in after.items())
    return max(deltas, default=0.0)


def tree_cpu_seconds(pid: int) -> float:
    """Return total CPU seconds burned by ``pid`` and its children.

    Args:
        pid: Root process id.

    Returns:
        User + system CPU seconds, or 0.0 if the process is already gone.
    """
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0.0
    total = 0.0
    for proc in [root, *root.children(recursive=True)]:
        try:
            times = proc.cpu_times()
            total += times.user + times.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def terminate(process: subprocess.Popen[bytes], name: str) -> None:
    """Stop a process group started with ``start_new_session``.

    The group is signalled even when the direct child has already exited.
    vLLM's TP workers are in the group but are not children of this script,
    and when the engine core dies the parent goes while they stay -- each
    still holding its whole GPU memory reservation, which makes the next
    configuration fail to start. Returning early on a dead child is how
    that leak happens.

    Args:
        process: The process whose group should be stopped.
        name: Label used in the message printed on a forced kill.
    """
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        # The child was reaped and its pid is gone; nothing addressable.
        return
    for sig, timeout in ((signal.SIGINT, 60), (signal.SIGKILL, 30)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return  # group is empty -- nothing left to stop
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        # The child is gone; make sure the rest of the group went with it.
        # They are reaped by init, not by us, so give them a moment rather
        # than reporting a leak that is really just a race.
        for _ in range(20):
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.5)
    print(f"  warning: {name} process group did not exit", file=sys.stderr)


####
# Load generation
####


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: list[int],
    output_len: int,
) -> float:
    """Stream one completion and return its time to first token.

    Args:
        session: The shared HTTP session.
        url: Completions endpoint.
        prompt: Token ids to send verbatim.
        output_len: Tokens to generate.

    Returns:
        Time to first token, in milliseconds.

    Raises:
        RuntimeError: If the server answered with an error status.
    """
    payload = {
        "model": "bench",
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    started = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        async for raw in resp.content:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue
            if line == b"data: [DONE]":
                break
            ttft_ms = (time.perf_counter() - started) * 1_000.0
            # Drain the rest so the connection is reusable.
            async for _ in resp.content:
                pass
            return ttft_ms
    raise RuntimeError("stream ended without producing a token")


async def run_pass(
    url: str, prompts: list[list[int]], output_len: int, concurrency: int
) -> PassStats:
    """Send every prompt once, ``concurrency`` in flight at a time.

    Args:
        url: Completions endpoint.
        prompts: The prompt set.
        output_len: Tokens to generate per request.
        concurrency: Maximum requests in flight.

    Returns:
        The pass's latency statistics.
    """
    gate = asyncio.Semaphore(concurrency)
    ttfts: list[float] = []
    failures = 0

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=3600),
        connector=aiohttp.TCPConnector(limit=concurrency * 2),
        # An SSE chunk is read as one line, and aiohttp caps a line at 512 KB
        # by default. Over that it raises LineTooLong, whose ``code`` is 400 --
        # so the failure reads like an HTTP 400 from the server even though
        # the server logged nothing and sent a perfectly good response. Raise
        # the ceiling rather than lose requests, which would silently reduce
        # the load this benchmark is trying to apply.
        read_bufsize=2**22,
        max_line_size=2**24,
        max_field_size=2**24,
    ) as session:

        async def guarded(prompt: list[int]) -> None:
            nonlocal failures
            async with gate:
                try:
                    ttfts.append(await one_request(session, url, prompt, output_len))
                except Exception as exc:  # noqa: BLE001 - report, do not abort the run
                    failures += 1
                    print(f"  request failed: {exc}", file=sys.stderr)

        started = time.perf_counter()
        await asyncio.gather(*(guarded(p) for p in prompts))
        elapsed = time.perf_counter() - started

    ttfts.sort()
    return PassStats(
        ttft_ms=summarize(ttfts),
        total_s=elapsed,
        completed=len(ttfts),
        failed=failures,
        samples=ttfts,
    )


def summarize(samples: list[float]) -> dict[str, float]:
    """Return mean/p50/p90/p99 of ``samples`` (empty dict when there are none)."""
    if not samples:
        return {}
    ordered = sorted(samples)

    def pct(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": ordered[-1],
    }


####
# Driver
####


async def measure(spec: RunSpec, log_dir: Path) -> RunResult:
    """Bring up one configuration, measure it, and tear it down.

    Args:
        spec: The configuration to measure.
        log_dir: Where server logs are written.

    Returns:
        The configuration's results.
    """
    print(f"[{spec.label}] starting LMCache server on :{spec.lmcache_port}")
    lmcache = launch_lmcache_server(spec, log_dir)
    vllm = None
    try:
        wait_for_port(
            spec.lmcache_port,
            time.monotonic() + 120.0,
            lmcache,
            "LMCache server",
            log_dir / f"{spec.label}-lmcache-server.log",
        )
        print(f"[{spec.label}] starting vLLM on :{spec.vllm_port}")
        vllm = launch_vllm(spec, log_dir)
        await wait_for_health(
            f"http://127.0.0.1:{spec.vllm_port}/health",
            time.monotonic() + STARTUP_TIMEOUT_S,
            vllm,
            "vLLM",
            log_dir / f"{spec.label}-vllm.log",
        )
        print(f"[{spec.label}] serving; generating prompts")

        prompts = make_prompts(
            spec.num_prompts,
            spec.input_len,
            vocab_size=model_vocab_size(spec.model),
            seed=20260915,
        )
        url = f"http://127.0.0.1:{spec.vllm_port}/v1/completions"

        lmcache_cpu_before = tree_cpu_seconds(lmcache.pid)
        vllm_cpu_before = tree_cpu_seconds(vllm.pid)
        threads_before = thread_cpu_seconds(lmcache.pid)
        measured_start = time.perf_counter()

        print(f"[{spec.label}] cold pass ({spec.num_prompts} prompts)")
        cold = await run_pass(url, prompts, spec.output_len, spec.concurrency)
        # Give asynchronous stores time to land before asking for them back.
        await asyncio.sleep(30.0)
        # Repeat the warm pass to raise n without enlarging the prompt set.
        # The effect being looked for here is small -- one LOOKUP and one
        # RETRIEVE per request, tens of milliseconds against a warm TTFT of
        # a few hundred -- so a handful of samples cannot resolve it, while
        # more prompts would need proportionally more L1 and risk evicting
        # the very entries the warm pass is meant to hit. The cache contents
        # are identical on every repeat, so the passes are samples of one
        # distribution and are pooled.
        warm_ttfts: list[float] = []
        warm = PassStats()
        for pass_index in range(spec.warm_passes):
            print(
                f"[{spec.label}] warm pass {pass_index + 1}/{spec.warm_passes} "
                "(same prompts, expect LMCache hits)"
            )
            one = await run_pass(url, prompts, spec.output_len, spec.concurrency)
            warm_ttfts.extend(one.samples)
            warm.total_s += one.total_s
            warm.completed += one.completed
            warm.failed += one.failed
        warm.ttft_ms = summarize(warm_ttfts)
        warm.samples = warm_ttfts

        return RunResult(
            label=spec.label,
            spec={**asdict(spec), "checkout": str(spec.checkout)},
            cold=cold,
            warm=warm,
            lmcache_server_cpu_s=tree_cpu_seconds(lmcache.pid) - lmcache_cpu_before,
            vllm_cpu_s=tree_cpu_seconds(vllm.pid) - vllm_cpu_before,
            main_loop_cpu_s=busiest_thread_delta(
                threads_before, thread_cpu_seconds(lmcache.pid)
            ),
            measured_wall_s=time.perf_counter() - measured_start,
        )
    finally:
        if vllm is not None:
            terminate(vllm, "vllm")
        terminate(lmcache, "lmcache server")


def cmd_run(args: argparse.Namespace) -> None:
    """Measure one configuration and write its JSON result."""
    spec = RunSpec(
        label=args.label,
        checkout=Path(args.checkout).resolve(),
        model=args.model,
        tensor_parallel_size=args.tp,
        input_len=args.input_len,
        output_len=args.output_len,
        num_prompts=args.num_prompts,
        concurrency=args.concurrency,
        lmcache_port=args.lmcache_port,
        vllm_port=args.vllm_port,
        l1_size_gb=args.l1_size_gb,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        vllm_extra_args=list(args.vllm_extra),
        warm_passes=args.warm_passes,
    )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(measure(spec, out_dir))
    destination = out_dir / f"{spec.label}.json"
    destination.write_text(json.dumps(asdict(result), indent=2))
    print(f"[{spec.label}] wrote {destination}")


def cmd_report(args: argparse.Namespace) -> None:
    """Print a comparison table over previously written JSON results."""
    results = [json.loads(Path(p).read_text()) for p in args.results]
    if not results:
        print("no results")
        return

    spec = results[0]["spec"]
    print(
        f"\n{spec['input_len']:,}-token prompts x {spec['num_prompts']}, "
        f"concurrency {spec['concurrency']}, TP={spec['tensor_parallel_size']}"
    )
    header = (
        f"  {'config':<10} {'cold p50':>10} {'warm p50':>10} {'warm p99':>10} "
        f"{'warm req/s':>11} {'lmcache CPU':>12} {'dispatch':>9} {'vLLM CPU':>10}"
    )
    print(header)
    for result in results:
        cold = result["cold"]["ttft_ms"]
        warm = result["warm"]["ttft_ms"]
        warm_s = result["warm"]["total_s"]
        rate = result["warm"]["completed"] / warm_s if warm_s else 0.0
        print(
            f"  {result['label']:<10} {cold.get('p50', 0):>9.0f}ms "
            f"{warm.get('p50', 0):>9.0f}ms {warm.get('p99', 0):>9.0f}ms "
            f"{rate:>11.2f} "
            f"{result['lmcache_server_cpu_s']:>11.1f}s "
            f"{result.get('main_loop_cpu_s', 0.0):>8.1f}s "
            f"{result['vllm_cpu_s']:>9.1f}s"
        )
    failures = sum(r["cold"]["failed"] + r["warm"]["failed"] for r in results)
    if failures:
        print(f"  note: {failures} request(s) failed; see the server logs")

    # Without this the table cannot be interpreted: the same CPU saving is
    # invisible below saturation and becomes throughput above it.
    print("\n  busiest LMCache server thread (the payload-decode loop):")
    for result in results:
        wall = result.get("measured_wall_s", 0.0)
        used = result.get("main_loop_cpu_s", 0.0)
        if wall <= 0:
            print(f"    {result['label']:<10} unmeasured")
            continue
        share = used / wall * 100
        verdict = (
            "SATURATED -- this run can show the change in latency/throughput"
            if share >= 80
            else "near saturation"
            if share >= 50
            else "far below saturation -- expect CPU savings only, not latency"
        )
        print(f"    {result['label']:<10} {share:5.1f}% of one core   {verdict}")


def main() -> None:
    """Parse arguments and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Measure one configuration.")
    run.add_argument("--label", required=True)
    run.add_argument("--checkout", required=True)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--tp", type=int, default=4)
    run.add_argument("--input-len", type=int, default=120_000)
    # One token is all TTFT needs, and generating more measures nothing extra:
    # a store op is emitted only once a whole LMCache chunk of new tokens has
    # accumulated (``GetStoreMetadata`` returns None below one chunk), and
    # decode adds one token per step, so a short generation emits no store ops
    # at all. Decode length is therefore pure wall-clock -- expensive here,
    # since --enforce-eager is what a DeepSeek-V4 run needs to start at all.
    run.add_argument("--output-len", type=int, default=1)
    run.add_argument("--num-prompts", type=int, default=8)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--lmcache-port", type=int, default=5599)
    run.add_argument("--vllm-port", type=int, default=8199)
    run.add_argument("--l1-size-gb", type=int, default=400)
    run.add_argument("--max-model-len", type=int, default=131_072)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    # Some models refuse "auto": DeepSeek-V4's fp8_ds_mla paged layout asserts
    # unless the KV cache dtype is fp8, and the assert fires in each TP worker
    # during load, so the failure looks like a worker crash rather than a
    # configuration error.
    run.add_argument("--kv-cache-dtype", default="auto")
    # Model-specific vLLM flags without growing a knob per model. Both arms
    # get the same list, so it cannot bias the A/B. Repeat the option once per
    # flag and attach the value with "=", since a bare
    # "--vllm-extra --enforce-eager" reads as a missing value to argparse:
    #   --vllm-extra=--enforce-eager --vllm-extra=--block-size=64
    run.add_argument("--vllm-extra", action="append", default=[])
    # Pooled repeats of the warm pass. The cache is already populated, so a
    # repeat costs one warm pass of wall-clock and nothing else.
    run.add_argument("--warm-passes", type=int, default=1)
    run.add_argument("--out-dir", default="benchmarks/e2e/results")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="Print a table over JSON results.")
    report.add_argument("results", nargs="+")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
