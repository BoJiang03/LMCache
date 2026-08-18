"""Agentic (SWE-agent) session replay against the MP connector.

Why this workload exists. The reported hot/cold and QASPER results cover
two shapes: a synthetic document set sized to isolate capacity pressure, and
real multi-round paper QA where a returning user re-sends one long, *fixed*
prefix. Agent serving is a third shape and the one production deployments
now generate most: a session's prompt **grows monotonically**, every step
extends the previous step's prompt with an action and a tool observation,
and the session idles between steps while the tool runs. So each step's KV
is written once and read once, a few seconds later -- exactly the window in
which a policy has to decide whether a lower-tier copy is worth making.

The regime this creates:

- reuse distance is short in wall-clock but the GPU pool holds only a few
  sessions' contexts, so a step's prefix survives to the next step only when
  no other session displaced it;
- the sum of all sessions' final contexts is the distinct working set, which
  the sweep moves across the L1 budget;
- eager offload stores every step's new chunks the moment they exist, so it
  writes the KV of sessions that are still GPU-resident; eviction-aware
  offload writes a session's KV when the GPU is about to drop it.

Load is held fixed while the working set is swept: session `s` releases its
step `k` at `t0 + (s + k * sessions) / RATE`, so the aggregate step rate is
`RATE` for every cohort size and every policy, and the per-session gap
(`sessions / RATE`) is the agent's tool-execution time. A step that cannot
be released on schedule -- because its own session's previous step is still
running -- is recorded as schedule lag rather than silently reshaping the
load.

The engine's answer is discarded and the *recorded* action is appended, so
both policies replay byte-identical request streams.
"""

import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import driver  # noqa: E402
from accuracy import L1Sampler, eviction_cycles  # noqa: E402
from driver import (  # noqa: E402
    LOGDIR,
    grep_final_counters,
    grep_ledgers,
    grep_tracebacks,
    grep_warnings,
    mode_lines,
    teardown,
)
from workload import (  # noqa: E402
    MODEL,
    TP_SIZE,
    _CONNECTOR_CONFIGS,
    _metrics,
    start_server_sized,
)

#: KV bytes per token for Qwen3-8B: 36 layers * 8 KV heads * 128 dims * 2
#: tensors * 2 bytes. Every GiB figure below is derived from it.
KV_BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2

#: The prepared cohort: real SWE-agent trajectories, see prepare_cohort.py.
COHORT = os.environ.get("AGENTIC_COHORT", "/raid/data/hub/pr4499_agentic/cohort_s12_48.json")

#: Sessions in the cohort under test. The sweep's independent variable: it
#: multiplies the distinct KV working set without changing the offered load.
SESSIONS = int(os.environ.get("AGENTIC_SESSIONS", "16"))

#: Aggregate step rate, steps per second, held fixed across cohort sizes.
RATE = float(os.environ.get("AGENTIC_RATE", "2.0"))

#: Generated tokens per step. Short on purpose: an agent step's cost is its
#: prompt, and the recorded action replaces the generated text anyway.
OUTPUT_LEN = int(os.environ.get("AGENTIC_OUTPUT_LEN", "32"))

#: GPU KV pool in 16-token blocks. 20 GiB, matching the reported hot/cold and
#: QASPER runs, so the three workloads sit on the same pool.
POOL_GIB = float(os.environ.get("AGENTIC_POOL_GIB", "20"))
POOL_BLOCKS = int(POOL_GIB * (1 << 30)) // KV_BYTES_PER_TOKEN // 16

#: L1 (host) budget in GiB.
L1_GB = int(os.environ.get("AGENTIC_L1_GB", "40"))

#: Context ceiling: the longest selected step prompt plus generation.
MAX_MODEL_LEN = int(os.environ.get("AGENTIC_MAX_MODEL_LEN", "24576"))

#: Per-request wall-clock ceiling. A step that exceeds it is recorded as a
#: failure rather than left to stall the session's schedule forever.
REQUEST_TIMEOUT = float(os.environ.get("AGENTIC_REQUEST_TIMEOUT", "600"))

#: Connector settings merged over the policy's own, as a JSON object. The
#: attribution runs use it to move one policy knob at a time without editing
#: the shared config table.
EXTRA_CONFIG: dict = json.loads(os.environ.get("AGENTIC_EXTRA_CONFIG", "{}"))

#: Appended to the run tag, so a variant does not overwrite the baseline.
TAG_SUFFIX = os.environ.get("AGENTIC_TAG_SUFFIX", "")

CONFIGS = ("off", "eager", "lazy")

_LEDGER_GAUGES = frozenset({"pending"})


def load_cohort(sessions: int) -> dict:
    """Read the prepared cohort and take its first `sessions` entries.

    Args:
        sessions: Number of sessions to replay.

    Returns:
        The cohort document with `cohort` truncated to the requested size.

    Raises:
        ValueError: if the file holds fewer sessions than requested.
    """
    with open(COHORT) as fh:
        document = json.load(fh)
    if len(document["cohort"]) < sessions:
        raise ValueError(
            f"cohort {COHORT} has {len(document['cohort'])} sessions, need {sessions}"
        )
    document["cohort"] = document["cohort"][:sessions]
    return document


def working_set_gib(cohort: list[dict]) -> float:
    """Distinct KV of a cohort in GiB: every session's final step prompt."""
    tokens = sum(session["step_prompt_tokens"][-1] for session in cohort)
    return tokens * KV_BYTES_PER_TOKEN / (1 << 30)


def start_engine(scenario: str, config: str) -> subprocess.Popen:
    """Start `vllm serve` sized for the agentic regime.

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
                **EXTRA_CONFIG,
            },
        }
        cmd += ["--kv-transfer-config", json.dumps(kv_cfg)]
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=driver.REPO
    )
    try:
        _wait_healthy(proc, f"http://127.0.0.1:{driver.VLLM_PORT}/health", 900.0)
    except Exception:
        teardown([proc])
        raise
    return proc


def gpu_processes() -> list[dict]:
    """Every compute process on the GPUs this run owns.

    The node is shared. A neighbour that grabs one of these GPUs part way
    through a run makes its latencies a story about contention, and the run
    has to be discarded rather than averaged. Recording the process list on
    both sides of the measurement is what makes that detectable afterwards.

    Returns:
        One entry per process, with its GPU index, pid and used MiB.
    """
    owned = {int(value) for value in driver.GPU.split(",")}
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_bus_id,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    index_of = {}
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        text=True,
    ).splitlines():
        index, bus = (part.strip() for part in line.split(","))
        index_of[bus] = int(index)
    processes = []
    for line in raw.splitlines():
        bus, pid, used = (part.strip() for part in line.split(","))
        index = index_of.get(bus)
        if index in owned:
            processes.append({"gpu": index, "pid": int(pid), "used_mib": int(used)})
    return processes


def _wait_healthy(proc: subprocess.Popen, url: str, deadline: float) -> None:
    """Wait for an engine to serve `url`, failing as soon as it dies.

    `driver.wait_for` polls for the whole deadline whether or not the process
    is alive, which turns a startup crash into a fifteen-minute stall in the
    middle of a sweep.

    Args:
        proc: The engine process.
        url: Its health endpoint.
        deadline: Seconds to wait before giving up.

    Raises:
        RuntimeError: if the engine exits or the deadline expires.
    """
    started = time.time()
    while time.time() - started < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited during startup with code {proc.returncode}")
        try:
            driver.http_get(url)
            print(f"[ag] vllm ready after {time.time() - started:.0f}s", flush=True)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"vllm not ready within {deadline:.0f}s")


def step(messages: list[dict[str, str]]) -> dict:
    """Send one agent step and time its first token.

    Args:
        messages: The conversation so far, ending with a tool observation.

    Returns:
        A record with `ttft_ms`, `e2e_ms`, token counts, and `ok`.
    """
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": OUTPUT_LEN,
            "temperature": 0,
            "seed": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{driver.VLLM_PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    first = 0.0
    usage: dict = {}
    generated = 0
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", ()):
                text = choice.get("delta", {}).get("content") or ""
                if text and not first:
                    first = time.time()
                generated += len(text)
    done = time.time()
    details = usage.get("prompt_tokens_details") or {}
    return {
        "ttft_ms": (first - started) * 1000.0 if first else math.nan,
        "e2e_ms": (done - started) * 1000.0,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": details.get("cached_tokens", -1),
        "generated_chars": generated,
        "ok": bool(first),
    }


def run_session(index: int, session: dict, sessions: int, t0: float) -> list[dict]:
    """Replay one trajectory on its own schedule.

    Args:
        index: Session index in the cohort; sets its offset in the schedule.
        session: One cohort entry.
        sessions: Cohort size, which sets the per-session step interval.
        t0: Run start time, shared by every session.

    Returns:
        One record per step, in step order.
    """
    messages = session["messages"]
    steps = len(messages) // 2
    records = []
    for k in range(steps):
        release = t0 + (index + k * sessions) / RATE
        lag = time.time() - release
        if lag < 0:
            time.sleep(-lag)
            lag = 0.0
        record = {
            "session": index,
            "instance_id": session["instance_id"],
            "step": k,
            "released_s": release - t0,
            "lag_ms": lag * 1000.0,
        }
        try:
            record.update(step(messages[: 2 + 2 * k]))
        except Exception as error:  # a failed step must not lose the run
            record.update({"ok": False, "error": repr(error)[:200]})
        record["finished_s"] = time.time() - t0
        records.append(record)
    return records


def _stats(values: list[float]) -> dict[str, float]:
    """Count, mean and percentiles of a latency list in ms."""
    ordered = sorted(value for value in values if not math.isnan(value))
    if not ordered:
        return {}
    return {
        "n": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": statistics.median(ordered),
        "p90_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max_ms": ordered[-1],
    }


def _phase_stats(records: list[dict], keep) -> dict:
    """TTFT and E2E statistics over the records `keep` selects."""
    chosen = [record for record in records if keep(record) and record.get("ok")]
    return {
        "ttft": _stats([record["ttft_ms"] for record in chosen]),
        "e2e": _stats([record["e2e_ms"] for record in chosen]),
        "prompt_tokens_sum": sum(record.get("prompt_tokens", 0) for record in chosen),
        "cached_tokens_sum": sum(
            max(0, record.get("cached_tokens", -1)) for record in chosen
        ),
    }


def run(config: str, rep: str = "0", sessions: int = 0, l1_gb: int = 0) -> dict:
    """One measurement point: boot, replay the cohort, collect, tear down.

    Args:
        config: One of CONFIGS.
        rep: Repetition label, part of the tag.
        sessions: Cohort size; 0 keeps SESSIONS.
        l1_gb: L1 budget in GiB; 0 keeps L1_GB.

    Returns:
        The result document, also written to logs/ag_<tag>.json.

    Raises:
        KeyError: if config is unknown.
    """
    if config not in CONFIGS:
        raise KeyError(f"unknown config {config!r}; known: {CONFIGS}")
    count = sessions or SESSIONS
    budget = l1_gb or L1_GB
    document = load_cohort(count)
    cohort = document["cohort"]
    tag = f"AG_{config}_s{count}r{RATE:g}_l{budget}_{rep}{TAG_SUFFIX}"
    result: dict = {
        "config": config,
        "tag": tag,
        "rep": rep,
        "model": MODEL,
        "tensor_parallel_size": TP_SIZE,
        "sessions": count,
        "steps": document["steps"],
        "rate": RATE,
        "session_gap_s": count / RATE,
        "output_len": OUTPUT_LEN,
        "l1_gb": budget,
        "pool_blocks": POOL_BLOCKS,
        "pool_gib": POOL_BLOCKS * 16 * KV_BYTES_PER_TOKEN / (1 << 30),
        "working_set_gib": working_set_gib(cohort),
        "cohort_sha256": document["cohort_sha256"],
        "extra_config": EXTRA_CONFIG,
        "expected_requests": count * document["steps"],
    }
    print(
        f"[ag] === {tag}: {count} sessions, {result['working_set_gib']:.1f} GiB "
        f"distinct KV through a {result['pool_gib']:.1f} GiB pool and "
        f"{budget} GiB of L1, {RATE:g} steps/s"
    )
    server = start_server_sized(tag, budget)
    try:
        engine = start_engine(tag, config)
    except Exception:
        teardown([server])
        raise
    records: list[dict] = []
    try:
        before = _metrics()
        result["gpus"] = driver.GPU
        result["gpu_processes_before"] = gpu_processes()
        sampler = L1Sampler()
        sampler.start()
        started = time.time()
        t0 = started + 1.0
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [
                pool.submit(run_session, index, session, count, t0)
                for index, session in enumerate(cohort)
            ]
            for future in futures:
                records.extend(future.result())
        elapsed = time.time() - started
        after = _metrics()
        result["l1"] = sampler.stop()
        result["cycles"] = eviction_cycles(tag)
        result["gpu_processes_after"] = gpu_processes()
    finally:
        teardown([engine, server])
    result["elapsed_s"] = elapsed
    result["requests"] = len(records)
    result["failed"] = [record for record in records if not record.get("ok")]
    result["delta"] = {key: after[key] - before[key] for key in before}
    result["first_step"] = _phase_stats(records, lambda record: record["step"] == 0)
    result["continuation"] = _phase_stats(records, lambda record: record["step"] > 0)
    result["all_steps"] = _phase_stats(records, lambda record: True)
    result["lag"] = _stats([record["lag_ms"] for record in records])
    result["records"] = records
    result["rc_engine"] = engine.returncode
    result["rc_server"] = server.returncode
    result["mode_lines"] = mode_lines(tag)
    result["ledger"] = grep_final_counters(tag) or {}
    result["ledger_lines"] = len(grep_ledgers(tag))
    result["tracebacks"] = grep_tracebacks(tag)
    result["warnings"] = [
        warning for warning in grep_warnings(tag) if driver._WARN_NO_SESSION not in warning
    ]
    out = LOGDIR / f"ag_{tag}.json"
    out.write_text(json.dumps(result, indent=1))
    delta = result["delta"]
    external = delta["ext_hits"] / delta["ext_queries"] if delta["ext_queries"] else float("nan")
    print(
        f"[ag] {tag}: {result['requests']} requests in {elapsed:.1f}s, "
        f"external hit {external:.3f}, cycles {result['cycles']}, "
        f"failed {len(result['failed'])}, wrote {out}"
    )
    return result


def main() -> int:
    """Run one point named by argv: `agentic.py <config> [rep] [sessions]`."""
    config = sys.argv[1] if len(sys.argv) > 1 else "lazy"
    rep = sys.argv[2] if len(sys.argv) > 2 else "0"
    sessions = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    result = run(config, rep, sessions)
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
