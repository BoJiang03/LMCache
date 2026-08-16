#!/usr/bin/env python3
"""Layer-1 GPU smoke driver for lazy offloading.

Usage: driver.py <S1|S2|S3|S4|S6|S9|S11|S12|S13|S14|S15|S16|S17|S18|S19>

Each scenario: start MP server + vllm serve, fire prompts, assert via the
MP server HTTP API (/cache/objects, /status), tear down, grep logs.
Exit code 0 = all assertions passed.
"""

import http.client
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# The reproduction package lives at <repo>/repro/pr4499. Every local default
# is derived from that location or the active Python environment; reviewers
# should not need to edit machine-specific paths in this file.
REPO = os.environ.get("SMOKE_REPO", str(Path(__file__).resolve().parents[2]))
PY = os.environ.get("SMOKE_PYTHON", sys.executable)
VLLM = os.environ.get("SMOKE_VLLM", shutil.which("vllm") or "vllm")
GPU = os.environ.get("SMOKE_GPU", "0")
CPATH = os.environ.get("CPATH", "")
MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-0.6B")

#: A hybrid-attention model: 18 layers, 5 sliding-window (512 tokens) for
#: every 1 full-attention, so vLLM builds several KV cache groups and puts
#: `block_pool.null_block` -- which has no block hash -- in the block table
#: for out-of-window positions. S18 is the only scenario that needs it.
SWA_MODEL = "google/gemma-3-270m-it"

#: The model of the engine currently up. `start_vllm` sets it, and the HTTP
#: helpers read it rather than MODEL: only one engine runs at a time, and a
#: request naming a model the running engine does not serve is a 404 the
#: scenario would have to interpret. Threading the name through every
#: `complete` call site instead would put it in ~80 places for the sake of
#: the one scenario that changes it.
_running_model = MODEL

MP_PORT = int(os.environ.get("SMOKE_MP_PORT", "26555"))
HTTP_PORT = int(os.environ.get("SMOKE_HTTP_PORT", "28085"))
VLLM_PORT = int(os.environ.get("SMOKE_VLLM_PORT", "28100"))

BASE = Path(__file__).parent
LOGDIR = Path(os.environ.get("SMOKE_LOGDIR", str(BASE / "logs")))


def http_get(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        raw = r.read()
    if not raw:
        return {}  # vllm /health returns an empty 200
    return json.loads(raw)


def wait_for(url: str, deadline: float, name: str) -> None:
    t0 = time.time()
    while time.time() - t0 < deadline:
        try:
            http_get(url)
            print(f"[driver] {name} ready after {time.time() - t0:.0f}s")
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"{name} not ready within {deadline}s")


def server_status() -> dict:
    """The MP server's /status document."""
    return http_get(f"http://127.0.0.1:{HTTP_PORT}/status")


def cache_object_count() -> int:
    status = server_status()
    return int(status["storage_manager"]["l1_manager"]["total_object_count"])


def active_sessions() -> int:
    """Sessions the MP server still holds open (`/status`).

    A session is created by the first lookup or store under a request id
    and removed by `end_session`. In lazy mode that call is deferred until
    the request's buffered ops leave the queue, so the count is the direct
    reading of the deferred-teardown contract: with an empty pending queue
    it must be zero, and a leaked session -- or a teardown that never fired
    -- shows up here even though every other assertion passes.
    """
    return int(server_status()["active_sessions"])


def chunk_size() -> int:
    """The server's chunk size: stores and retrievals are whole chunks."""
    return int(server_status()["chunk_size"])


def prompt_tokens(prompt: str) -> int:
    """The token count vLLM assigns to a prompt (`POST /tokenize`).

    `add_special_tokens` defaults to True on both /tokenize and
    /v1/completions, so this is the same count the engine prefills.

    Args:
        prompt: The prompt to tokenize. The vllm server must be up.

    Returns:
        The number of prompt tokens.
    """
    body = json.dumps({"model": _running_model, "prompt": prompt}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}/tokenize",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(json.loads(r.read())["count"])


def stored_tokens(prompt: str) -> int:
    """Tokens LMCache stores for a prompt: its full chunks, measured.

    Only whole chunks are stored, so this floors the measured token count
    to the server's chunk size. Retrieving exactly this much is the only
    evidence that *every* chunk of the prefix survived -- a `>= 1024` style
    floor also passes when half of it was lost.

    Measured rather than tabulated: `long_prompt` embeds its seed twice per
    sentence, so the token count depends on the seed's length as well as on
    the sentence count (100 sentences is 1965 tokens with a 6-character
    seed but 2165 with a 2-character one, which straddles a chunk
    boundary). A hardcoded constant silently becomes a false failure that
    looks like a policy bug.

    Args:
        prompt: The prompt whose stored prefix length is wanted. Both
            servers must be up.

    Returns:
        The number of tokens LMCache stores for the prompt.
    """
    chunk = chunk_size()
    return prompt_tokens(prompt) // chunk * chunk


#: Default (empty) env overlay for the process starters; never mutated.
_NO_EXTRA_ENV: dict[str, str] = {}


def start_server(
    scenario: str, extra_env: dict[str, str] = _NO_EXTRA_ENV
) -> subprocess.Popen:
    """Start the LMCache MP server and wait for its healthcheck.

    Args:
        scenario: Names the server's log file.
        extra_env: Environment overlay, applied last so it can override the
            defaults. A tensor-parallel scenario has to widen
            CUDA_VISIBLE_DEVICES here: the server resolves each worker's KV
            caches through CUDA IPC by device UUID, so a server that sees
            only rank 0's device fails rank 1's `register_kv_caches` with
            "Device UUID ... not found in the discovered devices" and the
            engine dies after a 300s timeout.

    Returns:
        The running server process.
    """
    log = open(LOGDIR / f"{scenario}_server.log", "w")
    env = dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES=GPU)
    env.update(extra_env)
    proc = subprocess.Popen(
        [
            PY, "-m", "lmcache.v1.multiprocess.http_server",
            "--host", "127.0.0.1", "--port", str(MP_PORT),
            "--http-host", "127.0.0.1", "--http-port", str(HTTP_PORT),
            "--l1-size-gb", "8", "--eviction-policy", "LRU",
            "--script-allowed-imports", "hashlib",
            "--max-workers", "4",
        ],
        stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO,
    )
    wait_for(f"http://127.0.0.1:{HTTP_PORT}/healthcheck", 60, "mp-server")
    return proc


def start_vllm_under(server: subprocess.Popen, scenario: str, extra_config: dict, vllm_args: list[str], extra_env: dict[str, str] = _NO_EXTRA_ENV, model: str = "") -> subprocess.Popen:
    try:
        return start_vllm(scenario, extra_config, vllm_args, extra_env, model)
    except Exception:
        teardown([server])
        raise


def start_vllm(scenario: str, extra_config: dict, vllm_args: list[str], extra_env: dict[str, str] = _NO_EXTRA_ENV, model: str = "") -> subprocess.Popen:
    """Start `vllm serve` under the MP connector and wait for its health.

    Args:
        scenario: Names the engine's log file.
        extra_config: Merged into `kv_connector_extra_config`.
        vllm_args: Appended to the command line.
        extra_env: Environment overlay, applied last.
        model: Model to serve; empty means MODEL. A scenario that names one
            here also redirects the HTTP helpers to it (`_running_model`),
            so its prompts are tokenized by the model that will prefill
            them -- token counts are not portable across tokenizers.

    Returns:
        The running engine process.
    """
    global _running_model
    _running_model = model or MODEL
    log = open(LOGDIR / f"{scenario}_vllm.log", "w")
    env = dict(
        os.environ,
        PYTHONPATH=REPO,
        CUDA_VISIBLE_DEVICES=GPU,
        VLLM_SERVER_DEV_MODE="1",
        CPATH=CPATH,
        PYTHONFAULTHANDLER="1",
    )
    env.update(extra_env)  # update, not **: a scenario may override the above
    kv_cfg = {
        "kv_connector": "LMCacheMPConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "lmcache.mp.host": "tcp://127.0.0.1",
            "lmcache.mp.port": MP_PORT,
            **extra_config,
        },
    }
    proc = subprocess.Popen(
        [
            VLLM, "serve", _running_model,
            "--port", str(VLLM_PORT),
            "--max-model-len", "8192",
            "--enforce-eager",
            "--kv-transfer-config", json.dumps(kv_cfg),
            *vllm_args,
        ],
        stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO,
    )
    try:
        wait_for(f"http://127.0.0.1:{VLLM_PORT}/health", 600, "vllm")
    except Exception:
        teardown([proc])
        raise
    return proc


def complete(prompt: str, max_tokens: int = 32, request_id: str = "") -> str:
    """Greedy completion. A non-empty request_id fixes the engine's id.

    vLLM derives the request id from the X-Request-Id header
    (entrypoints/openai/engine/serving.py::_base_request_id), giving
    `cmpl-{request_id}-0`. The engine then appends 8 random characters to
    guarantee uniqueness, so the id only reaches the connector verbatim --
    and only then can two requests collide on it -- when the engine runs
    with VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 (see :func:`scenario_S12`).
    Either way the id stays a stable substring, so log lines can be matched
    to the request that produced them.
    """
    body = json.dumps(
        {
            "model": _running_model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-Id"] = request_id
    req = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}/v1/completions",
        data=body,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["text"]


def complete_then_disconnect(
    prompt: str, request_id: str, tokens_before_close: int = 5
) -> str:
    """Stream a completion, then drop the connection mid-generation.

    Closing the socket while the engine still has decode work left makes
    vLLM abort the request (the API server's cancellation path calls
    AsyncLLM.abort), which is the only way to reach the connector's abort
    handling from a client.

    ignore_eos pins the premise: with 512 tokens to generate and no early
    stop possible, the request provably still had work left when the socket
    closed. Without it an early EOS would finish the request normally while
    `AsyncLLM.abort` still logs the line (it logs its *input* ids, after
    `output_processor.abort_requests` has already dropped ids whose state is
    gone), so the abort evidence would not follow from the log line alone.

    Args:
        prompt: The prompt to stream.
        request_id: Fixes the engine request id (see :func:`complete`), so
            the abort can be matched in the vllm log.
        tokens_before_close: SSE deltas to consume before disconnecting.

    Returns:
        The text streamed before the disconnect (greedy, so a full replay
        of the same prompt must start with it).
    """
    body = json.dumps(
        {
            "model": _running_model,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
        }
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", VLLM_PORT, timeout=120)
    text = ""
    try:
        conn.request(
            "POST", "/v1/completions", body=body,
            headers={"Content-Type": "application/json",
                     "X-Request-Id": request_id},
        )
        resp = conn.getresponse()
        deltas = 0
        while deltas < tokens_before_close:
            line = resp.readline()
            if not line:
                break
            payload = line.decode().strip()
            if not payload.startswith("data: ") or payload == "data: [DONE]":
                continue
            choices = json.loads(payload[6:])["choices"]
            if not choices:  # a usage-only chunk carries no text
                continue
            text += choices[0]["text"]
            deltas += 1
    finally:
        conn.close()  # abrupt: no [DONE] read, so the request is unfinished
    return text


def vllm_metric(name: str) -> float:
    """Sum a vllm Prometheus counter over its label sets (`/metrics`).

    Args:
        name: The full sample name, `_total` suffix included.

    Returns:
        The sum of every exposed sample of that name.

    Raises:
        RuntimeError: if no sample of that name is exposed. The absence of
            a metric must not read as a zero delta: a renamed or disabled
            metric would otherwise satisfy every "unchanged" assertion.
    """
    with urllib.request.urlopen(
        f"http://127.0.0.1:{VLLM_PORT}/metrics", timeout=10
    ) as r:
        body = r.read().decode()
    samples = [
        float(line.rsplit(" ", 1)[1])
        for line in body.splitlines()
        if line.startswith(name) and not line.startswith("#")
    ]
    if not samples:
        raise RuntimeError(f"{name} absent from /metrics")
    return sum(samples)


def vllm_generation_tokens() -> float:
    """Total tokens vllm has generated since boot (`vllm:generation_tokens`).

    A process-wide counter, so it only isolates one request's generation
    when nothing else is running -- which is how the abort check uses it:
    an abort that never reached the scheduler leaves the engine generating
    to max_tokens, and the delta separates the two cases by two orders of
    magnitude.

    Raises:
        RuntimeError: if the metric is absent from /metrics.
    """
    return vllm_metric("vllm:generation_tokens_total")


#: vLLM's own accounting of what the KV connector was asked and what it
#: answered, in tokens: queries is every token the scheduler had to look up
#: externally (prompt tokens minus the local APC hit), hits is the part the
#: connector claimed. A miss is only provable with both: `queries` rising
#: while `hits` stays flat says the lookup ran and found nothing, which is
#: what distinguishes "the store was withheld" from "no lookup happened".
_METRIC_EXT_QUERIES = "vllm:external_prefix_cache_queries_total"
_METRIC_EXT_HITS = "vllm:external_prefix_cache_hits_total"


def long_prompt(seed: str, sentences: int) -> str:
    """Deterministic long prompt; ~19.7 Qwen3 tokens per sentence.

    Measured: 50 sentences = 965 tokens, 100 = 1965, 150 = 3072. Scenarios
    that reason about the min_prefix_tokens gate need the real number --
    100 sentences sits just *below* a 2048-token gate, not above it.
    """
    return " ".join(
        f"Document {seed} section {i} explains rule number {i * 7 + 3} "
        f"about topic {seed}-{i}." for i in range(sentences)
    )


#: Grace window each process gets, on its own clock, to exit after SIGINT.
_GRACE_SECONDS = 30

#: rc 0 = clean exit, -2 = terminated by our SIGINT. -9 means the process
#: hung past the grace window and had to be SIGKILLed, which is how a
#: shutdown hang with pending sessions surfaces.
_CLEAN_RCS = (0, -2)


def teardown(procs: list[subprocess.Popen]) -> None:
    """SIGINT every process, then give each one its own grace window.

    The deadline is per process, not shared across the list: with one clock
    a slow engine could eat the whole window and the next process -- the MP
    server, whose log carries the retrieval evidence -- got SIGKILLed
    without a chance to flush or to exit cleanly, which also destroys its
    exit code as a signal.
    """
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    for p in procs:
        t0 = time.time()
        while p.poll() is None and time.time() - t0 < _GRACE_SECONDS:
            time.sleep(1)
        if p.poll() is None:
            p.kill()
    print("[driver] teardown done, exit codes:", [p.returncode for p in procs])


def expect_clean_exit(c: "Check", name: str, proc: subprocess.Popen) -> None:
    """Assert a torn-down process exited within the SIGINT grace window.

    Args:
        c: The scenario's check recorder.
        name: How to identify the process in the verdict line.
        proc: An already torn-down process.
    """
    c.expect(
        proc.returncode in _CLEAN_RCS,
        f"{name} shut down within the SIGINT grace window "
        f"(rc={proc.returncode})",
    )


#: Exception types whose tracebacks are our own SIGINT teardown propagating
#: out of uvicorn's asyncio.run -- fired after "Application shutdown
#: complete", inherent to stopping the server, and not a failure signal.
_BENIGN_FINAL_EXC = ("KeyboardInterrupt", "asyncio.exceptions.CancelledError")

#: First line of the one ERROR block that our own teardown provokes: vllm's
#: shutdown force-kills the engine core ("Process manager: force killing
#: remaining processes") while the API server's output handler is still
#: awaiting it, so the handler logs EngineDeadError. Intermittent -- it
#: depends on whether the kill beats the handler's cancellation -- and it is
#: exempted only inside the shutdown region and only when the block really
#: ends in EngineDeadError. The same error during a run still fails, as does
#: any other ERROR at shutdown (an LMCache session teardown error, say).
_TEARDOWN_ERROR_BLOCK = "AsyncLLM output_handler failed."


def _scenario_logs(scenario: str) -> list[Path]:
    """Every engine/server log the scenario produced.

    The four possible names are listed, not globbed. Phase-B engines log to
    {scenario}b_vllm.log (S6b, S11b, S14b), and everything else whose name
    starts with the scenario's has to stay out: this harness's transcripts
    quote the very lines the scans below report, so including one lets a
    failure report feed itself.

    `{scenario}_*.log` minus `_driver.log` was not enough. It also matched
    any transcript a caller redirected into the log directory, and a stale
    `S14_run.log` from a previous run -- holding that run's two quoted
    session warnings -- doubled the next sweep's warning count to 4.
    """
    names = [
        f"{scenario}{suffix}_{kind}.log"
        for suffix in ("", "b")
        for kind in ("vllm", "server")
    ]
    return sorted(p for p in (LOGDIR / name for name in names) if p.exists())


def grep_tracebacks(scenario: str) -> list[str]:
    """Unexpected Traceback/ERROR lines across the scenario's logs."""
    return [
        hit
        for f in _scenario_logs(scenario)
        for hit in scan_tracebacks(f.name, f.read_text())
    ]


def scan_tracebacks(name: str, text: str) -> list[str]:
    """Unexpected Traceback/ERROR lines in one log's text.

    Args:
        name: The log's name, for the `name:line:` prefix of each hit.
        text: The log's contents.

    Returns:
        One `name:line: text` entry per hit, in file order.
    """
    hits = []
    lines = text.splitlines()
    in_shutdown = False
    in_teardown_block = False
    for i, line in enumerate(lines):
        if "[shutdown]" in line:
            in_shutdown = True
        if in_teardown_block:
            # The block is one logger.exception call: every line of it
            # carries the ERROR level. The first line without it ends it.
            if "ERROR" in line:
                continue
            in_teardown_block = False
        if (
            in_shutdown
            and _TEARDOWN_ERROR_BLOCK in line
            and any("EngineDeadError" in ln for ln in lines[i : i + 40])
        ):
            in_teardown_block = True
            continue
        if "Traceback" in line:
            # Find the block's final exception line: the next non-indented,
            # non-"File"/source line after the frames.
            final = ""
            for follow in lines[i + 1 : i + 40]:
                if follow[:1] not in (" ", "\t") and follow.strip():
                    final = follow.strip()
                    break
            if any(final.startswith(exc) for exc in _BENIGN_FINAL_EXC):
                continue
            hits.append(f"{name}:{i + 1}: {line.strip()[:160]}")
        elif "ERROR" in line:
            hits.append(f"{name}:{i + 1}: {line.strip()[:160]}")
    return hits


#: WARNING texts inherent to how the harness configures its processes, not
#: signals: vllm's boot-time notices, the shutdown force-kill, our own
#: connector's preemption notice (the small-pool scenarios preempt by
#: construction), and the L1 force-clear that `cache_clear()` requests.
#: Everything else fails the scenario -- the scan exists because
#: `Session %s not found, skipping touch` sat in five of seven scenario
#: logs while all seven reported ALL PASS, the ERROR/Traceback scan being
#: blind to WARNING.
_BENIGN_WARNINGS = (
    "Inductor compilation was disabled",
    "Enforce eager set",
    "Initializing KVConnectorBase_V1",
    "Process manager: force killing remaining processes",
    "Default vLLM sampling parameters have been overridden",
    "Development endpoints are enabled",
    "Found duplicate keys",
    "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set",
    "<preempted> by preempted requests",
    "force-clearing all",
    # Only appears with the multiproc executor, i.e. TP > 1.
    "Reducing Torch parallelism",
    # The cap-sizing warning. Allow-listed rather than fatal because S14
    # exists to provoke it: phase B asserts it appears and phase A, on the
    # default cap, asserts it does not. No other scenario changes the cap,
    # and every scenario asserts throttled_drains through its ledger.
    "held back",
    # The hash-less-block skip. S18 provokes it and asserts it in both
    # directions -- exactly one line, with its request id and span, in the
    # lazy phase; none in the eager one. It cannot occur in any other
    # scenario: they all serve a full-attention model, whose block tables
    # never contain vLLM's hash-less null block.
    "skipping store for request",
    # vLLM's own performance notice, first seen on the sliding-window model
    # of S18. It reports a JIT compile inside a forward pass and says
    # nothing about LMCache.
    "Triton kernel JIT compilation during inference",
)


#: The MP server's warning when `end_session` names a request it has no
#: session for (`lookup.py`). Not allow-listed by text, because the same
#: text would cover a real double-teardown: it is counted instead, against
#: the number of requests each scenario knows cannot have a session.
#:
#: A session is created by the lookup handler, but only after
#: `compute_chunk_hashes` returns something -- a prompt shorter than one
#: chunk yields no hashes and the handler returns before `get_or_create`.
#: The connector still calls `end_session` when such a request finishes, so
#: the warning is exactly one per sub-chunk request. Every one of them here
#: is a ledger-flush request, which is sub-chunk by design (it must admit no
#: op of its own). Nothing is lost: the request stored nothing, so the touch
#: that is skipped had nothing to touch.
#:
#: This is why the warning first looked like it correlated with pressure and
#: preemption: only the pressure scenarios had flush requests. Adding one to
#: S2 -- no pressure, no preemption, one request queue that never drains --
#: produced exactly one warning, and S2 had never had one before.
_WARN_NO_SESSION = "not found, skipping touch"

#: Sub-chunk (hence sessionless) requests per scenario, all of them ledger
#: flushes: S11 flushes once per phase, S1 and S4 do not flush at all.
_SESSIONLESS_REQUESTS = {
    "S1": 0,
    "S2": 1,
    "S3": 1,
    "S4": 1,
    "S6": 1,
    "S9": 1,
    "S11": 2,
    "S12": 1,
    "S13": 1,
    "S14": 2,  # one ledger flush per phase
    "S15": 1,
    "S16": 1,
    "S17": 1,
    "S18": 2,  # one ledger flush per lazy-phase reading
    "S19": 1,
}


def grep_warnings(scenario: str) -> list[str]:
    """WARNING lines in the scenario's logs that are not allow-listed.

    Args:
        scenario: The scenario whose engine and server logs to scan.

    Returns:
        One `file:line: text` entry per unexpected WARNING, in file order.
    """
    hits = []
    for f in _scenario_logs(scenario):
        for i, line in enumerate(f.read_text().splitlines()):
            if "WARNING" not in line:
                continue
            if any(benign in line for benign in _BENIGN_WARNINGS):
                continue
            hits.append(f"{f.name}:{i + 1}: {line.strip()[:160]}")
    return hits


def grep_retrieved(scenario: str) -> list[int]:
    """Token counts from the MP server's 'Retrieved N tokens' INFO lines.

    Raises:
        FileNotFoundError: if the log is missing. Negative assertions are
            built on this returning an empty list, so a missing log must
            fail loudly instead of making them vacuously true.
    """
    log = LOGDIR / f"{scenario}_server.log"
    if not log.exists():
        raise FileNotFoundError(f"no server log for {scenario}: {log}")
    return [
        int(m.group(1))
        for m in re.finditer(r"Retrieved (\d+) tokens", log.read_text())
    ]


def grep_stored(scenario: str) -> list[int]:
    """Token counts from the MP server's 'Stored N tokens' INFO lines.

    One line per store submission, so the *sizes* read out how the policy
    batched its emissions: a request whose ops are emitted together are
    coalesced into a single store and appear as one large line, while ops
    emitted on separate steps appear as one line each. This is the only
    place a drain's batching is observable -- the policy logs drops but not
    emissions -- and the server line carries no request id, so a scenario
    must bracket the window it cares about by index.

    Raises:
        FileNotFoundError: if the log is missing. A missing log reads as
            "no stores", which satisfies any upper bound on store size.
    """
    log = LOGDIR / f"{scenario}_server.log"
    if not log.exists():
        raise FileNotFoundError(f"no server log for {scenario}: {log}")
    return [
        int(m.group(1)) for m in re.finditer(r"Stored (\d+) tokens", log.read_text())
    ]


def grep_lines(scenario: str, pattern: str) -> list[str]:
    """Lines of the scenario's vllm log matching a regex.

    Used for evidence that a specific engine/connector path executed (an
    abort, an id reclaim) rather than for counting: a scenario that asserts
    on the count of a periodic line must snapshot it before and after.

    Raises:
        FileNotFoundError: if the log is missing. Scenarios assert that
            certain patterns are *absent*, so an unreadable log must fail
            loudly rather than satisfy them vacuously.
    """
    log = LOGDIR / f"{scenario}_vllm.log"
    if not log.exists():
        raise FileNotFoundError(f"no vllm log for {scenario}: {log}")
    rx = re.compile(pattern)
    return [ln.strip() for ln in log.read_text().splitlines() if rx.search(ln)]


#: The pending store's init line, one per engine that enabled lazy offload.
#: Absent entirely when the connector runs eager.
_MODE_LINE = "lazy offload enabled"


def mode_lines(scenario: str) -> list[str]:
    """The engine's ``lazy offload enabled ...`` init lines, trimmed.

    The direct evidence that a scenario's policy configuration was honoured.
    Most scenarios prove it by consequence -- the eviction-aware policy
    withholds stores no other mode withholds -- but FIFO does not: with
    threshold 1 it stores every chunk, exactly as the eager path does, so a
    silently ignored `lazy_offload` flag produces the same object count,
    the same retrievals and the same outputs.

    Args:
        scenario: The scenario whose engine log to scan.

    Returns:
        One trimmed line per match, in file order.

    Raises:
        FileNotFoundError: if the engine log is missing. Callers assert this
            list is *empty* as well as non-empty, so an unreadable log must
            fail loudly rather than satisfy the negative form vacuously.
    """
    log = LOGDIR / f"{scenario}_vllm.log"
    if not log.exists():
        raise FileNotFoundError(f"no vllm log for {scenario}: {log}")
    return [
        ln.strip()[:200] for ln in log.read_text().splitlines() if _MODE_LINE in ln
    ]


_LEDGER_LINE = re.compile(r"Lazy offload (final )?counters: (.+)")


def _vllm_log(scenario: str) -> str:
    """The scenario's engine log text.

    Raises:
        FileNotFoundError: if the log is missing. Every ledger and drop-line
            reading below is a comparison against a count, and a missing log
            reads as zero of everything -- which satisfies most of them.
    """
    log = LOGDIR / f"{scenario}_vllm.log"
    if not log.exists():
        raise FileNotFoundError(f"no vllm log for {scenario}: {log}")
    return log.read_text()


def _log_through_last_ledger(scenario: str) -> str:
    """Return the log text up to and including the last ledger line.

    Everything the policy logs about a drain (the aggregate drop lines) is
    emitted before that drain's ledger line, so truncating here puts the
    drop lines and the counters on the same instant. Lines after it belong
    to drains the ledger snapshot never saw.
    """
    text = _vllm_log(scenario)
    last = None
    for m in _LEDGER_LINE.finditer(text):
        last = m
    return text[: last.end()] if last is not None else ""


def grep_ledgers(scenario: str) -> list[tuple[str, dict[str, int]]]:
    """Every ledger line up to the last one, as (kind, counters) pairs.

    The scheduler logs 'Lazy offload counters: k=v ...' periodically on
    drains and 'Lazy offload final counters: ...' from the connector's
    shutdown hook. Scenarios flush the periodic line with a trailing tiny
    request, so the last pair is the settled ledger.

    Args:
        scenario: The scenario whose engine log to parse.

    Returns:
        One ("periodic" | "final", counters) pair per ledger line, in log
        order.
    """
    return [
        ("final" if m.group(1) else "periodic",
         {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m.group(2))})
        for m in _LEDGER_LINE.finditer(_log_through_last_ledger(scenario))
    ]


def grep_final_counters(scenario: str) -> dict[str, int] | None:
    """Parse the settled (last) policy ledger line from the vllm log.

    Args:
        scenario: The scenario whose engine log to parse.

    Returns:
        The counters of the last ledger line, or None if the log has none.
    """
    ledgers = grep_ledgers(scenario)
    return ledgers[-1][1] if ledgers else None


#: Drop causes, as they appear in the aggregate INFO line's reason clause.
_CAUSE_EVICTED = "blocks evicted before drain"
_CAUSE_SHORT_PREFIX = "request prefix below the break-even length"


def count_drop_lines(scenario: str, cause: str) -> int:
    """Sum the per-drain aggregate 'dropped N store op(s)' lines of one cause.

    Counts only lines *before the last ledger line* so both sides describe
    the same instant: the ledger is usually a periodic snapshot, and a drop
    that lands after it used to fail the cross-check with entirely correct
    behavior.

    (The per-op detail lines are DEBUG and absent at the default level;
    their text has no digit after 'dropped', so they can't double-count.)

    Args:
        scenario: The scenario whose vllm log to read.
        cause: The reason clause to match, one of the ``_CAUSE_*``
            constants. Each cause has its own ledger counter, so they must
            not be summed together.

    Returns:
        The total number of ops reported dropped for that cause.
    """
    pattern = rf"dropped (\d+) store op\(s\): {re.escape(cause)}"
    return sum(
        int(n) for n in re.findall(pattern, _log_through_last_ledger(scenario))
    )


#: Ledger keys that count buffered ops leaving the queue. (The rejected_*
#: keys other than short_prefix are admission-time: those ops were never
#: admitted, so they don't count against admissions.)
_LEDGER_OUTCOMES = (
    "emitted", "dropped_evicted", "rejected_short_prefix",
    "dropped_on_request_drop", "dropped_failed_store", "dropped_id_reuse",
)

#: Every key the ledger line must carry. Asserted as a set, because the
#: counters used to be read with `.get(key, 0)`: a renamed or dropped key
#: then read as zero, which is exactly what most of these assertions want
#: to see. A missing key must fail instead.
#: `throttled_drains` counts *drains*, not ops, so it is deliberately not a
#: ledger outcome: adding it to the equation would break the accounting it
#: exists to explain.
_LEDGER_KEYS = frozenset(
    _LEDGER_OUTCOMES
    + ("admitted", "pending", "deduplicated",
       "rejected_unhashed", "rejected_prefix_broken", "throttled_drains")
)


def check_ledger(c: "Check", scenario: str, max_evicted: int) -> dict[str, int]:
    """Shared ledger assertions after teardown.

    The ledger must exist, carry every key, close as an equation, agree per
    cause with the aggregate drop INFO lines, and stay within the
    scenario's eviction-drop bound.

    The equation is `admitted == pending + emitted + every drop counter`:
    the policy logs the queue depth alongside the counters, taken from the
    same snapshot, so an op that left the queue without incrementing any
    outcome counter shows up here. (Before the ledger carried `pending`
    this could only be `outcomes <= admitted`, which catches over-counting
    only.)

    The drop-line cross-check is weaker than it looks and is kept for what
    it does catch: both sides originate in the same `DrainResult`, the line
    printing the length of the same list the counter was incremented by, so
    it cannot detect an accounting divergence inside the policy. What it
    does detect is the log going out of sync with the counters -- a renamed
    or reworded cause, a throttle added to the aggregate line, a truncation
    that swallows ops -- which is what every per-request-id drop grep in
    this file depends on.

    The ledger is usually the *periodic* line, not the final one: `vllm
    serve`'s SIGINT abort mode force-kills the engine core before
    `log_final_stats` runs, so scenarios flush the periodic line with a
    trailing sub-chunk request. Both sides of the drop cross-check are
    therefore truncated at that line (see `count_drop_lines`). The kind of
    every ledger line is printed rather than assumed, so the standing
    layer-1 gap (no scenario has yet produced a `final counters` line) is
    visible in the transcript instead of hidden behind a disjunction.

    Args:
        c: The scenario's check recorder.
        scenario: The scenario whose engine log to read.
        max_evicted: Upper bound on `dropped_evicted` for this scenario,
            from its measured value plus documented headroom. There is no
            configuration in which the bound is simply 0: a capped step
            budget keeps steady-state allocation inside the drain horizon,
            but pinned pending blocks shrink the usable pool enough to
            preempt even serial requests, and a resumed request re-produces
            its ops into an already deep queue where one can be lost at the
            margin. A bound is still the sensor for gate 1 -- unbounded, a
            regression that drops every op reads as ALL PASS.

    Returns:
        The settled ledger's counters, or an empty dict if none was logged.
    """
    ledgers = grep_ledgers(scenario)
    c.expect(ledgers != [], f"a counter ledger was logged ({len(ledgers)} lines)")
    if not ledgers:
        return {}
    kinds = [kind for kind, _ in ledgers]
    print(
        f"[{scenario}] ledger lines: periodic={kinds.count('periodic')} "
        f"final={kinds.count('final')}"
    )
    ledger = ledgers[-1][1]
    print(f"[{scenario}] ledger: " + " ".join(f"{k}={v}" for k, v in ledger.items()))
    c.expect(
        set(ledger) == set(_LEDGER_KEYS),
        f"the ledger carries exactly the expected keys "
        f"(missing={sorted(_LEDGER_KEYS - set(ledger))}, "
        f"unexpected={sorted(set(ledger) - _LEDGER_KEYS)})",
    )
    if set(ledger) != set(_LEDGER_KEYS):
        return ledger
    # A final line, when one exists, must agree with the last periodic one:
    # nothing drains between the flush request and shutdown.
    finals = [counters for kind, counters in ledgers if kind == "final"]
    periodics = [counters for kind, counters in ledgers if kind == "periodic"]
    c.expect(
        periodics != [],
        f"the periodic ledger line -- the one every check here reads -- was "
        f"logged ({len(periodics)} lines)",
    )
    if finals and periodics:
        c.expect(
            finals[-1] == periodics[-1],
            f"the shutdown ledger agrees with the last periodic one "
            f"(final={finals[-1]}, periodic={periodics[-1]})",
        )
    admitted = ledger["admitted"]
    evicted = ledger["dropped_evicted"]
    pending = ledger["pending"]
    outcomes = sum(ledger[k] for k in _LEDGER_OUTCOMES)
    c.expect(
        admitted == pending + outcomes,
        f"ledger closes as an equation (admitted={admitted}, "
        f"pending={pending}, outcomes={outcomes})",
    )
    evicted_lines = count_drop_lines(scenario, _CAUSE_EVICTED)
    c.expect(
        evicted_lines == evicted,
        f"aggregate eviction-drop INFO lines sum to the ledger "
        f"(sum={evicted_lines}, dropped_evicted={evicted})",
    )
    short_prefix_lines = count_drop_lines(scenario, _CAUSE_SHORT_PREFIX)
    c.expect(
        short_prefix_lines == ledger["rejected_short_prefix"],
        f"aggregate gate-3 drop INFO lines sum to the ledger "
        f"(sum={short_prefix_lines}, "
        f"rejected_short_prefix={ledger['rejected_short_prefix']})",
    )
    c.expect(
        evicted <= max_evicted,
        f"eviction drops stay within this scenario's bound "
        f"(dropped_evicted={evicted}, max={max_evicted})",
    )
    return ledger


class Check:
    """Assertion recorder: counts what ran, not only what failed.

    The count is what makes an ALL PASS legible. A scenario that bailed out
    of a block -- an empty ledger short-circuiting `check_ledger`, a key set
    that did not match, an early `return` -- otherwise prints exactly the
    same verdict as one that ran every assertion, and `main` has no way to
    tell the two apart.
    """

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.executed = 0

    def expect(self, cond: bool, msg: str) -> None:
        """Record one assertion, printing its verdict."""
        self.executed += 1
        tag = "PASS" if cond else "FAIL"
        print(f"[check] {tag}: {msg}")
        if not cond:
            self.failures.append(msg)


def scenario_S1() -> Check:
    """Lazy off: eager store happens, replay served, outputs saved as baseline.

    The control for every lazy scenario: the same prompts through the same
    connector with the lazy flag off must store eagerly, and a replay must
    come back out of LMCache with the exact stored prefix. Its second job is
    to pin the eager path's teardown contract -- with no deferral, the
    server must hold no session once the engine is down.
    """
    c = Check()
    server = start_server("S1")
    vllm = start_vllm_under(server, "S1", {}, ["--gpu-memory-utilization", "0.5"])
    try:
        prompts = [long_prompt(s, 100) for s in ("alpha", "beta", "gamma")]
        # One L1 object per stored chunk, and the eager path stores every
        # full chunk of every prompt: an exact count, since there is no
        # pressure here for anything to be dropped by. `objects > 0` would
        # also pass with two thirds of the prefixes missing. Summed per
        # prompt: the seeds differ in length, so the prompts differ in
        # token count.
        chunk = chunk_size()
        expected = sum(stored_tokens(p) // chunk for p in prompts)
        outs = [complete(p) for p in prompts]
        time.sleep(5)  # let async stores land
        n = cache_object_count()
        c.expect(
            n == expected,
            f"the eager path stored every chunk of every prompt "
            f"(objects={n}, expected={expected})",
        )
        replay = complete(prompts[0])
        c.expect(replay == outs[0], "replay of prompt alpha reproduces greedy output")
        (BASE / "s1_baseline.json").write_text(
            json.dumps({"prompts": prompts, "outputs": outs})
        )
        status = server_status()
        (LOGDIR / "S1_status.json").write_text(
            json.dumps(status, indent=2, default=str)
        )
    finally:
        teardown([vllm])
    expect_clean_exit(c, "vllm", vllm)
    try:
        # Eager mode defers no teardown, so every session opened by these
        # four requests must already be closed.
        sessions = active_sessions()
        c.expect(
            sessions == 0,
            f"the eager path left no session open (active_sessions={sessions})",
        )
    finally:
        teardown([server])
    expect_clean_exit(c, "mp-server", server)
    return c


def scenario_S2() -> Check:
    """Lazy on, big pool, no pressure: zero stores, requests finish cleanly.

    Four 30-sentence prompts against a pool nothing presses on. Each is one
    prefill step, so each admits exactly one op, and all four must still be
    queued at the end: nothing emitted, nothing dropped, four sessions held
    open by the deferred teardown.

    The ledger has to be flushed deliberately here. Drains only run on steps
    that schedule tokens, so once the last completion returns the counters
    stop being reprinted -- and the settled state has to be read from a line
    logged *after* the last request, not from the one the first request
    happened to trigger. (It was: the one counters line in this scenario's
    log predated all four `POST 200`s, so `emitted == 0` was being verified
    against a snapshot taken one second into a thirteen-second scenario.)
    """
    c = Check()
    server = start_server("S2")
    vllm = start_vllm_under(server, "S2",
        {"lmcache.mp.lazy_offload": True},
        ["--gpu-memory-utilization", "0.5"],
    )
    try:
        prompts = [long_prompt(s, 30) for s in ("p1", "p2", "p3", "p4")]
        outs = [complete(p) for p in prompts]
        c.expect(all(o for o in outs), "all requests completed with output")
        time.sleep(10)  # would-be drain window
        n = cache_object_count()
        c.expect(n == 0, f"no pressure => zero stores (objects={n})")
        # Flush the ledger after the last counter-changing request: the
        # sub-chunk prompt admits no op of its own, so it reprints the
        # settled counters without perturbing them.
        complete(long_prompt("flush", 2), 4)
        time.sleep(1)
        # One session per request whose ops are still buffered. The flush
        # request stores nothing, so its session closes on the spot.
        sessions = active_sessions()
        c.expect(
            sessions == len(prompts),
            f"every request with buffered ops still holds its session "
            f"(active_sessions={sessions}, requests={len(prompts)})",
        )
    finally:
        teardown([vllm, server])
    expect_clean_exit(c, "vllm", vllm)
    expect_clean_exit(c, "mp-server", server)
    ledger = check_ledger(c, "S2", max_evicted=0)
    if ledger:
        c.expect(
            ledger["emitted"] == 0,
            f"no pressure => nothing emitted (emitted={ledger['emitted']})",
        )
        # One op per single-step prefill, all four still queued: this is
        # what makes `emitted == 0` a statement about the policy holding
        # ops rather than about no op ever existing.
        c.expect(
            ledger["admitted"] == len(prompts) == ledger["pending"],
            f"all four requests' ops are admitted and still queued "
            f"(admitted={ledger['admitted']}, pending={ledger['pending']})",
        )
    return c


def scenario_S4() -> Check:
    """FIFO legacy: count-triggered stores happen without pressure.

    The negative control for the eviction-aware policy's central claim. This
    is the same no-pressure configuration as S2, where the eviction-aware
    policy stores nothing; under FIFO with threshold 1 the queue drains on
    count alone, so the objects must appear anyway. Both scenarios failing
    the same way would mean the pool was under pressure after all.
    """
    c = Check()
    server = start_server("S4")
    vllm = start_vllm_under(server, "S4",
        {
            "lmcache.mp.lazy_offload": True,
            "lmcache.mp.lazy_offload_policy": "FIFO",
            "lmcache.mp.lazy_offload_threshold": 1,
            "lmcache.mp.lazy_offload_select_count": 10,
        },
        ["--gpu-memory-utilization", "0.5"],
    )
    try:
        prompts = [long_prompt(s, 60) for s in ("f1", "f2")]
        # FIFO drains a request's whole op list on count, and nothing here
        # can drop one, so every full chunk of both prompts must be stored.
        chunk = chunk_size()
        expected = sum(stored_tokens(p) // chunk for p in prompts)
        for p in prompts:
            complete(p)
        time.sleep(10)
        # FIFO is count-triggered but still drains inside `collect_due`,
        # which only runs on a step that schedules tokens -- so the last
        # request's ops sit in the queue until something else runs, exactly
        # like the eviction-aware policy's held ops. Measured: without this
        # request the count stops at one prompt's five chunks. A sub-chunk
        # prompt is enough to drive the step and admits no op of its own.
        complete(long_prompt("flush", 2), 4)
        time.sleep(5)
        n = cache_object_count()
        c.expect(
            n == expected,
            f"FIFO threshold=1 stored every chunk without pressure "
            f"(objects={n}, expected={expected})",
        )
        # The premise the object count cannot carry. Eager storage produces
        # this exact count too, so without the init line the assertion above
        # passes whether the FIFO settings were honoured or dropped on the
        # floor. (The eviction-aware default is excluded either way -- it is
        # S2, which stores nothing here -- but "not EVICTION_AWARE" is not
        # "FIFO".)
        modes = mode_lines("S4")
        c.expect(
            len(modes) == 1 and "FIFO policy, offload threshold: 1" in modes[0],
            f"the engine came up in FIFO mode with threshold 1 (lines={modes})",
        )
    finally:
        teardown([vllm, server])
    expect_clean_exit(c, "vllm", vllm)
    expect_clean_exit(c, "mp-server", server)
    return c


def scenario_S3() -> Check:
    """Lazy on, tiny pool: pressure drives stores; evicted prefix retrieved intact."""
    c = Check()
    server = start_server("S3")
    vllm = start_vllm_under(server, "S3",
        {"lmcache.mp.lazy_offload": True},
        [
            "--gpu-memory-utilization", "0.5",
            "--max-model-len", "4096",  # later flag wins over the default 8192
            "--num-gpu-blocks-override", "448",  # 448*16 = 7168 tokens of KV
        ],
    )
    try:
        first = long_prompt("target", 100)  # 1965 tokens
        out_first = complete(first)
        # Fill the pool with distinct long prompts to evict "target".
        for s in ("w1", "w2", "w3", "w4", "w5"):
            complete(long_prompt(s, 100))
        time.sleep(5)
        n = cache_object_count()
        c.expect(n > 0, f"pressure drives lazy stores (objects={n})")
        # Negative control: unique prompts never retrieve, so a Retrieved
        # line before the replay would mean the probe measures noise.
        pre = grep_retrieved("S3")
        c.expect(pre == [], f"no retrieval before the replay (pre={pre})")
        replay = complete(first)
        c.expect(
            replay == out_first,
            "replayed evicted prefix reproduces original greedy output",
        )
        # The replay must be served by LMCache, not by a surviving GPU
        # prefix or silent recomputation: the server must log retrieves
        # summing to the *whole* stored prefix. (This also proves the target
        # was actually evicted from the GPU prefix cache -- a full APC hit
        # would skip retrieval entirely.) Exact, not a floor: a floor of
        # 1024 also passes when the tail of the prefix was lost, which is
        # the failure this scenario exists to catch.
        expected = stored_tokens(first)
        post = grep_retrieved("S3")
        c.expect(
            sum(post) == expected,
            f"replay retrieved the whole evicted prefix from LMCache "
            f"(retrieved {sum(post)}, expected {expected}, retrieves={post})",
        )
        status = server_status()
        (LOGDIR / "S3_status.json").write_text(
            json.dumps(status, indent=2, default=str)
        )
        # Ledger flush after the last counter-changing request.
        time.sleep(6)
        complete(long_prompt("flush", 2), 4)
        time.sleep(1)
    finally:
        teardown([vllm, server])
    expect_clean_exit(c, "vllm", vllm)
    expect_clean_exit(c, "mp-server", server)
    # Six serial single-step prefills against a 448-block pool: the
    # feedforward sees each one coming, and the measured value is 0.
    ledger = check_ledger(c, "S3", max_evicted=1)
    if ledger:
        # One op per single-step prefill for the target and the five
        # fillers. The replay admits none: its prefix comes back from
        # LMCache, so no new chunk is produced for it to store.
        c.expect(
            ledger["admitted"] == 6,
            f"one op admitted per prefill, none for the replay "
            f"(admitted={ledger['admitted']})",
        )
    return c


#: S5 (timing anatomy: phase A half-fills the pool and expects zero stores,
#: phase B exhausts it and expects drains, phase C idles and expects the
#: object count frozen) is retired, not merely unlisted. It was run once,
#: under temporary instrumentation that has since been rolled back, and
#: every claim it made now has a stricter home:
#:
#: - "no pressure, no stores" is S2, which additionally pins the ops as
#:   still queued and the sessions as still open;
#: - "pressure drains them, intact" is S3, which retrieves the exact stored
#:   prefix rather than counting objects;
#: - "idle freezes the count" is S13 phase A, which also proves the held ops
#:   were not silently lost by retrieving all of them afterwards.
#:
#: Its unique content was the per-request timestamp trace, which is a
#: debugging aid rather than an oracle. Keeping it registered but never
#: swept made the harness look like it covered ten scenarios.


PROBE_SCRIPT = b"""
import hashlib
sm = app.state.engine.context.storage_manager
objs = sm._l1_manager._objects
lines = []
for k, st in list(objs.items()):
    h = hashlib.md5(st.memory_obj.byte_array).hexdigest()
    lines.append(str(k) + "\\t" + h)
result = "\\n".join(lines)
"""


#: md5 of the empty byte string. Degenerate hashes are the failure mode a
#: hash comparison cannot see by itself: if the probe reads empty buffers,
#: every object matches every other object and the comparison passes.
_EMPTY_MD5 = "d41d8cd98f00b204e9800998ecf8427e"


def l1_md5s() -> dict[str, str]:
    """Fetch {object_key: md5(bytes)} for every L1 object via /run_script."""
    script = BASE / "probe_md5.py"
    script.write_bytes(PROBE_SCRIPT)
    out = subprocess.run(
        ["curl", "-s", "-F", f"script=@{script}",
         f"http://127.0.0.1:{HTTP_PORT}/run_script"],
        capture_output=True, text=True, check=True,
    ).stdout
    if out.startswith("Script execution failed"):
        raise RuntimeError(out)
    hashes = {}
    for line in out.splitlines():
        if "\t" in line:
            key, md5 = line.rsplit("\t", 1)
            hashes[key] = md5
    return hashes


def cache_clear() -> None:
    subprocess.run(
        ["curl", "-s", "-X", "POST", f"http://127.0.0.1:{HTTP_PORT}/cache/clear"],
        capture_output=True, check=True,
    )


def compare_stored_bytes(
    c: "Check", label: str, lazy: dict[str, str], eager: dict[str, str]
) -> None:
    """Assert the lazy run's stored objects are byte-identical to eager's.

    Three assertions, in the order a corruption would surface:

    - no lazy-only key. Keys are content-addressed, so a key the eager run
      never produced means the lazy path stored bytes for a token range it
      invented; that shows up here, not in the digest comparison.
    - every lazy key is under comparison. Without this, a run that stored
      one chunk and dropped the rest still reports byte identity "on all
      common chunks".
    - identical digests on all of them.

    The lazy side is allowed to be a subset: it drops ops under pressure by
    design, and a missing key cannot fake a match.

    Args:
        c: The scenario's check recorder.
        label: Scenario name, for the printed mismatches.
        lazy: {object key: md5} after the lazy run.
        eager: {object key: md5} after the eager run.
    """
    common = [k for k in lazy if k in eager]
    missing = [k for k in lazy if k not in eager]
    c.expect(
        missing == [],
        f"[{label}] every lazy key exists in the eager run "
        f"(lazy-only={missing[:3]})",
    )
    c.expect(
        len(common) == len(lazy),
        f"[{label}] every lazy chunk is under comparison "
        f"(common={len(common)}, lazy={len(lazy)})",
    )
    mismatched = [k for k in common if lazy[k] != eager[k]]
    c.expect(
        not mismatched,
        f"[{label}] stored KV bytes identical on all {len(common)} common "
        f"chunks (mismatched={len(mismatched)})",
    )


def scenario_S6() -> Check:
    """Content verification: lazy-stored KV bytes == eager-stored KV bytes.

    Same prompts, same engine config apart from the lazy flag, one shared
    server. Chunk keys are content-addressed, so matching keys across the
    two runs address the same tokens; their stored bytes must be identical.
    """
    c = Check()
    server = start_server("S6")
    vllm_args = [
        "--gpu-memory-utilization", "0.5",
        "--max-model-len", "4096",
        "--num-gpu-blocks-override", "448",
        "--max-num-batched-tokens", "512",
    ]
    prompts = [long_prompt(s, 100) for s in ("t", "f1", "f2", "f3")]
    try:
        vllm = start_vllm_under(server, "S6",
            {"lmcache.mp.lazy_offload": True}, vllm_args)
        try:
            # Measured while an engine is up: both phases use the same
            # prompts, so one reading serves the eager phase too. Summed per
            # prompt rather than multiplied: the seeds differ in length, so
            # `long_prompt(seed, 100)` is not the same token count for all
            # four of them.
            chunk = chunk_size()
            total_chunks = sum(stored_tokens(p) // chunk for p in prompts)
            lazy_outs = [complete(p) for p in prompts]
            time.sleep(6)
            # Flush the periodic ledger: a tiny (<1 chunk, so store-free)
            # request forces engine steps past the 5s throttle.
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
            lazy = l1_md5s()
        finally:
            teardown([vllm])
        expect_clean_exit(c, "lazy vllm", vllm)
        # Enough chunks to be a content test rather than an anecdote: every
        # prompt's full chunks, minus the couple preemption drops at the
        # margin under this config.
        floor = total_chunks - 8
        c.expect(
            len(lazy) >= floor,
            f"the lazy run stored most of the four prefixes' chunks "
            f"(n={len(lazy)}, floor={floor})",
        )
        # Non-degeneracy control for the comparison below: an empty or
        # constant buffer would make every hash match trivially. All four
        # prompts differ, so no two chunks can legitimately share bytes.
        c.expect(
            _EMPTY_MD5 not in lazy.values(),
            f"no stored object is an empty buffer "
            f"(empty={sum(1 for h in lazy.values() if h == _EMPTY_MD5)})",
        )
        c.expect(
            len(set(lazy.values())) == len(lazy),
            f"every stored chunk has distinct bytes "
            f"(unique={len(set(lazy.values()))}, objects={len(lazy)})",
        )
        # This config preempts (its own ledger shows
        # dropped_on_request_drop=2), and a resumed request's re-admitted
        # ops can lose one to eviction at the margin. A lost op costs S6
        # nothing -- its subject is the bytes of the chunks that did store,
        # and a missing lazy key cannot fake a match.
        check_ledger(c, "S6", max_evicted=2)

        cache_clear()
        c.expect(cache_object_count() == 0, "cache cleared between runs")

        vllm = start_vllm_under(server, "S6b", {}, vllm_args)
        try:
            eager_outs = [complete(p) for p in prompts]
            time.sleep(6)
            eager = l1_md5s()
        finally:
            teardown([vllm])
        expect_clean_exit(c, "eager vllm", vllm)
        # The eager run is the reference and nothing can drop from it, so
        # its chunk count is exact: four prefixes, every full chunk.
        expected_eager = total_chunks
        c.expect(
            len(eager) == expected_eager,
            f"the eager run stored every chunk (n={len(eager)}, "
            f"expected={expected_eager})",
        )
        c.expect(
            eager_outs == lazy_outs,
            "greedy outputs identical across lazy/eager runs",
        )

        compare_stored_bytes(c, "S6", lazy, eager)
        mismatched = [k for k in lazy if k in eager and lazy[k] != eager[k]]
        if mismatched:
            for k in mismatched[:5]:
                print(f"[S6] MISMATCH {k}: lazy={lazy[k]} eager={eager[k]}")
    finally:
        teardown([server])
    expect_clean_exit(c, "mp-server", server)
    return c


def scenario_S9() -> Check:
    """Concurrent multi-request pressure: correctness plus the drop ledger.

    A serial baseline request gets evicted by a 12-request concurrent burst
    through a small pool under the default (large) step token budget --
    per-step allocation far above the EMA between drains, the cross-step
    pin-cascade regime the in-call shift accounting does not cover. The
    burst must complete, the ledger must balance against the aggregate
    drop lines, and whatever survived the burst must be retrievable by a
    fresh engine (phase 2, empty APC, all 13 prompts replayed -- immune
    to WHICH ops the burst dropped). dropped_evicted is reported for
    calibration, not asserted zero: a burst the feedforward never saw
    coming may legitimately lose ops.
    """
    c = Check()
    server = start_server("S9")
    vllm_args = [
        "--gpu-memory-utilization", "0.5",
        "--max-model-len", "4096",
        "--num-gpu-blocks-override", "448",
    ]
    base_prompt = long_prompt("c0", 100)  # 1965 tokens
    burst = [long_prompt(f"c{i}", 100) for i in range(1, 13)]
    try:
        vllm = start_vllm_under(server, "S9",
            {"lmcache.mp.lazy_offload": True}, vllm_args)
        try:
            out0 = complete(base_prompt)
            with ThreadPoolExecutor(max_workers=8) as pool:
                outs = list(pool.map(complete, burst))
            c.expect(
                all(outs),
                f"all {len(burst)} concurrent requests completed with output",
            )

            # Let whatever the burst emitted land before the replay below.
            # The reading that used to be taken here moved next to the
            # ledger -- see the comment at that assertion.
            time.sleep(6)

            # In-engine replay checks byte identity only. Whether the
            # baseline's own op survived the burst is policy weather (~70%
            # of ops legally drop here); the retrieval evidence moves to
            # the fresh-engine phase below, where it is deterministic.
            replay = complete(base_prompt)
            c.expect(
                replay == out0,
                "baseline evicted by the burst replays byte-identically",
            )
            # Flush the ledger AFTER the last counter-changing request:
            # let the 5s throttle lapse, then drive one step so collect_due
            # reprints. The flush prompt is under one chunk -- it admits no
            # op and retrieves nothing, so it cannot perturb what it
            # flushes.
            time.sleep(6)
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
        finally:
            teardown([vllm])
        expect_clean_exit(c, "vllm", vllm)
        # 13 single-step prefills, so 13 ops; the burst legitimately loses
        # most of them (10 measured), but not all -- an upper bound is the
        # only sensor gate 1 has here, and without one a regression that
        # drops every op reads as ALL PASS.
        ledger = check_ledger(c, "S9", max_evicted=11)
        if ledger:
            # A range, not `== 13`. The step token budget covers a whole
            # 1965-token prompt, so each prefill *usually* produces one op
            # -- but this is the one concurrent scenario, and a step's
            # budget is shared: a prompt scheduled alongside others can be
            # split across two steps and admit two ops. Round 13 measured
            # 14 where twelve earlier rounds measured 13, with the ledger
            # equation still closing. The headroom is for that split and
            # nothing more: a policy admitting one op per *chunk* would
            # read around 91 here, and the lower bound still catches a
            # request whose ops never reached the queue.
            prefills = len(burst) + 1
            c.expect(
                prefills <= ledger["admitted"] <= prefills + 3,
                f"every prefill admitted at least one op, and no prefill "
                f"was split more than a step or two "
                f"(admitted={ledger['admitted']}, prefills={prefills})",
            )
            c.expect(
                ledger["emitted"] > 0,
                f"the burst emitted ops (emitted={ledger['emitted']})",
            )
            # Read here, not right after the burst. The two readings used
            # to come from different instants -- the object count a few
            # seconds after the burst, `emitted` from the settled ledger
            # after the flush -- and a drain in between made "0 objects"
            # contradict "2 emitted" in the same run. A burst can also
            # legitimately lose every op to eviction before any drain, so
            # the invariant is that the two readings agree, not that the
            # count has a floor.
            objects = cache_object_count()
            c.expect(
                (objects > 0) == (ledger["emitted"] > 0),
                f"L1 holds objects exactly when the drain emitted ops "
                f"(objects={objects}, emitted={ledger['emitted']})",
            )
            if ledger["pending"] == 0:
                # The queue is empty, so every deferred teardown must have
                # fired. This is the one reading that catches a session
                # leaking past its request's last op.
                sessions = active_sessions()
                c.expect(
                    sessions == 0,
                    f"a drained queue leaves no session open "
                    f"(active_sessions={sessions}, pending=0)",
                )
            else:
                # Whether the burst's last op drains depends on what pressed
                # on the pool after it, so this branch is reached on some
                # runs and not others. It used to print and skip, which made
                # the assertion count fall one short and failed the scenario
                # on a legitimate outcome. The drained-queue reading is
                # unavailable here, but the contract still says something:
                # a session stays open only for a request that still has
                # buffered ops, so there is at least one and no more than
                # the number of ops.
                sessions = active_sessions()
                pending = ledger["pending"]
                c.expect(
                    1 <= sessions <= pending,
                    f"the sessions still open belong to requests with "
                    f"buffered ops (active_sessions={sessions}, "
                    f"pending={pending})",
                )

        # --- Phase 2: retrieval evidence on a fresh engine (empty APC).
        # Phase 1 asserted emitted > 0; every emitted op's prefix belongs
        # to exactly one of the 13 distinct prompts, so replaying them all
        # against an empty APC must retrieve at least one stored prefix,
        # regardless of which ops the burst dropped.
        vllm = start_vllm_under(server, "S9b", {}, vllm_args)
        try:
            n0 = len(grep_retrieved("S9"))
            replays = [complete(p) for p in [base_prompt] + burst]
            # Only the baseline is compared byte-for-byte: it was generated
            # serially, like the replay. The burst outputs came from
            # concurrent batches, and greedy decoding is not batch-invariant.
            c.expect(
                replays[0] == out0 and all(replays),
                "fresh-engine replays complete, baseline byte-identical",
            )
            retr = grep_retrieved("S9")[n0:]
            # Which ops survived the burst is policy weather, so the total
            # cannot be exact here. What can be pinned is the shape: every
            # retrieval is a whole number of chunks and none exceeds one
            # prefix, so a retrieval spanning two prompts' chunks, or a
            # partial chunk, fails even though the floor would pass.
            chunk = chunk_size()
            # The longest of the 13 prefixes: the seeds differ in length, so
            # so do the prompts, and the bound has to hold for whichever
            # ones survived the burst.
            full = max(stored_tokens(p) for p in [base_prompt] + burst)
            c.expect(
                retr != []
                and all(v % chunk == 0 and 0 < v <= full for v in retr),
                f"the burst's surviving stores come back from a fresh "
                f"engine in whole chunks, none past one prefix "
                f"(chunk={chunk}, prefix={full}, retrieves={retr})",
            )
            c.expect(
                max(retr, default=0) >= 1024,
                f"at least one retrieval is a substantial prefix, not a "
                f"single chunk (retrieves={retr})",
            )
        finally:
            teardown([vllm])
        expect_clean_exit(c, "phase-2 vllm", vllm)
    finally:
        teardown([server])
    expect_clean_exit(c, "mp-server", server)
    return c


def scenario_S11() -> Check:
    """Shared-prefix dedup (phase A) and gate-3 min_prefix_tokens (phase B).

    Phase A: two requests share a 1965-token prefix; the second one's
    overlapping chunks must deduplicate against the first's buffered ops
    (ledger deduplicated > 0), and after eviction the second request must
    replay byte-identically out of LMCache -- its prefix only ever stored
    under the first request's ops.

    Phase B: with min_prefix_tokens=2048, 965-token requests coming due
    are rejected (rejected_short_prefix > 0, nothing of theirs stored, so
    their replay retrieves nothing), while a 3072-token request clears
    the gate, stores, and replays from LMCache.
    """
    c = Check()
    server = start_server("S11")
    vllm_args = [
        "--gpu-memory-utilization", "0.5",
        "--max-model-len", "4096",
        "--num-gpu-blocks-override", "448",
    ]
    try:
        # --- Phase A: dedup ---
        vllm = start_vllm_under(server, "S11",
            {"lmcache.mp.lazy_offload": True}, vllm_args)
        try:
            shared = long_prompt("shared", 100)  # 1965 tokens
            r1 = shared + " Continue with topic one."
            r2 = shared + " Instead, summarize everything."
            complete(r1)
            out2 = complete(r2)
            for s in ("w1", "w2", "w3", "w4"):  # evict the shared prefix
                complete(long_prompt(s, 100))
            time.sleep(6)
            n = cache_object_count()
            c.expect(n > 0, f"pressure drives lazy stores (objects={n})")
            n0 = len(grep_retrieved("S11"))
            replay2 = complete(r2)
            c.expect(
                replay2 == out2,
                "deduplicated request replays byte-identically after eviction",
            )
            retr = grep_retrieved("S11")[n0:]
            # Exact: the deduplicated request's prefix was only ever stored
            # under the *first* request's ops, so getting all of it back is
            # what says the dedup pointed at live data. A floor would pass
            # on a prefix that came back one chunk short -- which is what a
            # dedup against a partially dropped covering op looks like.
            expected = stored_tokens(shared)
            c.expect(
                sum(retr) == expected,
                f"replay of the deduplicated request retrieved its whole "
                f"shared prefix from LMCache (retrieved {sum(retr)}, "
                f"expected {expected}, retrieves={retr})",
            )
            # Ledger flush after the last counter-changing request (the
            # replay admits ops too); sub-chunk, so it perturbs nothing.
            time.sleep(6)
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
        finally:
            teardown([vllm])
        expect_clean_exit(c, "phase-A vllm", vllm)
        ledger = check_ledger(c, "S11", max_evicted=1)
        if ledger:
            # Exactly one: the second request's single op deduplicates
            # against the first's. A floor also passes when the policy
            # deduplicates too much -- the dangerous direction, since an op
            # wrongly folded into another request's coverage is a prefix
            # that never gets stored.
            c.expect(
                ledger["deduplicated"] == 1,
                f"the shared prefix deduplicated exactly once "
                f"(deduplicated={ledger['deduplicated']})",
            )
            # One op per single-step prefill for r1 and the four fillers;
            # r2's op deduplicates instead of being admitted, and the
            # replay's prefix comes back from LMCache so it produces none.
            c.expect(
                ledger["admitted"] == 5,
                f"only the non-deduplicated prefills admitted ops "
                f"(admitted={ledger['admitted']})",
            )

        # --- Phase B: gate-3 ---
        cache_clear()
        c.expect(cache_object_count() == 0, "cache cleared between phases")
        vllm = start_vllm_under(server, "S11b",
            {
                "lmcache.mp.lazy_offload": True,
                "lmcache.mp.lazy_offload_min_prefix_tokens": 2048,
            },
            vllm_args)
        try:
            shorts = [long_prompt(f"s{i}", 50) for i in range(4)]  # 965 tokens
            long_p = long_prompt("L", 150)  # 3072 tokens
            # Fixed ids for the below-gate requests, so the gate-3 drop
            # lines can be checked to name these and nothing else. vLLM
            # appends 8 random characters, so the id reaches the connector
            # as `cmpl-{rid}-0-XXXXXXXX` -- a stable substring either way.
            short_ids = [f"sh{i}" for i in range(len(shorts))]
            replay_id = "shr"
            short_outs = [
                complete(p, request_id=rid) for p, rid in zip(shorts, short_ids)
            ]
            out_long = complete(long_p)
            for s in ("x1", "x2", "x3", "x4"):  # evict everything
                complete(long_prompt(s, 100))
            time.sleep(6)
            n0 = len(grep_retrieved("S11"))  # one shared server log
            replay_long = complete(long_p)
            c.expect(
                replay_long == out_long,
                "long request above the gate replays byte-identically",
            )
            long_retr = grep_retrieved("S11")[n0:]
            # Exact: gate 3 is a per-request verdict, so an above-gate
            # request must get its *whole* prefix stored. A floor would also
            # pass if the gate withheld all but the first few chunks.
            expected_long = stored_tokens(long_p)
            c.expect(
                sum(long_retr) == expected_long,
                f"the long prefix cleared gate 3 whole and came back from "
                f"LMCache (retrieved {sum(long_retr)}, expected "
                f"{expected_long}, retrieves={long_retr})",
            )
            n1 = len(grep_retrieved("S11"))
            q0 = vllm_metric(_METRIC_EXT_QUERIES)
            h0 = vllm_metric(_METRIC_EXT_HITS)
            replay_short = complete(shorts[0], request_id=replay_id)
            c.expect(
                replay_short == short_outs[0],
                "short request below the gate recomputes byte-identically",
            )
            short_retr = grep_retrieved("S11")[n1:]
            c.expect(
                short_retr == [],
                f"gate 3 withheld short-prefix stores "
                f"(unexpected retrieves={short_retr})",
            )
            # Positive witness for that negative: the server only logs a
            # retrieve on a hit, so "no Retrieved line" alone cannot tell a
            # withheld store from a lookup that never happened. vLLM's own
            # accounting can: the lookup ran (queries rose by the tokens it
            # had to ask about) and found nothing (hits flat).
            queried = vllm_metric(_METRIC_EXT_QUERIES) - q0
            hit = vllm_metric(_METRIC_EXT_HITS) - h0
            c.expect(
                queried > 0 and hit == 0,
                f"the short replay did query LMCache and missed "
                f"(queried={queried:.0f} tokens, hit={hit:.0f})",
            )
            # Ledger flush after the last counter-changing request (the
            # short replay is gate-rejected at drain, which counts too).
            time.sleep(6)
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
        finally:
            teardown([vllm])
        expect_clean_exit(c, "phase-B vllm", vllm)
        ledgerb = check_ledger(c, "S11b", max_evicted=2)
        if ledgerb:
            # Five below-gate ops exist here (the four 965-token requests
            # plus the replay of one of them), and only they may be
            # rejected. The upper bound is the half that bites: rejecting
            # an above-gate op is silent cache-quality loss.
            c.expect(
                1 <= ledgerb["rejected_short_prefix"] <= len(shorts) + 1,
                f"gate 3 rejected only below-gate ops "
                f"(rejected_short_prefix={ledgerb['rejected_short_prefix']}, "
                f"below-gate ops={len(shorts) + 1})",
            )
            # One op per single-step prefill: 4 shorts + the long request +
            # 4 fills + the short replay. The long replay produces none (it
            # comes back from LMCache).
            c.expect(
                ledgerb["admitted"] == 10,
                f"one op admitted per prefill (admitted={ledgerb['admitted']})",
            )
            # The x1-x4 fills are 100 sentences of a *2-character* seed --
            # 2165 tokens, which chunk-floors to 2048 and so clears the
            # 2048 gate (`known_prefix < min_prefix_tokens` is false at
            # equality). They plus the long request are the five above-gate
            # ops, and `emitted` is dominated by them, which is why it is
            # not evidence about the long request: that comes from its exact
            # retrieval above. (An earlier note here computed 1965 tokens
            # for the fills from the 6-character-seed measurement and
            # concluded they were gate-rejected. The server's `Stored 2048
            # tokens` lines say otherwise.)
            c.expect(
                ledgerb["emitted"] >= 1,
                f"above-gate ops still emitted (emitted={ledgerb['emitted']})",
            )
            # Gate-3 drops must be attributable to a request, not just
            # counted: the aggregate INFO line names the request and the
            # prefix length that failed the gate.
            gate_lines = grep_lines("S11b", re.escape(_CAUSE_SHORT_PREFIX))
            c.expect(
                gate_lines != [],
                f"gate-3 drops are logged, not counted silently "
                f"({len(gate_lines)} line(s))",
            )
            # Check the *identity* of what was rejected, not its prefix
            # length. "Every named prefix is below the gate" restates gate
            # 3's own selection rule -- it drops a request's whole op list
            # on `ops[-1].prefix_end_tokens < min_prefix_tokens`, so every
            # op it can name necessarily has a prefix below the threshold,
            # and no reachable bug makes that assertion fail. Naming the
            # expected request ids instead is falsifiable: an above-gate
            # request appearing here fails, and so does a rejection count
            # the lines cannot account for.
            assert_no_truncated_drop_lines(c, "S11b")
            named = [
                m.group(1)
                for ln in gate_lines
                for m in re.finditer(r"(cmpl-[\w.-]+) \(prefix \d+\)", ln)
            ]
            expected_ids = tuple(f"cmpl-{rid}-0-" for rid in short_ids + [replay_id])
            unexpected = [
                rid for rid in named if not rid.startswith(expected_ids)
            ]
            c.expect(
                named != [] and unexpected == [],
                f"the gate-3 drop lines name only the below-gate requests "
                f"(named={len(named)}, unexpected={unexpected})",
            )
            c.expect(
                len(named) == ledgerb["rejected_short_prefix"],
                f"every rejected op is accounted for by name "
                f"(named={len(named)}, "
                f"rejected_short_prefix={ledgerb['rejected_short_prefix']})",
            )
    finally:
        teardown([server])
    expect_clean_exit(c, "mp-server", server)
    return c


#: vllm args shared by the small-pool scenarios: 448*16 = 7168 tokens of KV
#: and a 512-token step budget, so steady-state allocation stays inside the
#: drain horizon. It does NOT buy dropped_evicted == 0: pinned pending
#: blocks shrink the usable pool enough that even serial requests get
#: preempted, and a resumed request's re-admitted ops can lose one at the
#: margin (0 and 1 measured across two runs of S12). Scenarios using this
#: therefore pass a small nonzero `max_evicted` to `check_ledger` and assert
#: drop-freedom per request id.
_SMALL_POOL = [
    "--gpu-memory-utilization", "0.5",
    "--max-model-len", "4096",
    "--num-gpu-blocks-override", "448",
    "--max-num-batched-tokens", "512",
]

def assert_no_truncated_drop_lines(c: "Check", scenario: str) -> None:
    """Require every dropped op to be named in the aggregate drop lines.

    The eviction drop line names at most `_DROP_LOG_SAMPLE_OPS` ops and
    collapses the rest into "+N more", so a per-request-id grep for drops
    is only sound while no line truncates. Scenarios that assert drop
    freedom for one id must call this, or a large drain could hide that id.
    """
    truncated = grep_lines(scenario, r"\+\d+ more")
    c.expect(
        not truncated,
        f"no drop line truncated its op list, so per-id drop greps see "
        f"every drop (lines={truncated})",
    )


def scenario_S12() -> Check:
    """Request id reuse: a deferred predecessor is reclaimed, not conflated.

    In lazy mode a finished request keeps its buffered ops while vLLM frees
    its id at once (request_finished returns False), so a new request under
    that id arrives while the predecessor's teardown is still deferred.
    Both requests here are 1965 tokens (123 blocks) against a 448-block
    pool, so nothing presses on it before the reuse and the predecessor
    still holds buffered ops when its id comes back. That premise is
    arithmetic, not a guarantee -- longer prompts or a smaller pool would
    drain the predecessor first -- so the scenario asserts it (objects == 0
    at reuse time) instead of assuming it.

    Reaching this from an HTTP client needs
    VLLM_DISABLE_REQUEST_ID_RANDOMIZATION: vLLM otherwise appends 8 random
    characters to every externally supplied id
    (v1/engine/input_processor.py::assign_request_id), so the connector
    never sees a duplicate. That env var -- deprecated but still supported
    -- plus embedded callers that drive the engine core with their own ids
    are exactly the configurations where the reclaim path matters, so it is
    the setting under test here, not a harness trick.

    Pinned: the reclaim really runs on a live engine (its INFO line plus
    dropped_id_reuse, two readings of one event), and the predecessor's
    discarded ops do not take the successor's with them -- every chunk of
    the successor's prefix stores and comes back out of LMCache, and its
    output replays byte-identically. A prefix-close over the successor's
    ops, or a failure to clear the predecessor's `_prefix_broken` mark,
    fails one of those.

    Not pinned, deliberately: the *session* half of the reclaim contract
    (that the predecessor's `end_session` must fire here rather than ride
    the finished marker). Observing it needs the successor's queue entry to
    empty while the successor is still running, and a request's ops only
    leave the queue once its blocks are freed -- i.e. after it finished. In
    a single-request harness the buggy and correct variants differ only in
    which call site emits `end_session`, so the difference is unobservable;
    `TestIdReuseReclaim` in tests/v1/test_lazy_offload_policy.py covers it.
    The in-flight branch of the reclaim is out of reach for the same class
    of reason: the receipt is processed on the next step that schedules
    tokens, and the drain that starts the batch is itself driven by a
    stepping engine, so the window closes a step later.
    """
    c = Check()
    server = start_server("S12")
    rid = "lz-reuse"
    try:
        vllm = start_vllm_under(server, "S12",
            {"lmcache.mp.lazy_offload": True}, _SMALL_POOL,
            {"VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1"})
        try:
            pred = long_prompt("pred", 100)  # 1965 tokens
            succ = long_prompt("succ", 100)
            out_pred = complete(pred, request_id=rid)
            # Reuse the id before anything drains: no pressure yet, so the
            # predecessor's ops are still buffered and its session open.
            out_succ = complete(succ, request_id=rid)
            c.expect(
                bool(out_pred) and bool(out_succ),
                "both requests under the reused id completed with output",
            )
            n = cache_object_count()
            c.expect(n == 0, f"no pressure yet => nothing stored (objects={n})")
            reuse = grep_lines("S12", r"reused while its predecessor")
            c.expect(
                len(reuse) == 1,
                f"connector reclaimed the reused id exactly once "
                f"(lines={reuse})",
            )

            # Drive pressure so the successor's ops drain, and keep going
            # until its prefix is evicted from the GPU cache -- otherwise
            # the replay below would be an APC hit and prove nothing.
            for s in ("w1", "w2", "w3", "w4", "w5"):
                complete(long_prompt(s, 100))
            time.sleep(6)
            n = cache_object_count()
            c.expect(n > 0, f"pressure drives lazy stores (objects={n})")
            n0 = len(grep_retrieved("S12"))
            # Replayed under a fresh id: reusing rid again would reclaim a
            # second time and muddy the counter.
            replay = complete(succ)
            c.expect(
                replay == out_succ,
                "the successor of the reused id replays byte-identically",
            )
            retr = grep_retrieved("S12")[n0:]
            expected = stored_tokens(succ)
            c.expect(
                sum(retr) == expected,
                f"every chunk of the successor's prefix survived the "
                f"predecessor's reclaim and came back from LMCache "
                f"(retrieved {sum(retr)}, expected {expected}, "
                f"lines={retr})",
            )
            # What must not happen is the *successor's* ops being caught by
            # a drop path under the shared id. The predecessor's four ops
            # were discarded on purpose -- that is the reclaim, counted as
            # dropped_id_reuse below -- and the connector's reclaim line
            # does not word it as a drop, so it is not matched here. The
            # scan is deliberately per-id rather than on the ledger total:
            # the pinned pending queue shrinks the usable pool, so filler
            # requests do get preempted and legitimately dropped.
            subject_drops = [
                ln for ln in grep_lines("S12", rid) if "drop" in ln.lower()
            ]
            c.expect(
                not subject_drops,
                f"no eviction or gate drop line names the reused id "
                f"(lines={subject_drops})",
            )
            assert_no_truncated_drop_lines(c, "S12")
            # Ledger flush after the last counter-changing request.
            time.sleep(6)
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
        finally:
            teardown([vllm])
        expect_clean_exit(c, "vllm", vllm)
        ledger = check_ledger(c, "S12", max_evicted=2)
        if ledger:
            # Exact, and derived rather than tabulated: with a 512-token
            # step budget a 1965-token prefill takes four steps, each
            # completing whole chunks, so the predecessor held four ops when
            # its id came back. A floor would also pass if the reclaim
            # discarded one op and orphaned the other three -- which is the
            # leak this scenario exists to catch.
            c.expect(
                ledger["dropped_id_reuse"] == 4,
                f"the reclaim discarded all four of the predecessor's "
                f"buffered ops (dropped_id_reuse={ledger['dropped_id_reuse']})",
            )
            # A floor on the drain as a whole (the fillers alone satisfy
            # it), not on the subject: the subject's storage is proven by
            # its exact retrieval above.
            c.expect(
                ledger["emitted"] >= 1,
                f"the drain emitted ops (emitted={ledger['emitted']})",
            )
            if ledger["pending"] == 0:
                sessions = active_sessions()
                c.expect(
                    sessions == 0,
                    f"a drained queue leaves no session open, the reclaimed "
                    f"predecessor's included (active_sessions={sessions})",
                )
            else:
                # Same reasoning as S9's branch above: an undrained queue
                # is a legitimate outcome, and what still holds is that a
                # session stays open only for a request with buffered ops.
                sessions = active_sessions()
                pending = ledger["pending"]
                c.expect(
                    1 <= sessions <= pending,
                    f"the sessions still open belong to requests with "
                    f"buffered ops (active_sessions={sessions}, "
                    f"pending={pending})",
                )
    finally:
        teardown([server])
    expect_clean_exit(c, "mp-server", server)
    return c


def scenario_S13() -> Check:
    """Idle holds buffered ops; an abort keeps them storable.

    Phase A (idle): one 1965-token request against a 448-block pool nothing
    is pressing on. No scheduler steps run once it finishes, so nothing
    drains -- the object count stays at zero and no new ledger line appears
    across a 30s idle window. Then pressure drains the very same ops and
    *every* chunk of the prefix replays out of LMCache: idle held them, it
    did not lose them. (The exact retrieval is what rules out a silent drop
    during the window; a frozen ledger line count alone would not, since a
    counter change inside the 5s throttle prints nothing.)

    Phase B (abort): the client disconnects mid-generation. An abort is not
    a drop -- it routes through the finished path and the buffered ops stay
    storable -- so after pressure drains them the aborted prompt's prefix
    comes back from LMCache and continues the same greedy output it had
    started to stream. Two independent facts establish that an abort really
    happened: vLLM logs it (--enable-log-requests), and generation stopped
    after a handful of the 512 requested tokens, which an engine that never
    saw the disconnect could not do (ignore_eos rules out an early stop).

    What phase B does *not* touch: the connector has no abort-specific
    hook, so on the LMCache side this is phase A's path plus a regression
    test of vLLM's contract that an aborted request still reaches
    `request_finished`.
    """
    c = Check()
    server = start_server("S13")
    try:
        vllm = start_vllm_under(server, "S13",
            {"lmcache.mp.lazy_offload": True},
            _SMALL_POOL + ["--enable-log-requests"])
        try:
            # --- Phase A: idle ---
            idle_p = long_prompt("idle", 100)  # 1965 tokens
            expected_idle = stored_tokens(idle_p)
            out_idle = complete(idle_p)
            time.sleep(3)
            n_before = cache_object_count()
            c.expect(
                n_before == 0,
                f"half-full pool => nothing stored before idling "
                f"(objects={n_before})",
            )
            # The premise of the whole phase: there is something to hold.
            # Without it, "the count stayed at zero across the idle window"
            # is 0 == 0 on a queue that was empty all along.
            buffered = grep_final_counters("S13")
            c.expect(
                buffered is not None and buffered["pending"] >= 1,
                f"ops are buffered going into the idle window "
                f"(ledger={buffered})",
            )
            ledger_lines = len(grep_lines("S13", r"Lazy offload counters:"))
            # Floor first: without it the comparison below is 0 == 0 the
            # moment periodic ledger logging is renamed or silenced.
            c.expect(
                ledger_lines >= 1,
                f"the request's admissions were logged before idling "
                f"({ledger_lines} ledger lines)",
            )
            print(f"[S13] idling with {ledger_lines} ledger line(s) logged")
            time.sleep(30)
            n_idle = cache_object_count()
            c.expect(
                n_idle == n_before,
                f"idle never drains (objects {n_before} -> {n_idle})",
            )
            after_idle = len(grep_lines("S13", r"Lazy offload counters:"))
            # What this proves is exactly "no new ledger line appeared":
            # a counter change inside the 5s throttle would print nothing,
            # so it is the retrieval check below -- exact, not a floor --
            # that rules out a silent drop during the window.
            c.expect(
                after_idle == ledger_lines,
                f"no ledger line while idle "
                f"({ledger_lines} -> {after_idle} lines)",
            )

            for s in ("i1", "i2", "i3", "i4", "i5"):
                complete(long_prompt(s, 100))
            time.sleep(6)
            n_after = cache_object_count()
            c.expect(
                n_after > 0,
                f"pressure after the idle window drains the held ops "
                f"(objects={n_after})",
            )
            n0 = len(grep_retrieved("S13"))
            replay_idle = complete(idle_p)
            c.expect(
                replay_idle == out_idle,
                "the op held across the idle window replays byte-identically",
            )
            retr_idle = grep_retrieved("S13")[n0:]
            c.expect(
                sum(retr_idle) == expected_idle,
                f"idle held every op rather than losing any: the whole "
                f"prefix came back from LMCache (retrieved {sum(retr_idle)}, "
                f"expected {expected_idle}, lines={retr_idle})",
            )

            # --- Phase B: abort ---
            victim = long_prompt("victim", 150)  # 3072 tokens
            expected_victim = stored_tokens(victim)
            gen0 = vllm_generation_tokens()
            partial = complete_then_disconnect(victim, "lz-abort")
            c.expect(
                len(partial.strip()) >= 10,
                f"the aborted request streamed real text before the "
                f"disconnect (partial={partial!r})",
            )
            # 512 requested tokens with ignore_eos: an engine that never saw
            # the disconnect would still be generating (or just done) after
            # this wait, and the delta would be ~512. The log line alone
            # cannot show that -- AsyncLLM.abort logs the ids it was asked
            # to abort, whether or not any of them still existed.
            time.sleep(15)
            generated = vllm_generation_tokens() - gen0
            c.expect(
                generated < 100,
                f"the abort stopped generation early "
                f"({generated:.0f} tokens of the 512 requested)",
            )
            aborts = grep_lines("S13", r"Aborted request.*lz-abort")
            c.expect(
                len(aborts) >= 1,
                f"vllm aborted the disconnected request (lines={aborts})",
            )
            # An abort must not take the stale-state path. Asserted per id,
            # not on the ledger total: pinned pending blocks shrink the
            # usable pool, so unrelated filler requests do get preempted
            # (and legitimately dropped) under this configuration. Every
            # drop path names its ops, so a line mentioning this id and the
            # word "drop" is the signal; the completeness of that naming is
            # what `assert_no_truncated_drop_lines` below checks.
            abort_drops = [
                ln for ln in grep_lines("S13", "lz-abort") if "drop" in ln.lower()
            ]
            c.expect(
                not abort_drops,
                f"no drop line names the aborted request at abort time "
                f"(lines={abort_drops})",
            )
            for s in ("y1", "y2", "y3", "y4", "y5"):
                complete(long_prompt(s, 100))
            time.sleep(6)
            n1 = len(grep_retrieved("S13"))
            replay_victim = complete(victim)
            c.expect(
                replay_victim.startswith(partial),
                f"the aborted prompt's replay continues its streamed output "
                f"(streamed={partial!r}, replay={replay_victim[:80]!r})",
            )
            retr_victim = grep_retrieved("S13")[n1:]
            c.expect(
                sum(retr_victim) == expected_victim,
                f"an abort is not a drop: every chunk of the aborted "
                f"request's prefix stored and came back from LMCache "
                f"(retrieved {sum(retr_victim)}, expected "
                f"{expected_victim}, lines={retr_victim})",
            )
            # Re-checked after the drain window: the earlier grep only
            # covered abort time, and the ops leave the queue here.
            abort_drops = [
                ln for ln in grep_lines("S13", "lz-abort") if "drop" in ln.lower()
            ]
            c.expect(
                not abort_drops,
                f"nothing of the aborted request was dropped during the "
                f"drain either (lines={abort_drops})",
            )
            assert_no_truncated_drop_lines(c, "S13")
            time.sleep(6)
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
        finally:
            teardown([vllm])
        expect_clean_exit(c, "vllm", vllm)
        ledger = check_ledger(c, "S13", max_evicted=3)
        if ledger:
            c.expect(
                ledger["emitted"] >= 1,
                f"held and aborted ops emitted (emitted={ledger['emitted']})",
            )
    finally:
        teardown([server])
    expect_clean_exit(c, "mp-server", server)
    return c


#: One step's token budget under `_SMALL_POOL` (`--max-num-batched-tokens`).
#: An admitted op covers the full blocks a single step made known, so no op
#: can carry more than this many tokens. That bound is what makes "every
#: store submission held exactly one op" checkable from the server's store
#: sizes alone, which is the only observable a drain's batching has.
_STEP_BUDGET_TOKENS = 512

#: The policy's cap-sizing WARNING, emitted once per process when a drain
#: both hit `max_drain_per_step` and lost ops to eviction in the same step.
_CAP_WARNING = r"held back \d+ due store op\(s\)"


def op_prefix_ends(prompt: str) -> list[int]:
    """The `prefix_end_tokens` of each op a fresh prefill admits, in order.

    Under `_SMALL_POOL` a step makes at most `_STEP_BUDGET_TOKENS` new tokens
    known and admits one op for them, and an op covers whole chunks only --
    so the ends are the multiples of the step budget below the prompt's
    stored length, followed by that length.

    Measured against the ledger rather than assumed: the default-cap phase
    of S14 admits exactly 24 ops for a 1965-token request (ends 512, 1024,
    1536, 1792) plus five 2165-token fillers (ends 512, 1024, 1536, 2048 --
    the 117 tokens past 2048 complete no chunk, so they admit no fifth op).

    Args:
        prompt: The prompt to size. Both servers must be up.

    Returns:
        One prefix end per op, ascending.
    """
    stored = stored_tokens(prompt)
    return list(range(_STEP_BUDGET_TOKENS, stored, _STEP_BUDGET_TOKENS)) + [stored]


def drop_prefixes(scenario: str, request_id: str) -> set[int]:
    """Prefix ends of the ops that drop lines attribute to one request.

    Every drop path names the request and the prefix end of each op it
    dropped, so this reads out *which* of a request's ops were lost -- the
    only way to tell a lost tail from a lost request. Sound only while no
    drop line truncates its op list, which `assert_no_truncated_drop_lines`
    is for.

    Args:
        scenario: The scenario whose engine log to read.
        request_id: The id passed to `complete`, before vllm's random
            suffix.

    Returns:
        The prefix ends dropped for that request, of any cause.
    """
    marker = f"cmpl-{request_id}-0-"
    return {
        int(m.group(2))
        for line in grep_lines(scenario, r"dropped \d+ store op\(s\)")
        for m in re.finditer(r"(cmpl-[\w.-]+) \(prefix (\d+)\)", line)
        if m.group(1).startswith(marker)
    }


def check_prefix_closure(
    c: "Check",
    scenario: str,
    request_id: str,
    ends: list[int],
    retrieved: list[int],
    ranks: int = 1,
) -> None:
    """Assert partial loss left a contiguous stored prefix, and only that.

    The design's promise about a request that loses some of its buffered
    ops: the drop takes a *suffix* of its op list, so the surviving ops
    still form a chain from token 0 and stay storable, and a later lookup
    gets back exactly that run -- never a prefix with a hole in it, which
    LMCache cannot use past the hole and which no counter would reveal.

    Reads which ops were dropped from the drop lines (so it needs the
    request to have been sent with a fixed id) and compares three things:
    every named prefix is one of the request's op ends, the survivors are a
    front run, and the retrieval sums to the last surviving end. Holds
    unchanged when nothing was dropped, where it becomes "the whole prefix
    came back".

    Args:
        c: The scenario's check recorder.
        scenario: The scenario whose engine log to read.
        request_id: The id passed to `complete` for the request under test.
        ends: The request's op prefix ends, ascending (`op_prefix_ends`).
        retrieved: The server's `Retrieved N tokens` counts for a replay of
            that request, and nothing else.
        ranks: How many tensor-parallel workers the engine runs. Each holds
            its own shard of the KV and the server logs one transfer per
            rank, so a TP=2 replay of a 1792-token prefix reports
            `[1792, 1792]`; the expected total scales with this. Left at 1
            it is the single-worker reading.
    """
    # A truncated drop line ("+N more") hides drops from the reader below,
    # which would report a dropped op as a survivor -- the one direction
    # that turns a loss into a pass.
    assert_no_truncated_drop_lines(c, scenario)
    dropped = drop_prefixes(scenario, request_id)
    c.expect(
        dropped <= set(ends),
        f"[{scenario}] every prefix the drop lines attribute to the request "
        f"is one of its op ends (dropped={sorted(dropped)}, ends={ends})",
    )
    survivors = [end for end in ends if end not in dropped]
    c.expect(
        survivors == ends[: len(survivors)],
        f"[{scenario}] prefix closure: the surviving ops are a front run of "
        f"the op list, so nothing stored without its prefix "
        f"(survivors={survivors}, dropped={sorted(dropped)}, ends={ends})",
    )
    surviving_prefix = survivors[-1] if survivors else 0
    c.expect(
        sum(retrieved) == surviving_prefix * ranks,
        f"[{scenario}] the replay retrieved exactly the surviving prefix on "
        f"every rank, no more and no less (retrieved {sum(retrieved)}, "
        f"surviving prefix {surviving_prefix} x {ranks} rank(s), "
        f"lines={retrieved})",
    )


def scenario_S14() -> Check:
    """max_drain_per_step: the throttle splits a batch, and loses its tail.

    Nine scenarios ran on the default cap of 64, which no workload here
    reaches, so this knob had never been set. It is the only control over
    the D2H burst a drain issues, and turning it down moves a request's ops
    from one coalesced store to one store per step -- across which the
    request holds an in-flight batch and `collect_due` skips it entirely,
    waiting for the completion receipt. Nothing had exercised that hand-off
    on hardware.

    Both phases run the identical workload: a 1965-token request (four ops)
    against a 448-block pool nothing is pressing on, so its ops are still
    buffered when five larger fillers arrive to drive the drain; then a
    replay. They differ only in `max_drain_per_step`:

    - phase A (default 64) coalesces the four ops into one store, so a
      submission of exactly the prompt's stored length appears;
    - phase B (1) may not exceed one op per submission, so no store in the
      whole phase carries more than one step's tokens, and the same
      workload takes strictly more submissions.

    Correctness is *prefix closure*, asserted the same way in both phases:
    read which of the request's ops the drop lines name, and require the
    survivors to be a front run of its op list and the replay to retrieve
    exactly that run. This is what the design promises about partial loss --
    the tail goes, the head stays storable, and what comes back is a
    contiguous prefix rather than a prefix with a hole -- and it holds
    whether nothing was dropped (phase A: the whole 1792 tokens come back)
    or the tail was (phase B: 1024 tokens come back, ops at 1536 and 1792
    dropped). Asserting "everything comes back" instead would fail phase B
    for behaving exactly as designed.

    Measured, and the reason phase B loses anything at all: a cap of 1
    drains one op per step while a prefilling request *admits* one op per
    step, so the cap is break-even with a single prefilling request and can
    never work off a backlog. Phase B emitted 11 of 26 admitted ops, dropped
    6 and left 9 buffered at teardown; phase A emitted 21 of 24 and left
    none. A cap below the number of concurrently prefilling requests is a
    steady-state loss setting, not a burst-shaping one.
    """
    c = Check()
    server = start_server("S14")
    victim = long_prompt("throttle", 100)  # 1965 tokens -> 4 ops
    fillers = ("t1", "t2", "t3", "t4", "t5")

    #: Prompt sizing, filled by the first phase. `prompt_tokens` needs a live
    #: vllm, and the cross-phase assertions below run after both engines are
    #: down, so the measurement has to be carried out of the phase.
    sizing: dict[str, int] = {}

    def drive(
        scenario: str, extra_config: dict, max_evicted: int
    ) -> tuple[list[int], dict[str, int]]:
        """Run one phase; return its stores during the drain and its ledger."""
        vllm = start_vllm_under(server, scenario, extra_config,
                                _SMALL_POOL)
        victim_id = f"{scenario.lower()}-victim"
        try:
            ends = op_prefix_ends(victim)
            sizing.update(stored=ends[-1], ops=len(ends))
            sizing[f"{scenario}-filler-ops"] = sum(
                len(op_prefix_ends(long_prompt(seed, 100))) for seed in fillers
            )
            out = complete(victim, request_id=victim_id)
            time.sleep(3)
            n_idle = cache_object_count()
            c.expect(
                n_idle == 0,
                f"[{scenario}] the unpressed pool stored nothing, so all of "
                f"the request's ops are still buffered (objects={n_idle})",
            )
            # A floor, not the op count: the ledger line is throttled to one
            # per 5s and only printed from a drain, and a drain only runs on a
            # step that schedules tokens -- so while the engine idles here the
            # last line is the stale one from the request's first step. The op
            # count is asserted on the phase's settled ledger instead.
            buffered = grep_final_counters(scenario)
            c.expect(
                buffered is not None and buffered["pending"] >= 1,
                f"[{scenario}] ops are buffered going into the drain "
                f"(ledger={buffered})",
            )
            s0 = len(grep_stored("S14"))
            for seed in fillers:
                complete(long_prompt(seed, 100))
            time.sleep(6)
            stores = grep_stored("S14")[s0:]
            print(f"[{scenario}] store submissions during the drain: {stores}")
            n0 = len(grep_retrieved("S14"))
            replay = complete(victim, request_id=f"{scenario.lower()}-replay")
            c.expect(
                replay == out,
                f"[{scenario}] the request replays byte-identically after "
                f"the drain",
            )
            retr = grep_retrieved("S14")[n0:]
            # Read the drops here, not after the flush: the survivor set has
            # to describe the same instant as the retrieval above, and the
            # replay's own ops can be dropped later in the phase.
            check_prefix_closure(c, scenario, victim_id, ends, retr)
            # Ledger flush after the last counter-changing request.
            time.sleep(6)
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
        finally:
            teardown([vllm])
        expect_clean_exit(c, f"{scenario} vllm", vllm)
        return stores, check_ledger(c, scenario, max_evicted=max_evicted)

    try:
        # --- Phase A: default cap, the coalescing control ---
        control, ledger_a = drive(
            "S14", {"lmcache.mp.lazy_offload": True}, max_evicted=4
        )
        expected = sizing["stored"]
        # The premise every batching assertion rests on, checked where it is
        # readable: the settled ledger of a phase with no backlog. One op per
        # prefill step of the whole workload accounts for every admission, so
        # the request under test contributed exactly its own op count -- and
        # that count is > 1, or there was never a batch to split.
        if ledger_a:
            workload_ops = sizing["ops"] + sizing["S14-filler-ops"]
            c.expect(
                sizing["ops"] >= 2 and ledger_a["admitted"] == workload_ops,
                f"the default-cap phase admitted one op per prefill step and "
                f"nothing else, and the request under test is a "
                f"{sizing['ops']}-op batch (admitted={ledger_a['admitted']}, "
                f"workload={workload_ops})",
            )
        c.expect(
            expected in control,
            f"the default cap coalesced the whole buffered batch into one "
            f"store (expected a {expected}-token submission, saw {control})",
        )
        # Guards the phase-B bound against a store size that happens to sit
        # under one step's budget for reasons other than the throttle.
        c.expect(
            expected > _STEP_BUDGET_TOKENS,
            f"that coalesced store exceeds one step's tokens, so the "
            f"phase-B bound is a real difference (coalesced={expected}, "
            f"step budget={_STEP_BUDGET_TOKENS})",
        )

        # --- Phase B: cap of 1 ---
        cache_clear()
        c.expect(cache_object_count() == 0, "cache cleared between phases")
        # A wider eviction bound than phase A on purpose: one op per step is
        # a slower drain, so more ops can lose the race to an allocation. The
        # bound is still a sensor -- a cap that lost most of the workload
        # would blow through it -- and how far the two phases actually differ
        # is a measurement this scenario reports.
        throttled, ledger_b = drive(
            "S14b",
            {
                "lmcache.mp.lazy_offload": True,
                "lmcache.mp.lazy_offload_max_drain_per_step": 1,
            },
            max_evicted=10,
        )
        c.expect(
            throttled != [] and max(throttled) <= _STEP_BUDGET_TOKENS,
            f"a cap of 1 kept every store submission down to a single op "
            f"(largest={max(throttled) if throttled else None}, one op "
            f"<= {_STEP_BUDGET_TOKENS} tokens, stores={throttled})",
        )
        c.expect(
            len(throttled) > len(control),
            f"the same workload took strictly more submissions under the cap "
            f"(throttled={len(throttled)}, default={len(control)})",
        )
        if ledger_a and ledger_b:
            # The cap is about *when* an op is emitted, never whether it is
            # admitted, so the workload's admissions may not shrink under it.
            # (Not equality: pinned pending blocks preempt more under a slower
            # drain, and a resumed request re-admits the ops it re-prefills.)
            c.expect(
                ledger_b["admitted"] >= ledger_a["admitted"],
                f"the cap did not cost the workload admissions "
                f"(admitted: default={ledger_a['admitted']}, "
                f"throttled={ledger_b['admitted']})",
            )
            c.expect(
                ledger_b["emitted"] >= sizing["ops"],
                f"the in-flight hand-off kept emitting under the cap "
                f"(emitted={ledger_b['emitted']})",
            )
            # The sizing sensor, in both directions. Without the phase-A
            # reading a counter wired to always fire would pass here.
            c.expect(
                ledger_a["throttled_drains"] == 0,
                f"the default cap never bound on this workload, so an "
                f"operator sees no sizing complaint "
                f"(throttled_drains={ledger_a['throttled_drains']})",
            )
            c.expect(
                ledger_b["throttled_drains"] > 0,
                f"the cap that lost the tail reported itself as the cause "
                f"(throttled_drains={ledger_b['throttled_drains']})",
            )
        # The WARNING is the operator-facing half: the counter alone does not
        # say which knob to turn. Once per process, and only where the cap
        # both bound and lost ops -- phase A must stay silent.
        warned_a = grep_lines("S14", _CAP_WARNING)
        warned_b = grep_lines("S14b", _CAP_WARNING)
        c.expect(
            warned_a == [] and len(warned_b) == 1,
            f"exactly the throttled phase warned about max_drain_per_step, "
            f"once (default={len(warned_a)}, capped={len(warned_b)})",
        )
        c.expect(
            all("max_drain_per_step=1" in line for line in warned_b),
            f"the warning names the configured value it is complaining "
            f"about (lines={warned_b})",
        )
    finally:
        teardown([server])
    expect_clean_exit(c, "mp-server", server)
    return c


#: vLLM's GPU block size for this model. An op covers whole blocks only, so
#: two prompts sharing a prefix admit ops with the same content key exactly
#: while their difference stays inside the last, partial block.
_GPU_BLOCK_TOKENS = 16

#: vllm args for the scenarios that want one op per request: a 448-block pool
#: and no step budget, so a 3072-token prefill is scheduled in a single step.
_ONE_STEP_POOL = [
    "--gpu-memory-utilization", "0.5",
    "--max-model-len", "4096",
    "--num-gpu-blocks-override", "448",
]


def scenario_S15() -> Check:
    """Deduplication is bounded by unique content, not by request count.

    The *reason* dedup exists: what the queue holds is bounded by the unique
    content resident on the GPU, so N requests sharing a prefix buffer one
    copy of it, not N. That is the only memory-safety property in the
    design -- the pending queue lives in the scheduler process, on the hot
    path, with no cap of its own -- and S11 established the mechanism with
    two requests, which cannot tell a bound from a coincidence. Layer 0 goes
    to three.

    Twelve followers share a 3072-token prefix, each with a unique tail
    short enough to stay inside the prefix's last partial block, so each
    admits one op whose content key matches the cover's. Nothing presses on
    the pool, and a follower's own prefix-cache hit lifts the cover's blocks
    out of the free queue for as long as it runs, so no drain can take the
    cover away mid-experiment -- which is what lets all twelve meet the same
    cover.

    The bound is read off the ledger lines logged across that window: with
    twelve followers admitted and deduplicated, no line may show more than
    the cover's one op admitted or pending. A degeneration to one buffered
    copy per request -- the failure that grows the queue with the request
    rate -- shows up here and nowhere else, since every other signal
    (retrieval, object count, byte-identity) is identical either way.

    Then pressure drains the cover and the first follower replays: its
    prefix only ever stored under the cover's op, so retrieving all of it is
    what says the dedup pointed at live data rather than at a corpse.
    """
    c = Check()
    server = start_server("S15")
    shared = long_prompt("share", 150)  # 3072 tokens = 192 blocks, one op
    followers = [shared + f" Next, cover angle {i}." for i in range(12)]
    try:
        vllm = start_vllm_under(server, "S15",
            {"lmcache.mp.lazy_offload": True}, _ONE_STEP_POOL)
        try:
            cover_end = stored_tokens(shared)
            blocks = prompt_tokens(shared) // _GPU_BLOCK_TOKENS
            tails = {prompt_tokens(f) // _GPU_BLOCK_TOKENS for f in followers}
            # The premise of the whole scenario: if a tail spilled into a new
            # block, that follower's op would cover more blocks than the
            # cover's, its content key would differ, and it would admit
            # instead of deduplicating -- reading as a policy failure.
            c.expect(
                tails == {blocks},
                f"every follower's tail stays inside the cover's last partial "
                f"block, so their ops cover the same {blocks} blocks "
                f"(follower block counts={sorted(tails)})",
            )
            # The cover goes first, so it is the op every follower meets and
            # the one whose id the drop lines name.
            complete(shared, request_id="s15-cover")
            time.sleep(3)
            n = cache_object_count()
            c.expect(
                n == 0,
                f"the unpressed pool stored nothing, so the cover is still "
                f"buffered when the followers arrive (objects={n})",
            )
            outs = [
                complete(follower, request_id=f"s15-f{i}")
                for i, follower in enumerate(followers)
            ]
            time.sleep(3)
            n = cache_object_count()
            c.expect(
                n == 0,
                f"no follower forced a drain, so all of them met the same "
                f"cover (objects={n})",
            )
            window = [counters for _, counters in grep_ledgers("S15")]
            c.expect(
                len(window) >= 2,
                f"the follower window logged ledger lines to read the bound "
                f"from ({len(window)} lines)",
            )
            worst_pending = max((ln["pending"] for ln in window), default=-1)
            worst_admitted = max((ln["admitted"] for ln in window), default=-1)
            print(
                f"[S15] across {len(window)} ledger lines with "
                f"{len(followers)} followers sharing one prefix: "
                f"max pending={worst_pending}, max admitted={worst_admitted}"
            )
            c.expect(
                worst_pending == 1 and worst_admitted == 1,
                f"{len(followers)} requests sharing one prefix buffered one "
                f"copy of it: the queue never held more than the cover's op "
                f"(max pending={worst_pending}, max admitted={worst_admitted})",
            )

            for seed in ("p1", "p2", "p3", "p4", "p5"):  # evict the prefix
                complete(long_prompt(seed, 100))
            time.sleep(6)
            n = cache_object_count()
            c.expect(n > 0, f"pressure drains the cover (objects={n})")
            n0 = len(grep_retrieved("S15"))
            replay = complete(followers[0], request_id="s15-r0")
            c.expect(
                replay == outs[0],
                "a deduplicated follower replays byte-identically after the "
                "cover drained",
            )
            # Against the *cover's* id: the follower's own op never entered
            # the queue, so whether its prefix survived is a statement about
            # the cover's op, and the follower's replay is how it is read.
            check_prefix_closure(c, "S15", "s15-cover", [cover_end],
                                 grep_retrieved("S15")[n0:])
            # Ledger flush after the last counter-changing request.
            time.sleep(6)
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
        finally:
            teardown([vllm])
        expect_clean_exit(c, "vllm", vllm)
        ledger = check_ledger(c, "S15", max_evicted=3)
        if ledger:
            # Exact: one per follower. A lower count means some follower
            # buffered its own copy (the bound is gone); a higher one means
            # something outside this experiment deduplicated, and the
            # `pending` reading above no longer describes what it claims.
            c.expect(
                ledger["deduplicated"] == len(followers),
                f"every follower deduplicated, exactly once each "
                f"(deduplicated={ledger['deduplicated']}, "
                f"followers={len(followers)})",
            )
    finally:
        teardown([server])
    expect_clean_exit(c, "mp-server", server)
    return c


#: The two GPUs the tensor-parallel scenario runs its engine on. The MP
#: server keeps its own `SMOKE_GPU`; L1 is host memory, so sharing the first
#: device with the engine costs nothing.
TP_GPUS = os.environ.get("SMOKE_TP_GPUS", "2,3")


def scenario_S16() -> Check:
    """Tensor parallelism: two workers, one policy, receipts from both.

    Every other scenario runs TP=1, where the store path has exactly one
    worker and the completion receipt has one source. The policy itself is
    unaffected -- it lives in the scheduler process and there is one of it at
    any TP -- so what this exercises is the plumbing around it: a released
    batch fans out to both ranks, each stores its own shard of the KV, and
    `notify_store_complete` has to fire for the request once, from whatever
    the engine aggregates back. A receipt that goes missing on one rank
    leaves the request in flight forever, which stalls its remaining ops and
    holds its session open.

    Same shape as the throttle scenario's control phase: a 1965-token
    request buffered against an unpressed pool, five fillers to drive the
    drain, a replay judged by prefix closure. Two TP-specific readings on
    top: the engine log must show two workers, and once everything has
    drained the server must hold no session at all -- the direct reading of
    "every in-flight batch was receipted", and the assertion a lost rank-1
    receipt fails.
    """
    c = Check()
    # Both devices, on both processes: the engine needs them to shard over,
    # and the server needs them to resolve each rank's KV caches by device
    # UUID over CUDA IPC.
    tp_env = {"CUDA_VISIBLE_DEVICES": TP_GPUS}
    server = start_server("S16", tp_env)
    victim = long_prompt("shard", 100)  # 1965 tokens -> 4 ops
    try:
        vllm = start_vllm_under(server, "S16",
            {"lmcache.mp.lazy_offload": True},
            _SMALL_POOL + ["--tensor-parallel-size", "2"], tp_env)
        try:
            # The premise: this really is a two-rank engine. Each worker logs
            # its own `world_size=N rank=R` line from `parallel_state`, so a
            # silent fallback to one rank -- which would make the whole
            # scenario a duplicate of S14 phase A -- fails here rather than
            # passing itself off as TP coverage.
            init = {
                (m.group(1), m.group(2))
                for line in grep_lines("S16", r"world_size=\d+ rank=\d+")
                for m in re.finditer(r"world_size=(\d+) rank=(\d+)", line)
            }
            c.expect(
                init == {("2", "0"), ("2", "1")},
                f"the engine came up with two tensor-parallel workers "
                f"(world_size/rank lines={sorted(init)})",
            )
            ends = op_prefix_ends(victim)
            out = complete(victim, request_id="s16-victim")
            time.sleep(3)
            n = cache_object_count()
            c.expect(
                n == 0,
                f"the unpressed pool stored nothing, so the ops are buffered "
                f"going into the drain (objects={n})",
            )
            for seed in ("q1", "q2", "q3", "q4", "q5"):
                complete(long_prompt(seed, 100))
            time.sleep(6)
            n = cache_object_count()
            c.expect(n > 0, f"pressure drains the buffered ops (objects={n})")
            n0 = len(grep_retrieved("S16"))
            replay = complete(victim, request_id="s16-replay")
            c.expect(
                replay == out,
                "the request replays byte-identically from a two-rank engine",
            )
            check_prefix_closure(c, "S16", "s16-victim", ends,
                                 grep_retrieved("S16")[n0:], ranks=2)
            # Ledger flush after the last counter-changing request.
            time.sleep(6)
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
            sessions = active_sessions()
            c.expect(
                sessions == 0,
                f"every in-flight batch was receipted and every drained "
                f"request's session ended (active_sessions={sessions})",
            )
        finally:
            teardown([vllm])
        expect_clean_exit(c, "vllm", vllm)
        ledger = check_ledger(c, "S16", max_evicted=4)
        if ledger:
            # Nothing may still be in flight or queued: a rank that never
            # reported would leave the emitting request stuck, and its
            # remaining ops with it.
            c.expect(
                ledger["pending"] == 0 and ledger["emitted"] >= len(ends),
                f"the queue settled empty under TP=2 "
                f"(pending={ledger['pending']}, emitted={ledger['emitted']})",
            )
    finally:
        teardown([server])
    expect_clean_exit(c, "mp-server", server)
    return c


#: vllm args for the preemption storm: 224 blocks * 16 = 3584 tokens of KV,
#: less than two 2165-token prompts need at once, so a concurrent pair
#: cannot both keep their prefix and one is preempted. The model length is
#: cut to 3072 to stay inside the pool.
_TINY_POOL_MODEL_LEN = 3072
_TINY_POOL_BLOCKS = 224
_S17_MAX_TOKENS = 32  # `complete`'s default generation length

_TINY_POOL = [
    "--gpu-memory-utilization", "0.5",
    "--max-model-len", str(_TINY_POOL_MODEL_LEN),
    "--num-gpu-blocks-override", str(_TINY_POOL_BLOCKS),
    "--max-num-batched-tokens", "512",
]

#: The connector's preemption drop line. Unlike the two aggregate drop
#: lines it names no ops, only a count -- the path discards *all* of the
#: request's buffered ops, so the id and the count say which ones. That is
#: also why `check_prefix_closure` cannot be used on a preempted request:
#: `drop_prefixes` reads prefix ends out of the drop lines and would report
#: a preempt-dropped op as a survivor.
_PREEMPT_DROP_LINE = r"dropped (\d+) buffered store op\(s\) of preempted request"


def preempt_drop_counts(scenario: str) -> list[int]:
    """Ops discarded by each preemption reset, in log order.

    Truncated at the last ledger line like `count_drop_lines`, so the sum
    is comparable with the settled ledger's `dropped_on_request_drop`.

    Args:
        scenario: The scenario whose engine log to read.

    Returns:
        One op count per preemption drop line.
    """
    return [
        int(m.group(1))
        for m in re.finditer(
            _PREEMPT_DROP_LINE, _log_through_last_ledger(scenario)
        )
    ]


def scenario_S17() -> Check:
    """Preemption storm: the resumed request is re-admitted, not blacklisted.

    `rejected_prefix_broken` is the one admission outcome no scenario has
    ever produced. The path this scenario was built to construct -- a
    request loses buffered ops while preempted, then resumes and re-admits
    into the blacklist -- turns out to be closed by design, and the point
    of running it is to hold that reading against a real engine rather
    than against a code walk:

    - The blacklist is set by the eviction drop and by gate 3, both of
      which need the request's blocks to be in the free queue. A running
      request holds its own blocks, so its snapshot cannot break; the
      blocks only reach the free queue when it finishes or is preempted.
    - A finished request admits nothing more, and is cleared from the
      blacklist when its queue drains empty.
    - A preempted request's resume goes through the tracker reset, which
      calls `drop_request` -- and that discards the blacklist entry
      *before* the resumed request's first admission, because vLLM asks
      the connector for the tracker (`get_num_new_matched_tokens`) before
      it schedules the resumed tokens.

    So on a healthy store path with a fully hashable model the branch is
    defensive: it is reachable only through an unhashable block (hybrid
    attention) or a failed store batch. The scenario drives the preemption
    path as hard as this box allows -- four 2165-token prompts against
    3584 tokens of KV -- and asserts the reading: preemption drops fire,
    the drops balance the ledger, and the reject counter stays 0.

    The serial tail request additionally shows that a preempted request's
    re-prefill is not merely dropped: it re-admits from token zero, so the
    prefix a fresh engine gets back is still one of the request's op ends.
    """
    c = Check()
    server = start_server("S17")
    # Seeds decide the length as much as the sentence count does -- the
    # per-sentence filler is seeded text -- so these are measured below
    # rather than assumed. p0..p3 are 2165 tokens (136 blocks) each, so any
    # two of them overflow the 224-block pool; the tail is 1965 (123).
    storm = [long_prompt(f"p{i}", 100) for i in range(4)]
    tail = long_prompt("pt", 100)
    try:
        vllm = start_vllm_under(
            server, "S17", {"lmcache.mp.lazy_offload": True}, _TINY_POOL
        )
        try:
            # A prompt whose generation cannot fit the model length is
            # rejected with a bare HTTP 400 that logs no traceback, so the
            # scenario would die on an unrelated-looking error instead of
            # failing an assertion.
            sizes = [prompt_tokens(p) for p in storm + [tail]]
            c.expect(
                max(sizes) + _S17_MAX_TOKENS <= _TINY_POOL_MODEL_LEN,
                f"every prompt leaves room for its generation inside the "
                f"model length (max={max(sizes)}+{_S17_MAX_TOKENS}, "
                f"limit={_TINY_POOL_MODEL_LEN}, sizes={sizes})",
            )
            c.expect(
                2 * (min(sizes) // _GPU_BLOCK_TOKENS) > _TINY_POOL_BLOCKS,
                f"two prompts cannot hold their prefixes at once, so the "
                f"storm must preempt (blocks per prompt>="
                f"{min(sizes) // _GPU_BLOCK_TOKENS}, pool="
                f"{_TINY_POOL_BLOCKS})",
            )
            with ThreadPoolExecutor(max_workers=4) as pool:
                outs = list(pool.map(complete, storm))
            c.expect(
                all(outs),
                f"all {len(storm)} storm requests completed with output "
                f"under a pool too small to hold two of them",
            )
            ends = op_prefix_ends(tail)
            out_tail = complete(tail, request_id="s17-tail")
            c.expect(
                bool(out_tail),
                f"the serial tail request completed with output "
                f"(op ends={ends})",
            )
            time.sleep(6)
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
        finally:
            teardown([vllm])
        expect_clean_exit(c, "vllm", vllm)
        # The storm is built to lose ops: five prompts of 136 blocks each
        # against a 224-block pool. The bound is a sensor, not a target.
        ledger = check_ledger(c, "S17", max_evicted=25)
        drops = preempt_drop_counts("S17")
        c.expect(
            drops != [],
            f"preemption discarded buffered ops at least once -- the "
            f"premise of everything below (drop lines={drops})",
        )
        if ledger:
            c.expect(
                ledger["dropped_on_request_drop"] == sum(drops),
                f"the preemption drop lines sum to the ledger "
                f"(sum={sum(drops)}, "
                f"dropped_on_request_drop={ledger['dropped_on_request_drop']})",
            )
            c.expect(
                ledger["rejected_prefix_broken"] == 0,
                f"no admission was rejected as prefix-broken: the tracker "
                f"reset clears the blacklist before the resumed request "
                f"re-admits (rejected_prefix_broken="
                f"{ledger['rejected_prefix_broken']})",
            )
            c.expect(
                ledger["rejected_unhashed"] == 0,
                f"the model hashes every block, so nothing entered the "
                f"blacklist by that route either "
                f"(rejected_unhashed={ledger['rejected_unhashed']})",
            )
            c.expect(
                ledger["admitted"] > len(storm),
                f"the storm re-admitted after its resets rather than "
                f"stalling (admitted={ledger['admitted']}, "
                f"prompts={len(storm) + 1})",
            )

        # --- Phase 2: what a preempted request left behind is a front run.
        # A preemption discards every buffered op of the request, and the
        # resumed request re-produces them from token zero, so the stored
        # prefix is still one of the op ends -- never a middle chunk.
        vllm = start_vllm_under(server, "S17b", {}, _TINY_POOL)
        try:
            n0 = len(grep_retrieved("S17"))
            replay = complete(tail, request_id="s17-tail")
            c.expect(
                replay == out_tail,
                "the tail request replays byte-identically from a fresh "
                "engine",
            )
            tail_retr = grep_retrieved("S17")[n0:]
            total = sum(tail_retr)
            c.expect(
                total in {0, *ends},
                f"the fresh engine got back one of the tail request's op "
                f"ends, not a partial or over-long prefix "
                f"(retrieved={total}, ends={ends}, lines={tail_retr})",
            )
            print(f"[S17] tail retrieval: {total} of {ends[-1]} tokens")
            # The tail alone can legally retrieve nothing -- it is the last
            # request, so nothing presses on the pool afterwards and its ops
            # can still be pending at shutdown (measured: 4 pending, 0
            # retrieved). Without the storm replays below, the byte-identity
            # check above would then be comparing two uncached runs of a
            # deterministic model, and phase 2 would assert nothing about
            # stored KV at all.
            n1 = len(grep_retrieved("S17"))
            replays = [complete(p) for p in storm]
            storm_retr = grep_retrieved("S17")[n1:]
            longest = max(stored_tokens(p) for p in storm)
            chunk = chunk_size()
            c.expect(
                all(replays) and storm_retr != [],
                f"the storm's own emissions come back on a fresh engine, so "
                f"phase 2 rests on a cache hit rather than on determinism "
                f"(replays={len(replays)}, retrievals={storm_retr})",
            )
            c.expect(
                all(v % chunk == 0 and 0 < v <= longest for v in storm_retr),
                f"every storm retrieval is a whole number of chunks inside "
                f"one prompt's prefix (chunk={chunk}, longest={longest}, "
                f"retrievals={storm_retr})",
            )
        finally:
            teardown([vllm])
        expect_clean_exit(c, "phase-2 vllm", vllm)
    finally:
        teardown([server])
    expect_clean_exit(c, "mp-server", server)
    return c


#: The MP server logs one of these per kernel group when the engine
#: registers its KV caches. `sw_size_tokens` is the group's attention
#: window, -1 for full attention.
#: `.*?` rather than `[^)]*`: the line carries a nested `shape_desc=(...)`
#: whose closing parenthesis ends a negated-class match early, and the
#: window is the last field.
_KERNEL_GROUP_LINE = r"KernelGroupInfo\(.*?sw_size_tokens=(-?\d+)\)"


def kernel_group_windows(scenario: str) -> list[int]:
    """Attention window per registered kernel group, in registration order.

    S18's whole premise is that the engine really did build interleaved
    local/global attention groups. Reading it off the registration lines
    rather than off the model's config is the difference between asserting
    what vLLM did and asserting what the config file says.

    Args:
        scenario: The scenario whose server log to read.

    Returns:
        One window size per kernel group; -1 for a full-attention group.

    Raises:
        FileNotFoundError: if the log is missing. An empty list would make
            "the groups are not all alike" vacuously false, i.e. a failure,
            which is the safe direction -- but the cause must be legible.
    """
    log = LOGDIR / f"{scenario}_server.log"
    if not log.exists():
        raise FileNotFoundError(f"no server log for {scenario}: {log}")
    return [
        int(m.group(1)) for m in re.finditer(_KERNEL_GROUP_LINE, log.read_text())
    ]


#: The pending store's warning when admission rejects an operation whose
#: blocks carry no prefix-cache hash.
_UNHASHED_LINE = r"skipping store for request (\S+) tokens \[(\d+), (\d+)\)"


def unhashed_skips(scenario: str) -> list[tuple[str, int, int]]:
    """Every hash-less-block skip the connector reported, in log order.

    Args:
        scenario: The scenario whose engine log to read.

    Returns:
        One (request_id, prefix_start, prefix_end) per warning line.
    """
    return [
        (m.group(1), int(m.group(2)), int(m.group(3)))
        for m in re.finditer(_UNHASHED_LINE, _vllm_log(scenario))
    ]


#: Engine config for the sliding-window scenario. Deliberately unpressed:
#: the replay has to hit vLLM's *own* prefix cache for its block table to
#: contain null blocks, and pool pressure evicts that cache. Measured -- the
#: same sequence under a 1400-block pool re-prefilled the replay from
#: scratch and produced no rejection at all.
_SWA_ARGS = [
    "--gpu-memory-utilization", "0.5",
    "--max-num-seqs", "4",
    "--max-num-batched-tokens", "512",
]

#: Generation length for the replay. It has to carry the request past the
#: next chunk boundary above the first pass's stored end, so the replay
#: produces a *second* store op after its first was rejected: that second
#: op is what makes `rejected_prefix_broken` observable.
_S18_REPLAY_TOKENS = 400


def scenario_S18() -> Check:
    """Sliding-window attention: the admission guard on hash-less blocks.

    Under sliding-window attention vLLM replaces a request's out-of-window
    blocks with `block_pool.null_block`, which has no block hash -- both
    while the request runs and, the case this scenario builds, when its
    prefix comes back from vLLM's own prefix cache. The KV of those
    positions no longer exists on the GPU for the sliding-window layers.

    Two engines on one server, same model and same config apart from the
    lazy flag, each running the same sequence: prompt, clear, replay.

    - Lazy: admission rejects the replay's first operation
      (`rejected_unhashed`), blacklists the request, and rejects the
      operation its continued generation produces afterwards
      (`rejected_prefix_broken`). Nothing is stored.
    - Eager: no admission check, so the replay stores under the same
      content-addressed keys the first pass wrote -- with different bytes.

    The eager phase is what makes the lazy phase mean something. "The
    counter incremented" only says a branch was taken; the byte comparison
    says what taking it avoided. It also pins the reading that the two
    rejections are reachable on real hardware at all: they are the two
    admission outcomes no full-attention scenario has ever produced, and
    S17 established that the preemption path cannot produce the second one.
    """
    c = Check()
    server = start_server("S18")
    prompt = long_prompt("swa", 100)
    try:
        vllm = start_vllm_under(
            server, "S18", {"lmcache.mp.lazy_offload": True}, _SWA_ARGS,
            model=SWA_MODEL,
        )
        try:
            windows = kernel_group_windows("S18")
            print(f"[S18] kernel group windows: {windows}")
            c.expect(
                len(set(windows)) > 1,
                f"the engine built interleaved attention groups "
                f"(distinct windows={sorted(set(windows))})",
            )
            local = [w for w in windows if w > 0]
            c.expect(
                local != [] and -1 in windows,
                f"both a sliding-window and a full-attention group exist "
                f"(windows={sorted(set(windows))})",
            )
            n = prompt_tokens(prompt)
            window = max(local) if local else 0
            # Without this the prompt could sit inside the window, every
            # block would stay live and hashed, and the scenario would
            # assert the absence of an event it never staged.
            c.expect(
                n > 3 * window,
                f"the prompt is several windows long, so most of it falls "
                f"out of the attention window (tokens={n}, window={window})",
            )
            end = stored_tokens(prompt)

            complete(prompt, max_tokens=16, request_id="swa-first")
            time.sleep(6)
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
            first = check_ledger(c, "S18", max_evicted=0)
            # The negative control. A guard that rejected every operation
            # would satisfy every assertion below; the fresh prefill must
            # admit normally, because its blocks are in the window as each
            # chunk is buffered.
            c.expect(
                first.get("admitted", 0) > 0,
                f"the fresh prefill admitted its operations "
                f"(admitted={first.get('admitted')})",
            )
            c.expect(
                first.get("rejected_unhashed") == 0
                and first.get("rejected_prefix_broken") == 0,
                f"and none of them was rejected "
                f"(rejected_unhashed={first.get('rejected_unhashed')}, "
                f"rejected_prefix_broken="
                f"{first.get('rejected_prefix_broken')})",
            )

            lazy_replay = complete(
                prompt, max_tokens=_S18_REPLAY_TOKENS, request_id="swa-replay"
            )
            time.sleep(6)
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
            second = check_ledger(c, "S18", max_evicted=0)
            c.expect(
                second.get("rejected_unhashed") == 1,
                f"the replay's covering operation was rejected exactly once "
                f"(rejected_unhashed={second.get('rejected_unhashed')})",
            )
            c.expect(
                second.get("rejected_prefix_broken", 0) >= 1,
                f"and the operations its continued generation produced were "
                f"rejected as prefix-broken "
                f"(rejected_prefix_broken="
                f"{second.get('rejected_prefix_broken')})",
            )
            # The blacklist is what connects the two counters: without it
            # the later operations would be admitted and stored without
            # their prefix, i.e. unreachable on retrieval.
            c.expect(
                second.get("admitted") == first.get("admitted"),
                f"the replay admitted nothing at all "
                f"(admitted={second.get('admitted')}, "
                f"before={first.get('admitted')})",
            )
            skips = unhashed_skips("S18")
            print(f"[S18] skips: {skips}")
            c.expect(
                len(skips) == 1,
                f"exactly one hash-less-block skip was reported "
                f"(skips={len(skips)})",
            )
            if skips:
                request_id, start, stop = skips[0]
                c.expect(
                    "swa-replay" in request_id,
                    f"the skip names the replay, not the first pass "
                    f"(request={request_id})",
                )
                c.expect(
                    (start, stop) == (0, end),
                    f"and covers the whole cached prefix "
                    f"(span=[{start}, {stop}), expected=[0, {end}))",
                )
            # The null blocks came from vLLM's own prefix cache, not from an
            # LMCache hit: nothing was ever stored, so nothing could be
            # retrieved. Without this the replay's cache hit has two
            # possible sources and the mechanism is not pinned.
            retrieved = grep_retrieved("S18")
            c.expect(
                retrieved == [],
                f"nothing was retrieved from LMCache (retrieved={retrieved})",
            )
            objects = cache_object_count()
            c.expect(
                objects == 0,
                f"the lazy run stored nothing (objects={objects})",
            )
        finally:
            teardown([vllm])
        expect_clean_exit(c, "lazy vllm", vllm)

        vllm = start_vllm_under(
            server, "S18b", {}, _SWA_ARGS, model=SWA_MODEL
        )
        try:
            complete(prompt, max_tokens=16, request_id="swa-eager-first")
            time.sleep(6)
            before = l1_md5s()
            c.expect(
                len(before) > 0,
                f"the eager run stored the first pass (objects={len(before)})",
            )
            cache_clear()
            c.expect(cache_object_count() == 0, "cache cleared between passes")
            eager_replay = complete(
                prompt, max_tokens=_S18_REPLAY_TOKENS,
                request_id="swa-eager-replay",
            )
            time.sleep(6)
            after = l1_md5s()
            c.expect(
                lazy_replay == eager_replay,
                "greedy outputs identical across the lazy and eager replays",
            )
            shared = sorted(set(before) & set(after))
            differing = [k for k in shared if before[k] != after[k]]
            print(
                f"[S18] eager replay: shared keys={len(shared)} "
                f"differing bytes={len(differing)}"
            )
            c.expect(
                len(shared) > 0,
                f"the eager replay rewrote content-addressed keys the first "
                f"pass had written (shared={len(shared)})",
            )
            # Measured: 7 of 8 chunks differ. The one that matches is the
            # chunk still inside the attention window, whose sliding-window
            # blocks were never nulled. Asserted as a floor rather than as 7
            # because the split depends on where the window boundary falls
            # inside the last chunk.
            c.expect(
                len(differing) > 0,
                f"and wrote different bytes under them -- this is what the "
                f"lazy guard refuses (differing={len(differing)} of "
                f"{len(shared)})",
            )
            for k in differing[:3]:
                print(f"[S18] {k}: {before[k]} -> {after[k]}")
            # The guard is the lazy path's; the eager path has no admission
            # step to report from.
            eager_skips = unhashed_skips("S18b")
            c.expect(
                eager_skips == [],
                f"the eager path reported no skip (skips={len(eager_skips)})",
            )
        finally:
            teardown([vllm])
        expect_clean_exit(c, "eager vllm", vllm)
    finally:
        teardown([server])
    expect_clean_exit(c, "mp-server", server)
    return c


#: Engine config for S19. 1400 blocks over six kernel groups leaves each
#: group 233 blocks (3728 tokens), enough for the 3072-token model length
#: and short enough that the free queue turns over while the requests run.
#: Measured: drains happen and no request is preempted, which is what this
#: scenario needs -- a resumed request's prefix comes back from vLLM's own
#: prefix cache, i.e. from null blocks, and the eager reference would then
#: be storing the very garbage S18 is about.
#:
#: The floor is `6 * max_model_len / block_size`: at 448 blocks (S6's
#: number) each group gets 74, or 1184 tokens, and the engine refuses to
#: start.
_SWA_PRESSURE = [
    "--gpu-memory-utilization", "0.5",
    "--max-model-len", "3072",
    "--num-gpu-blocks-override", "1400",
    "--max-num-batched-tokens", "512",
]


def scenario_S19() -> Check:
    """Content verification on a hybrid-attention model: lazy bytes == eager.

    S6's subject on S18's model. S18 shows that the lazy path *refuses* the
    one case the eager path gets wrong on a sliding-window model; it says
    nothing about the ops it does store there, and those are the ops a
    reviewer asks about once the PR is shown running on such a model.

    The hazard is specific to sliding-window attention: a block leaves a
    *running* request's table when the window slides past it and goes back
    on the free queue with its hash intact, so an operation buffered
    earlier can be drained long after its data stopped being the request's.
    The guard against that is the block-hash snapshot -- reallocation
    clears the hash -- and this scenario is where that guard is measured
    against real KV rather than against a fake pool.

    A high `dropped_evicted` is expected and is not a defect: the attention
    window recycles blocks far faster than full attention does, so the
    policy loses a larger share of its buffered ops here than on Qwen. The
    subject is the bytes of the ops that survived.
    """
    c = Check()
    server = start_server("S19")
    prompts = [long_prompt(s, 100) for s in ("h1", "h2", "h3", "h4")]
    try:
        vllm = start_vllm_under(
            server, "S19", {"lmcache.mp.lazy_offload": True}, _SWA_PRESSURE,
            model=SWA_MODEL,
        )
        try:
            chunk = chunk_size()
            total_chunks = sum(stored_tokens(p) // chunk for p in prompts)
            lazy_outs = [complete(p) for p in prompts]
            time.sleep(6)
            complete(long_prompt("flush", 2), 4)
            time.sleep(1)
            lazy = l1_md5s()
        finally:
            teardown([vllm])
        expect_clean_exit(c, "lazy vllm", vllm)
        print(f"[S19] lazy stored {len(lazy)} of {total_chunks} chunks")
        # A floor, not an equality: the window recycles blocks under the
        # drain. It exists so that "byte-identical on all common chunks"
        # cannot pass on a run that stored almost nothing.
        floor = total_chunks // 3
        c.expect(
            len(lazy) >= floor,
            f"the lazy run stored a substantial share of the chunks "
            f"(n={len(lazy)}, floor={floor}, total={total_chunks})",
        )
        c.expect(
            _EMPTY_MD5 not in lazy.values(),
            f"no stored object is an empty buffer "
            f"(empty={sum(1 for h in lazy.values() if h == _EMPTY_MD5)})",
        )
        c.expect(
            len(set(lazy.values())) == len(lazy),
            f"every stored chunk has distinct bytes "
            f"(unique={len(set(lazy.values()))}, objects={len(lazy)})",
        )
        # Measured 8 of 21 on this config. The bound is the sensor: without
        # one, a regression that drops everything reads as ALL PASS through
        # the floor above only.
        ledger = check_ledger(c, "S19", max_evicted=16)
        if ledger:
            # No preemption, by construction -- see `_SWA_PRESSURE`. A
            # resumed request would re-prefill from null blocks and the
            # eager reference below would no longer be a reference.
            c.expect(
                ledger["dropped_on_request_drop"] == 0,
                f"no request was preempted "
                f"(dropped_on_request_drop={ledger['dropped_on_request_drop']})",
            )
            # The lazy path must not have refused these: nothing here
            # replays a cached prefix, so every block is hashed.
            c.expect(
                ledger["rejected_unhashed"] == 0
                and ledger["rejected_prefix_broken"] == 0,
                f"fresh prefills on this model admit normally "
                f"(rejected_unhashed={ledger['rejected_unhashed']}, "
                f"rejected_prefix_broken={ledger['rejected_prefix_broken']})",
            )

        cache_clear()
        c.expect(cache_object_count() == 0, "cache cleared between runs")

        vllm = start_vllm_under(
            server, "S19b", {}, _SWA_PRESSURE, model=SWA_MODEL
        )
        try:
            eager_outs = [complete(p) for p in prompts]
            time.sleep(6)
            eager = l1_md5s()
        finally:
            teardown([vllm])
        expect_clean_exit(c, "eager vllm", vllm)
        print(f"[S19] eager stored {len(eager)} of {total_chunks} chunks")
        c.expect(
            eager_outs == lazy_outs,
            "greedy outputs identical across lazy/eager runs",
        )
        c.expect(
            unhashed_skips("S19") == [],
            "the lazy run reported no hash-less-block skip",
        )
        compare_stored_bytes(c, "S19", lazy, eager)
        mismatched = [k for k in lazy if k in eager and lazy[k] != eager[k]]
        for k in mismatched[:5]:
            print(f"[S19] MISMATCH {k}: lazy={lazy[k]} eager={eager[k]}")
    finally:
        teardown([server])
    expect_clean_exit(c, "mp-server", server)
    return c


class _Tee:
    """Write to two streams. Keeps the verdicts as a retained artifact.

    Without this the PASS/FAIL lines live only in the caller's terminal, so
    a later audit can re-derive them from the engine and server logs but
    cannot check what the run actually reported.
    """

    def __init__(self, primary: "object", mirror: "object") -> None:
        self._streams = (primary, mirror)

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)  # type: ignore[attr-defined]
            s.flush()  # type: ignore[attr-defined]
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()  # type: ignore[attr-defined]


#: The scenarios and their entry points. S5 is retired (see the note above
#: `scenario_S6`); every name here is swept, so the registry is the coverage
#: claim.
_SCENARIOS = {
    "S1": scenario_S1,
    "S2": scenario_S2,
    "S3": scenario_S3,
    "S4": scenario_S4,
    "S6": scenario_S6,
    "S9": scenario_S9,
    "S11": scenario_S11,
    "S12": scenario_S12,
    "S13": scenario_S13,
    "S14": scenario_S14,
    "S15": scenario_S15,
    "S16": scenario_S16,
    "S17": scenario_S17,
    "S18": scenario_S18,
    "S19": scenario_S19,
}

#: Assertions each scenario must actually execute, from the count of a
#: passing run. A scenario that runs fewer has skipped a block -- an empty
#: ledger short-circuiting `check_ledger`, a conditional whose premise did
#: not hold, an exception swallowed by a `finally` -- and must not report
#: ALL PASS: "3 assertions passed" and "34 assertions passed" print
#: identically otherwise.
#: (`main` adds one more of its own: the sessionless-request count.)
_MIN_CHECKS = {
    "S1": 6,
    "S2": 15,
    "S3": 15,
    "S4": 5,
    "S6": 20,
    "S9": 20,
    "S11": 36,
    "S12": 21,
    "S13": 27,
    "S14": 44,
    "S15": 22,
    "S16": 20,
    "S17": 24,
    "S18": 37,
    "S19": 22,
}


def main() -> int:
    scenario = sys.argv[1]
    LOGDIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.stdout, open(LOGDIR / f"{scenario}_driver.log", "w"))
    c = _SCENARIOS[scenario]()
    tb = grep_tracebacks(scenario)
    if tb:
        print(f"[check] FAIL: {len(tb)} Traceback/ERROR lines in logs:")
        for line in tb[:10]:
            print("   ", line)
        c.failures.append(f"{len(tb)} Traceback/ERROR lines in logs")
    warnings = grep_warnings(scenario)
    sessionless = [w for w in warnings if _WARN_NO_SESSION in w]
    expected_sessionless = _SESSIONLESS_REQUESTS[scenario]
    c.expect(
        len(sessionless) == expected_sessionless,
        f"only this scenario's sub-chunk requests ended a session the "
        f"server never created (warnings={len(sessionless)}, sub-chunk "
        f"requests={expected_sessionless})",
    )
    for line in sessionless:
        print("   ", line)
    unexpected = [w for w in warnings if _WARN_NO_SESSION not in w]
    if unexpected:
        print(f"[check] FAIL: {len(unexpected)} unexpected WARNING lines in logs:")
        for line in unexpected[:10]:
            print("   ", line)
        c.failures.append(f"{len(unexpected)} unexpected WARNING lines in logs")
    minimum = _MIN_CHECKS[scenario]
    print(f"[driver] {scenario}: {c.executed} assertions executed (min {minimum})")
    if c.executed < minimum:
        c.failures.append(
            f"only {c.executed} of at least {minimum} assertions executed"
        )
    verdict = "ALL PASS" if not c.failures else "FAILURES: " + "; ".join(
        c.failures
    )
    print(f"[driver] {scenario}: {verdict}")
    return 0 if not c.failures else 1


if __name__ == "__main__":
    sys.exit(main())

