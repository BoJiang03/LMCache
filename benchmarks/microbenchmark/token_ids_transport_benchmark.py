#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for the connector's per-op ``token_ids`` transmission cost.

Every ``LoadStoreOp`` the MP connector emits carries the request's **entire**
token id list (``LMCacheMPRequestTracker.get_token_ids()``), not just the tokens
the op actually covers.  At long context that whole list is copied, pickled,
msgpacked and shipped on every scheduler step, on three process hops.  The
stages below are each measured against the real production types imported from
this checkout:

====  =========================================================  ==============
 id   what it measures                                           whose CPU
====  =========================================================  ==============
 A    ``get_token_ids()`` -> ``list(ConstantList)``              scheduler
 B1   ``pickle.dumps`` of ``LMCacheMPConnectorMetadata``         scheduler
 B2   ``pickle.loads`` of the same                               each worker
 C1   ``_create_key``: ``tuple()`` + key + msgspec encode        each worker
 C2   ``msgspec`` decode back into ``IPCCacheServerKey``         server
 D    ``Session.set_tokens`` + memoized ``get_hashes``           server
 E    LOOKUP's uncached full ``compute_chunk_hashes``            server
====  =========================================================  ==============

B1/B2 are the scheduler -> worker hop: vLLM's ``shm_broadcast.MessageQueue``
pickles ``SchedulerOutput`` (connector metadata inside) once and every worker
unpickles its own copy.  C1/C2 are the worker -> LMCache server hop over ZMQ,
issued once per TP rank.  Stage Z additionally times a real cross-process
DEALER/ROUTER round trip as an end-to-end check on C1 + wire + C2.

Each stage is measured in two variants:

``full``
    what ships today -- the complete token id list.
``delta``
    the lower bound -- only ``token_ids[start:end]``, the tokens the op covers.

The gap between them is the headroom.  Run with::

    python benchmarks/microbenchmark/token_ids_transport_benchmark.py
    python benchmarks/microbenchmark/token_ids_transport_benchmark.py --quick

Stage D is memoized per request session on the server, so its ``full`` variant
is dominated by the ``list(key.token_ids)`` copy rather than by hashing; stage E
is not memoized and re-hashes the whole prefix on every LOOKUP.
"""

# Standard
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import argparse
import multiprocessing as mp
import os
import pickle
import random
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# First Party
from _native_stubs import install_native_stubs  # noqa: E402

install_native_stubs()

# Third Party
from vllm.v1.utils import ConstantList  # noqa: E402
import msgspec  # noqa: E402
import zmq  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey  # noqa: E402
from lmcache.v1.multiprocess.session import Session  # noqa: E402
from lmcache.v1.multiprocess.token_hasher import TokenHasher  # noqa: E402

CONTEXT_LENGTHS = (1_024, 4_096, 16_384, 32_768, 65_536, 131_072, 200_000)
QUICK_CONTEXT_LENGTHS = (1_024, 32_768, 200_000)

LMCACHE_CHUNK_SIZE = 256
"""LMCache tokens per chunk (``chunk_size`` default)."""

PREFILL_CHUNK_TOKENS = 8_192
"""vLLM ``max_num_batched_tokens``; sets how many steps a prefill takes."""

VOCAB_SIZE = 128_256
"""Llama-3 class vocabulary -- token ids wide enough to need multi-byte ints."""

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

STAGE_A = "A  sched get_token_ids"
STAGE_B1 = "B1 sched pickle.dumps"
STAGE_B2 = "B2 worker pickle.loads"
STAGE_C1 = "C1 worker key+encode"
STAGE_C2 = "C2 server msgspec dec"
STAGE_D = "D  server session hash"
STAGE_E = "E  server LOOKUP hash"
STAGE_Z = "Z  worker->server RTT"

WIRE_STAGES = (STAGE_B1, STAGE_C1)
"""Stages whose ``*_bytes`` are actual bytes crossing a process boundary."""


@dataclass(frozen=True)
class Timing:
    """Median and p90 wall time of a repeated measurement, in microseconds."""

    median_us: float
    p90_us: float


@dataclass(frozen=True)
class StageResult:
    """One stage measured at one context length, for both payload variants."""

    stage: str
    context_len: int
    full: Timing
    delta: Timing
    full_bytes: int
    delta_bytes: int

    @property
    def speedup(self) -> float:
        """How many times faster the delta variant is than the full one."""
        if self.delta.median_us == 0.0:
            return float("inf")
        return self.full.median_us / self.delta.median_us


def measure(fn: Callable[[], object], repeats: int, warmup: int = 3) -> Timing:
    """Time ``fn`` ``repeats`` times and return its median and p90.

    Args:
        fn: The zero-argument callable to time.
        repeats: Number of timed calls.
        warmup: Number of untimed calls made first.

    Returns:
        Timing: Median and p90 wall time in microseconds.
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1_000.0)
    samples.sort()
    return Timing(
        median_us=statistics.median(samples),
        p90_us=samples[min(len(samples) - 1, int(0.9 * len(samples)))],
    )


def make_token_ids(context_len: int, seed: int = 1234) -> list[int]:
    """Build a deterministic pseudo-random token id list of ``context_len``."""
    rng = random.Random(seed)
    return [rng.randrange(VOCAB_SIZE) for _ in range(context_len)]


def op_range(context_len: int) -> tuple[int, int]:
    """Return the ``[start, end)`` a mid-prefill store op would cover.

    Models the last chunked-prefill step of a ``context_len`` prompt: the op
    covers the final ``PREFILL_CHUNK_TOKENS`` window, chunk-aligned, while
    ``token_ids`` still carries the whole prompt.

    Args:
        context_len: The prompt length.

    Returns:
        The chunk-aligned ``(start, end)`` token indices of the op.
    """
    end = context_len - context_len % LMCACHE_CHUNK_SIZE
    start = max(0, end - PREFILL_CHUNK_TOKENS)
    return start, end


def build_block_ids(
    start: int, end: int, tokens_per_block: int = 16
) -> list[list[int]]:
    """Build one engine group's block ids covering ``[start, end)``."""
    return [list(range(start // tokens_per_block, end // tokens_per_block))]


def make_connector_metadata(
    token_ids: list[int], start: int, end: int, num_requests: int
) -> LMCacheMPConnectorMetadata:
    """Build the real connector metadata a scheduler step would broadcast.

    Args:
        token_ids: The token id list each op carries.
        start: Store op start index.
        end: Store op end index.
        num_requests: Requests sharing the step.

    Returns:
        The populated connector metadata.
    """
    metadata = LMCacheMPConnectorMetadata()
    for request_idx in range(num_requests):
        op = LoadStoreOp(
            token_ids=token_ids,
            block_ids=build_block_ids(start, end),
            start=start,
            end=end,
        )
        metadata.add_request_metadata(
            LMCacheMPRequestMetadata(
                request_id=f"req-{request_idx}",
                direction="STORE",
                op=op,
                cache_salt="",
                request_configs=None,
            )
        )
    return metadata


def make_ipc_key(token_ids: Sequence[int], start: int, end: int) -> IPCCacheServerKey:
    """Build the real server key a worker sends for a store op."""
    return IPCCacheServerKey(
        model_name=MODEL_NAME,
        world_size=8,
        worker_id=0,
        num_kv_readers=1,
        token_ids=tuple(token_ids),
        start=start,
        end=end,
        request_id="req-0",
        cache_salt="",
        request_configs=None,
    )


####
# Stage A -- scheduler-side list copy
####


def bench_stage_a(
    token_ids: list[int], start: int, end: int, repeats: int
) -> StageResult:
    """Measure ``get_token_ids()``'s ``list(ConstantList)`` copy.

    Args:
        token_ids: The request's full token id list.
        start: Store op start index.
        end: Store op end index.
        repeats: Number of timed calls.

    Returns:
        StageResult: Full-list copy versus delta-slice copy.
    """
    constant = ConstantList(token_ids)
    return StageResult(
        stage=STAGE_A,
        context_len=len(token_ids),
        full=measure(lambda: list(constant), repeats),
        delta=measure(lambda: list(constant[start:end]), repeats),
        # The CPython list's pointer array is the part that scales; the int
        # objects themselves are shared with the source list.
        full_bytes=8 * len(token_ids),
        delta_bytes=8 * (end - start),
    )


####
# Stages B1/B2 -- scheduler -> worker, pickled into vLLM's shm ring
####


def bench_stage_b(
    token_ids: list[int], start: int, end: int, repeats: int, num_requests: int
) -> tuple[StageResult, StageResult]:
    """Measure both halves of the pickle hop vLLM's ``MessageQueue`` performs.

    ``shm_broadcast.MessageQueue.enqueue`` pickles ``SchedulerOutput`` (with the
    connector metadata inside it) at ``pickle.HIGHEST_PROTOCOL``; each worker
    then unpickles its own copy out of the ring, so the two halves have
    different multipliers.

    Args:
        token_ids: The request's full token id list.
        start: Store op start index.
        end: Store op end index.
        repeats: Number of timed calls.
        num_requests: Concurrent requests sharing the step.

    Returns:
        The dumps-side and loads-side results.
    """
    full_meta = make_connector_metadata(token_ids, start, end, num_requests)
    delta_meta = make_connector_metadata(
        token_ids[start:end], 0, end - start, num_requests
    )
    full_blob = pickle.dumps(full_meta, protocol=pickle.HIGHEST_PROTOCOL)
    delta_blob = pickle.dumps(delta_meta, protocol=pickle.HIGHEST_PROTOCOL)

    dumps = StageResult(
        stage=STAGE_B1,
        context_len=len(token_ids),
        full=measure(
            lambda: pickle.dumps(full_meta, protocol=pickle.HIGHEST_PROTOCOL), repeats
        ),
        delta=measure(
            lambda: pickle.dumps(delta_meta, protocol=pickle.HIGHEST_PROTOCOL), repeats
        ),
        full_bytes=len(full_blob),
        delta_bytes=len(delta_blob),
    )
    loads = StageResult(
        stage=STAGE_B2,
        context_len=len(token_ids),
        full=measure(lambda: pickle.loads(full_blob), repeats),
        delta=measure(lambda: pickle.loads(delta_blob), repeats),
        full_bytes=0,
        delta_bytes=0,
    )
    return dumps, loads


####
# Stages C1/C2/Z -- worker -> LMCache server over ZMQ
####


def _echo_server(endpoint: str, ready_path: str) -> None:
    """Run a ROUTER that decodes each key the way the LMCache server does.

    Args:
        endpoint: The ``ipc://`` endpoint to bind.
        ready_path: File to create once the socket is bound.
    """
    context = zmq.Context(io_threads=1)
    socket = context.socket(zmq.ROUTER)
    socket.bind(endpoint)
    with open(ready_path, "w") as handle:
        handle.write("ready")
    while True:
        identity, payload = socket.recv_multipart()
        if payload == b"stop":
            socket.send_multipart([identity, b"stopped"])
            break
        key = msgspec.msgpack.decode(payload, type=IPCCacheServerKey)
        socket.send_multipart([identity, str(len(key.token_ids)).encode()])
    socket.close()
    context.term()


class ZmqHarness:
    """A DEALER client wired to a child-process ROUTER that decodes keys."""

    def __init__(self) -> None:
        self._dir = tempfile.mkdtemp(prefix="token-ids-bench-")
        self._endpoint = f"ipc://{self._dir}/router"
        ready_path = os.path.join(self._dir, "ready")
        ctx = mp.get_context("spawn")
        self._proc = ctx.Process(
            target=_echo_server, args=(self._endpoint, ready_path), daemon=True
        )
        self._proc.start()
        deadline = time.monotonic() + 60.0
        while not os.path.exists(ready_path):
            if time.monotonic() > deadline:
                raise RuntimeError("ZMQ echo server did not bind in time")
            time.sleep(0.01)
        self._context = zmq.Context(io_threads=1)
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.connect(self._endpoint)

    def roundtrip(self, payload: bytes) -> None:
        """Send an encoded key and wait for the server's decode to ack."""
        self._socket.send(payload)
        self._socket.recv()

    def close(self) -> None:
        """Stop the child process and release both ZMQ contexts."""
        try:
            self._socket.send(b"stop")
            self._socket.recv()
        finally:
            self._socket.close()
            self._context.term()
            self._proc.join(timeout=10)
            for name in os.listdir(self._dir):
                os.unlink(os.path.join(self._dir, name))
            os.rmdir(self._dir)


def bench_stage_c(
    harness: ZmqHarness, token_ids: list[int], start: int, end: int, repeats: int
) -> tuple[StageResult, StageResult, StageResult]:
    """Measure the worker's key build, the server's decode, and the round trip.

    ``tuple(token_ids)`` is inside C1's timed region because ``_create_key``
    performs it on every submit, and at 200k it is not free.

    Args:
        harness: The connected DEALER/ROUTER pair.
        token_ids: The request's full token id list.
        start: Store op start index.
        end: Store op end index.
        repeats: Number of timed calls.

    Returns:
        The encode-side, decode-side and end-to-end results.
    """
    delta_ids = token_ids[start:end]
    delta_end = end - start
    full_blob = msgspec.msgpack.encode(make_ipc_key(token_ids, start, end))
    delta_blob = msgspec.msgpack.encode(make_ipc_key(delta_ids, 0, delta_end))

    encode = StageResult(
        stage=STAGE_C1,
        context_len=len(token_ids),
        full=measure(
            lambda: msgspec.msgpack.encode(make_ipc_key(token_ids, start, end)), repeats
        ),
        delta=measure(
            lambda: msgspec.msgpack.encode(make_ipc_key(delta_ids, 0, delta_end)),
            repeats,
        ),
        full_bytes=len(full_blob),
        delta_bytes=len(delta_blob),
    )
    decode = StageResult(
        stage=STAGE_C2,
        context_len=len(token_ids),
        full=measure(
            lambda: msgspec.msgpack.decode(full_blob, type=IPCCacheServerKey), repeats
        ),
        delta=measure(
            lambda: msgspec.msgpack.decode(delta_blob, type=IPCCacheServerKey), repeats
        ),
        full_bytes=0,
        delta_bytes=0,
    )
    roundtrip = StageResult(
        stage=STAGE_Z,
        context_len=len(token_ids),
        full=measure(lambda: harness.roundtrip(full_blob), repeats),
        delta=measure(lambda: harness.roundtrip(delta_blob), repeats),
        full_bytes=0,
        delta_bytes=0,
    )
    return encode, decode, roundtrip


####
# Stages D/E -- server-side token handling
####


def bench_stage_d(
    token_ids: list[int], start: int, end: int, repeats: int
) -> StageResult:
    """Measure the server's per-store ``Session`` token handling.

    ``EngineContext.resolve_obj_keys`` calls ``Session.set_tokens(list(...))``
    and then ``Session.get_hashes(start, end)``.  Hashing is memoized across a
    request's stores, so a warm session pays only for new chunks -- but the
    ``list(key.token_ids)`` copy is paid in full on every store, by every rank.

    Args:
        token_ids: The request's full token id list.
        start: Store op start index.
        end: Store op end index.
        repeats: Number of timed calls.

    Returns:
        StageResult: Full-list session update versus delta-slice update.
    """
    delta_ids = token_ids[start:end]
    delta_end = end - start

    def warm(ids: list[int], hash_start: int, hash_end: int) -> Session:
        session = Session(
            request_id="req-0", hasher=TokenHasher(chunk_size=LMCACHE_CHUNK_SIZE)
        )
        session.set_tokens(ids)
        session.get_hashes(hash_start, hash_end)
        return session

    full_session = warm(token_ids, start, end)
    delta_session = warm(delta_ids, 0, delta_end)

    def full_store() -> None:
        full_session.set_tokens(list(token_ids))
        full_session.get_hashes(start, end)

    def delta_store() -> None:
        delta_session.set_tokens(list(delta_ids))
        delta_session.get_hashes(0, delta_end)

    return StageResult(
        stage=STAGE_D,
        context_len=len(token_ids),
        full=measure(full_store, repeats),
        delta=measure(delta_store, repeats),
        full_bytes=0,
        delta_bytes=0,
    )


def bench_stage_e(token_ids: list[int], start: int, repeats: int) -> StageResult:
    """Measure LOOKUP's uncached full-sequence chunk hash.

    ``PrefetchLookupHandler`` calls ``compute_chunk_hashes(list(key.token_ids))``
    with no ``end``, outside the session memoization, so it re-hashes the whole
    sequence.  The delta variant is the cost if the client carried the prefix
    hash forward and the server hashed only the new tokens.

    Args:
        token_ids: The request's full token id list.
        start: Store op start index, i.e. where the new tokens begin.
        repeats: Number of timed calls.

    Returns:
        StageResult: Whole-sequence hashing versus new-tokens-only hashing.
    """
    hasher = TokenHasher(chunk_size=LMCACHE_CHUNK_SIZE)
    delta_ids = token_ids[start:]
    prefix_hash = hasher.hash_tokens(token_ids[:LMCACHE_CHUNK_SIZE])
    return StageResult(
        stage=STAGE_E,
        context_len=len(token_ids),
        full=measure(lambda: hasher.compute_chunk_hashes(list(token_ids)), repeats),
        delta=measure(
            lambda: hasher.compute_chunk_hashes(
                list(delta_ids), prefix_hash=prefix_hash
            ),
            repeats,
        ),
        full_bytes=0,
        delta_bytes=0,
    )


####
# Reporting
####


def format_bytes(count: int) -> str:
    """Render a byte count in the largest unit that keeps it above 1."""
    if count == 0:
        return "-"
    for unit, scale in (("MB", 1 << 20), ("KB", 1 << 10)):
        if count >= scale:
            return f"{count / scale:.2f} {unit}"
    return f"{count} B"


def print_stage_table(results: Sequence[StageResult]) -> None:
    """Print one stage's results across all measured context lengths."""
    print(f"\n{results[0].stage}")
    print(
        f"  {'ctx':>8} {'full (us)':>10} {'delta (us)':>11} {'ratio':>8} "
        f"{'full wire':>11} {'delta wire':>11}"
    )
    for result in results:
        print(
            f"  {result.context_len:>8,} {result.full.median_us:>10.1f} "
            f"{result.delta.median_us:>11.1f} {result.speedup:>7.1f}x "
            f"{format_bytes(result.full_bytes):>11} "
            f"{format_bytes(result.delta_bytes):>11}"
        )


def print_request_rollup(
    by_stage: dict[str, dict[int, StageResult]],
    context_len: int,
    tensor_parallel_size: int,
) -> None:
    """Project the per-op numbers onto one full ``context_len`` prefill.

    A chunked prefill takes ``ceil(context_len / PREFILL_CHUNK_TOKENS)``
    scheduler steps, and each step emits one store op.  Scheduler-side stages
    run once per step; worker- and server-side stages run once per step per TP
    rank, because every rank submits its own store.  LOOKUP runs once.

    Args:
        by_stage: Stage name -> context length -> result.
        context_len: The prompt length to project.
        tensor_parallel_size: Number of worker ranks issuing stores.
    """
    steps = -(-context_len // PREFILL_CHUNK_TOKENS)
    per_rank = steps * tensor_parallel_size
    multipliers = {
        STAGE_A: steps,
        STAGE_B1: steps,
        STAGE_B2: per_rank,
        STAGE_C1: per_rank,
        STAGE_C2: per_rank,
        STAGE_D: per_rank,
        STAGE_E: 1,
    }

    print(
        f"\nPer-request rollup: {context_len:,}-token prefill, "
        f"TP={tensor_parallel_size}"
    )
    print(
        f"  {steps} scheduler steps x 1 store op, "
        f"prefill chunk={PREFILL_CHUNK_TOKENS:,}, lmcache chunk={LMCACHE_CHUNK_SIZE}"
    )
    print(
        f"  {'stage':<24} {'x':>5} {'full (ms)':>10} {'delta (ms)':>11} "
        f"{'saved (ms)':>11}"
    )

    total_full = 0.0
    total_delta = 0.0
    for stage, multiplier in multipliers.items():
        result = by_stage[stage][context_len]
        full_ms = multiplier * result.full.median_us / 1_000.0
        delta_ms = multiplier * result.delta.median_us / 1_000.0
        total_full += full_ms
        total_delta += delta_ms
        print(
            f"  {stage:<24} {multiplier:>5} {full_ms:>10.1f} {delta_ms:>11.1f} "
            f"{full_ms - delta_ms:>11.1f}"
        )
    print(
        f"  {'TOTAL CPU':<24} {'':>5} {total_full:>10.1f} {total_delta:>11.1f} "
        f"{total_full - total_delta:>11.1f}"
    )

    wire_full = sum(
        multipliers[stage] * by_stage[stage][context_len].full_bytes
        for stage in WIRE_STAGES
    )
    wire_delta = sum(
        multipliers[stage] * by_stage[stage][context_len].delta_bytes
        for stage in WIRE_STAGES
    )
    print(
        f"  bytes crossing process boundaries: "
        f"{format_bytes(wire_full)} -> {format_bytes(wire_delta)}"
    )

    # Stage Z sends an already-encoded blob, so it covers transport plus a
    # decode in the child process, but not the client's encode (C1).  It lands
    # at roughly the in-process C2 time, which says the ZMQ hop itself is small
    # next to the msgspec decode of the token list -- the decode is the cost.
    rtt = by_stage[STAGE_Z][context_len]
    decode = by_stage[STAGE_C2][context_len]
    print(
        f"  cross-check: ZMQ round trip {rtt.full.median_us:.0f} us vs "
        f"in-process decode {decode.full.median_us:.0f} us "
        f"-- transport is minor next to decoding the token list"
    )


def main() -> None:
    """Run every stage across the configured context lengths and report."""
    parser = argparse.ArgumentParser(
        description="Measure the connector's token_ids transmission overhead."
    )
    parser.add_argument(
        "--quick", action="store_true", help="measure 3 context lengths, fewer repeats"
    )
    parser.add_argument(
        "--repeats", type=int, default=0, help="timed calls per point (0 = auto)"
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="requests sharing one scheduler step in stage B",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=8, help="TP ranks for the rollup"
    )
    args = parser.parse_args()

    lengths = QUICK_CONTEXT_LENGTHS if args.quick else CONTEXT_LENGTHS
    repeats = args.repeats or (10 if args.quick else 50)

    print("Connector token_ids transmission overhead")
    print(f"  context lengths      : {', '.join(f'{n:,}' for n in lengths)}")
    print(f"  repeats per point    : {repeats}")
    print(f"  prefill chunk tokens : {PREFILL_CHUNK_TOKENS:,}")
    print(f"  lmcache chunk size   : {LMCACHE_CHUNK_SIZE}")
    print(f"  concurrent requests  : {args.concurrent_requests}")

    harness = ZmqHarness()
    by_stage: dict[str, dict[int, StageResult]] = {}
    try:
        for context_len in lengths:
            token_ids = make_token_ids(context_len)
            start, end = op_range(context_len)
            dumps, loads = bench_stage_b(
                token_ids, start, end, repeats, args.concurrent_requests
            )
            encode, decode, roundtrip = bench_stage_c(
                harness, token_ids, start, end, repeats
            )
            for result in (
                bench_stage_a(token_ids, start, end, repeats),
                dumps,
                loads,
                encode,
                decode,
                roundtrip,
                bench_stage_d(token_ids, start, end, repeats),
                bench_stage_e(token_ids, start, repeats),
            ):
                by_stage.setdefault(result.stage, {})[context_len] = result
    finally:
        harness.close()

    for stage in by_stage:
        print_stage_table([by_stage[stage][n] for n in lengths])

    print_request_rollup(by_stage, lengths[-1], args.tensor_parallel_size)


if __name__ == "__main__":
    main()
