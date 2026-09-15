#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for the connector's per-op ``token_ids`` transmission cost.

Every scheduler step, each in-flight request emits a store (and, once, a
retrieve) whose token ids travel three process hops: scheduler -> worker
through vLLM's shm ring, then worker -> LMCache server over ZMQ, once per TP
rank. Two independent things decide what that costs.

**How many tokens ride along.** An op covers ``[start, end)`` but can carry
the request's whole sequence anyway. ``lmcache.mp.delta_token_ids`` (default
on) makes it carry only its own range, at ``token_offset``, chained onto the
prefix the server's session already holds.

**How each token is encoded.** As ``list[int]`` every hop has to materialize
one Python ``int`` per token. Packed big-endian ``uint32``
(:mod:`lmcache.v1.multiprocess.token_codec`) makes the same hop a memcpy --
and since that is the exact byte layout blake3 already hashes, chunk hashes
are unchanged.

The two are independent, so all four combinations are measured and each
effect can be read off on its own:

``list+full``
    what shipped before this work: whole sequence, list of ints.
``list+delta``
    only the op's range, still a list -- isolates the delta change.
``packed+full``
    whole sequence, packed -- isolates the encoding change.
``packed+delta``
    what ships now: both.

====  =========================================================  ==============
 id   what it measures                                           whose CPU
====  =========================================================  ==============
 A    building the op's token payload on the scheduler           scheduler
 B1   ``pickle.dumps`` of ``LMCacheMPConnectorMetadata``         scheduler
 B2   ``pickle.loads`` of the same                               each worker
 C1   ``_create_key`` + msgspec encode                           each worker
 C2   ``msgspec`` decode back into the key                       server
 D    session token splice + memoized chunk hashing              server
 E    LOOKUP's uncached whole-sequence chunk hash                server
 Z    real cross-process ZMQ round trip (check on C1+wire+C2)    --
====  =========================================================  ==============

Stage E is unchanged by ``delta_token_ids``: LOOKUP is the call that seeds
the session every delta chains onto, so it always carries the whole
sequence. Its delta columns therefore repeat the matching full ones;
packing still speeds it up.

Run with::

    cd <repo root>
    PYTHONPATH=$PWD python \
        benchmarks/microbenchmark/token_ids_transport_benchmark.py [--quick]
"""

# Standard
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
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
    LMCacheMPRequestTracker,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey  # noqa: E402
from lmcache.v1.multiprocess.session import Session  # noqa: E402
from lmcache.v1.multiprocess.token_codec import pack_token_ids  # noqa: E402
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

LIST_FULL = "list+full"
LIST_DELTA = "list+delta"
PACKED_FULL = "packed+full"
PACKED_DELTA = "packed+delta"
VARIANTS = (LIST_FULL, LIST_DELTA, PACKED_FULL, PACKED_DELTA)

STAGE_A = "A  sched build payload"
STAGE_B1 = "B1 sched pickle.dumps"
STAGE_B2 = "B2 worker pickle.loads"
STAGE_C1 = "C1 worker key+encode"
STAGE_C2 = "C2 server msgspec dec"
STAGE_D = "D  server session hash"
STAGE_E = "E  server LOOKUP hash"
STAGE_Z = "Z  worker->server RTT"

WIRE_STAGES = (STAGE_B1, STAGE_C1)
"""Stages whose ``wire_bytes`` actually cross a process boundary."""


####
# Result types
####


@dataclass(frozen=True)
class Timing:
    """Median and p90 wall time of a repeated measurement, in microseconds."""

    median_us: float
    p90_us: float


@dataclass(frozen=True)
class StageResult:
    """One stage at one context length, measured for every variant."""

    stage: str
    context_len: int
    timings: dict[str, Timing]
    wire_bytes: dict[str, int] = field(default_factory=dict)

    def median_us(self, variant: str) -> float:
        """Median microseconds for ``variant``."""
        return self.timings[variant].median_us

    def bytes_for(self, variant: str) -> int:
        """Bytes ``variant`` puts on the wire, or 0 when the stage moves none."""
        return self.wire_bytes.get(variant, 0)

    def speedup(self, variant: str, baseline: str = LIST_FULL) -> float:
        """How many times faster ``variant`` is than ``baseline``."""
        if self.median_us(variant) == 0.0:
            return float("inf")
        return self.median_us(baseline) / self.median_us(variant)


def measure(
    fn: Callable[[], object],
    repeats: int,
    warmup: int = 3,
    setup: Callable[[], None] = lambda: None,
) -> Timing:
    """Time ``fn`` ``repeats`` times and return its median and p90.

    Args:
        fn: The zero-argument callable to time.
        repeats: Number of timed calls.
        warmup: Number of untimed calls made first.
        setup: Run before each call, outside the timed region. Use it to
            undo state ``fn`` mutates, so every iteration does equal work.

    Returns:
        Timing: Median and p90 wall time in microseconds.
    """
    for _ in range(warmup):
        setup()
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        setup()
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1_000.0)
    samples.sort()
    return Timing(
        median_us=statistics.median(samples),
        p90_us=samples[min(len(samples) - 1, int(0.9 * len(samples)))],
    )


####
# The pre-change representation, for the baseline column
####


@dataclass
class _ListLoadStoreOp:
    """``LoadStoreOp`` as it was before packing: token ids as a list.

    Mirrors the field layout so the baseline column is measured on a real
    pickle of a real object graph rather than estimated.
    """

    token_ids: list[int]
    block_ids: list[list[int]]
    start: int = 0
    end: int = 0
    skip_first_n_tokens: int = 0


@dataclass(order=True, frozen=True)
class _ListIPCCacheServerKey:
    """``IPCCacheServerKey`` as it was before packing: token ids as a tuple.

    msgspec encodes both this and the real key as maps of the same fields,
    so encoding and decoding it measures exactly the old wire cost.
    """

    model_name: str
    world_size: int
    worker_id: int | None
    token_ids: tuple[int, ...]
    start: int
    end: int
    request_id: str
    cache_salt: str = ""
    request_configs: dict[str, Any] | None = None
    num_kv_readers: int = 0


####
# Fixtures
####


def make_token_ids(context_len: int, seed: int = 1234) -> list[int]:
    """Build a deterministic pseudo-random token id list of ``context_len``."""
    rng = random.Random(seed)
    return [rng.randrange(VOCAB_SIZE) for _ in range(context_len)]


def op_range(context_len: int) -> tuple[int, int]:
    """Return the ``[start, end)`` a mid-prefill store op would cover.

    Models the last chunked-prefill step of a ``context_len`` prompt: the op
    covers the final ``PREFILL_CHUNK_TOKENS`` window, chunk-aligned.

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


def make_request_tracker(token_ids: list[int]) -> LMCacheMPRequestTracker:
    """Build a real request tracker over a text-only prompt.

    The tracker reads only these fields off the vLLM request, so a stand-in
    is enough to exercise the real payload-building code.

    Args:
        token_ids: The request's token ids.

    Returns:
        A tracker whose ``all_token_ids`` view wraps ``token_ids``.
    """
    request = SimpleNamespace(
        request_id="req-0",
        cache_salt="",
        prompt_token_ids=token_ids,
        all_token_ids=ConstantList(token_ids),
        mm_features=[],
        sampling_params=SimpleNamespace(extra_args=None),
    )
    return LMCacheMPRequestTracker(request)


def make_connector_metadata(
    op: LoadStoreOp | _ListLoadStoreOp, num_requests: int
) -> LMCacheMPConnectorMetadata:
    """Wrap ``op`` in the real connector metadata a scheduler step broadcasts.

    Args:
        op: The store op every request in the step carries.
        num_requests: Requests sharing the step.

    Returns:
        The populated connector metadata.
    """
    metadata = LMCacheMPConnectorMetadata()
    for request_idx in range(num_requests):
        metadata.add_request_metadata(
            LMCacheMPRequestMetadata(
                request_id=f"req-{request_idx}",
                direction="STORE",
                op=op,  # type: ignore[arg-type]
                cache_salt="",
                request_configs=None,
            )
        )
    return metadata


def make_ipc_key(
    token_bytes: bytes, start: int, end: int, token_offset: int = 0
) -> IPCCacheServerKey:
    """Build the real server key a worker sends for a store op."""
    return IPCCacheServerKey(
        model_name=MODEL_NAME,
        world_size=8,
        worker_id=0,
        num_kv_readers=1,
        token_bytes=token_bytes,
        start=start,
        end=end,
        request_id="req-0",
        cache_salt="",
        request_configs=None,
        token_offset=token_offset,
    )


def make_list_ipc_key(
    token_ids: Sequence[int], start: int, end: int
) -> _ListIPCCacheServerKey:
    """Build the pre-change key a worker used to send for a store op."""
    return _ListIPCCacheServerKey(
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
# Stage A -- scheduler builds the op's token payload
####


def bench_stage_a(
    token_ids: list[int], start: int, end: int, repeats: int
) -> StageResult:
    """Measure building one step's token payload on the scheduler.

    The packed variants pay for packing only the tokens this step added --
    the tracker keeps the packed prefix across steps -- plus the copy of
    whatever the op carries. The baseline copies the whole list every step.

    Args:
        token_ids: The request's full token id list.
        start: Store op start index.
        end: Store op end index.
        repeats: Number of timed calls.

    Returns:
        StageResult: Payload build cost per variant.
    """
    tracker = make_request_tracker(token_ids)
    packed_prefix = tracker.packed_token_slice(0, start)

    def rewind() -> None:
        """Leave the tracker packed only through ``start``, as a step starts."""
        tracker.packed_tokens = bytearray(packed_prefix)

    return StageResult(
        stage=STAGE_A,
        context_len=len(token_ids),
        timings={
            LIST_FULL: measure(tracker.get_token_ids, repeats),
            LIST_DELTA: measure(
                lambda: tracker.get_token_ids_slice(start, end), repeats
            ),
            PACKED_FULL: measure(tracker.packed_token_ids, repeats, setup=rewind),
            PACKED_DELTA: measure(
                lambda: tracker.packed_token_slice(start, end), repeats, setup=rewind
            ),
        },
        wire_bytes={
            # The CPython list's pointer array is the part that scales; the
            # int objects themselves are shared with the source list.
            LIST_FULL: 8 * len(token_ids),
            LIST_DELTA: 8 * (end - start),
            PACKED_FULL: 4 * len(token_ids),
            PACKED_DELTA: 4 * (end - start),
        },
    )


####
# Stages B1/B2 -- scheduler -> worker, pickled into vLLM's shm ring
####


def bench_stage_b(
    token_ids: list[int], start: int, end: int, repeats: int, num_requests: int
) -> tuple[StageResult, StageResult]:
    """Measure both halves of the pickle hop vLLM's ``MessageQueue`` performs.

    ``shm_broadcast.MessageQueue.enqueue`` pickles ``SchedulerOutput`` (with
    the connector metadata inside it) at ``pickle.HIGHEST_PROTOCOL``; each
    worker then unpickles its own copy out of the ring, so the two halves
    have different multipliers.

    Args:
        token_ids: The request's full token id list.
        start: Store op start index.
        end: Store op end index.
        repeats: Number of timed calls.
        num_requests: Concurrent requests sharing the step.

    Returns:
        The dumps-side and loads-side results.
    """
    block_ids = build_block_ids(start, end)
    metas = {
        LIST_FULL: make_connector_metadata(
            _ListLoadStoreOp(
                token_ids=token_ids, block_ids=block_ids, start=start, end=end
            ),
            num_requests,
        ),
        LIST_DELTA: make_connector_metadata(
            _ListLoadStoreOp(
                token_ids=token_ids[start:end],
                block_ids=block_ids,
                start=start,
                end=end,
            ),
            num_requests,
        ),
        PACKED_FULL: make_connector_metadata(
            LoadStoreOp(
                token_bytes=pack_token_ids(token_ids),
                block_ids=block_ids,
                start=start,
                end=end,
            ),
            num_requests,
        ),
        PACKED_DELTA: make_connector_metadata(
            LoadStoreOp(
                token_bytes=pack_token_ids(token_ids[start:end]),
                block_ids=block_ids,
                start=start,
                end=end,
                token_offset=start,
            ),
            num_requests,
        ),
    }
    blobs = {
        name: pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
        for name, meta in metas.items()
    }

    def dump(meta: LMCacheMPConnectorMetadata) -> Callable[[], object]:
        return lambda: pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)

    def load(blob: bytes) -> Callable[[], object]:
        return lambda: pickle.loads(blob)

    dumps = StageResult(
        stage=STAGE_B1,
        context_len=len(token_ids),
        timings={n: measure(dump(metas[n]), repeats) for n in VARIANTS},
        wire_bytes={n: len(blobs[n]) for n in VARIANTS},
    )
    loads = StageResult(
        stage=STAGE_B2,
        context_len=len(token_ids),
        timings={n: measure(load(blobs[n]), repeats) for n in VARIANTS},
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
        identity, kind, payload = socket.recv_multipart()
        if kind == b"stop":
            socket.send_multipart([identity, b"stopped"])
            break
        if kind == b"packed":
            size = msgspec.msgpack.decode(payload, type=IPCCacheServerKey).num_tokens
        else:
            size = len(
                msgspec.msgpack.decode(payload, type=_ListIPCCacheServerKey).token_ids
            )
        socket.send_multipart([identity, str(size).encode()])
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

    def roundtrip(self, kind: bytes, payload: bytes) -> None:
        """Send an encoded key and wait for the server's decode to ack."""
        self._socket.send_multipart([kind, payload])
        self._socket.recv()

    def close(self) -> None:
        """Stop the child process and release both ZMQ contexts."""
        try:
            self._socket.send_multipart([b"stop", b""])
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

    Key construction is inside C1's timed region because ``_create_key``
    performs it on every submit.

    Args:
        harness: The connected DEALER/ROUTER pair.
        token_ids: The request's full token id list.
        start: Store op start index.
        end: Store op end index.
        repeats: Number of timed calls.

    Returns:
        The encode-side, decode-side and end-to-end results.
    """
    full_packed = pack_token_ids(token_ids)
    delta_packed = pack_token_ids(token_ids[start:end])

    builders: dict[str, Callable[[], bytes]] = {
        LIST_FULL: lambda: msgspec.msgpack.encode(
            make_list_ipc_key(token_ids, start, end)
        ),
        LIST_DELTA: lambda: msgspec.msgpack.encode(
            make_list_ipc_key(token_ids[start:end], start, end)
        ),
        PACKED_FULL: lambda: msgspec.msgpack.encode(
            make_ipc_key(full_packed, start, end)
        ),
        PACKED_DELTA: lambda: msgspec.msgpack.encode(
            make_ipc_key(delta_packed, start, end, token_offset=start)
        ),
    }
    blobs = {name: build() for name, build in builders.items()}
    kinds = {
        LIST_FULL: b"list",
        LIST_DELTA: b"list",
        PACKED_FULL: b"packed",
        PACKED_DELTA: b"packed",
    }

    def decode(name: str) -> Callable[[], object]:
        blob = blobs[name]
        if name in (LIST_FULL, LIST_DELTA):
            return lambda: msgspec.msgpack.decode(blob, type=_ListIPCCacheServerKey)
        return lambda: msgspec.msgpack.decode(blob, type=IPCCacheServerKey)

    def trip(name: str) -> Callable[[], object]:
        return lambda: harness.roundtrip(kinds[name], blobs[name])

    encode = StageResult(
        stage=STAGE_C1,
        context_len=len(token_ids),
        timings={n: measure(builders[n], repeats) for n in VARIANTS},
        wire_bytes={n: len(blobs[n]) for n in VARIANTS},
    )
    decoded = StageResult(
        stage=STAGE_C2,
        context_len=len(token_ids),
        timings={n: measure(decode(n), repeats) for n in VARIANTS},
    )
    roundtrip = StageResult(
        stage=STAGE_Z,
        context_len=len(token_ids),
        timings={n: measure(trip(n), repeats) for n in VARIANTS},
    )
    return encode, decoded, roundtrip


####
# Stages D/E -- server-side token handling
####


def bench_stage_d(
    token_ids: list[int], start: int, end: int, repeats: int
) -> StageResult:
    """Measure the server's per-store session token handling.

    ``EngineContext.resolve_obj_keys`` splices the key's tokens into the
    request's session and then asks it for ``[start, end)``'s chunk hashes.
    Hashing is memoized across a request's stores, so a warm session pays
    only for new chunks; what a whole-sequence key still pays on every
    store, by every rank, is the copy of the whole context out of the
    decoded key.

    Args:
        token_ids: The request's full token id list.
        start: Store op start index.
        end: Store op end index.
        repeats: Number of timed calls.

    Returns:
        StageResult: Session update cost per variant.
    """
    hasher = TokenHasher(chunk_size=LMCACHE_CHUNK_SIZE)
    full_packed = pack_token_ids(token_ids)
    delta_packed = pack_token_ids(token_ids[start:end])
    token_tuple = tuple(token_ids)
    delta_tuple = tuple(token_ids[start:end])

    def warm() -> Session:
        session = Session(request_id="req-0", hasher=hasher)
        session.set_tokens(full_packed)
        session.get_hashes(start, end)
        return session

    packed_full_session = warm()
    packed_delta_session = warm()

    def packed_full_store() -> None:
        packed_full_session.absorb_tokens(0, full_packed)
        packed_full_session.get_hashes(start, end)

    def packed_delta_store() -> None:
        packed_delta_session.absorb_tokens(start, delta_packed)
        packed_delta_session.get_hashes(start, end)

    return StageResult(
        stage=STAGE_D,
        context_len=len(token_ids),
        timings={
            # The pre-change path did ``set_tokens(list(key.token_ids))``
            # then memoized hashing, so a warm session's cost was the
            # whole-context list copy out of the decoded tuple.
            LIST_FULL: measure(lambda: list(token_tuple), repeats),
            LIST_DELTA: measure(lambda: list(delta_tuple), repeats),
            PACKED_FULL: measure(packed_full_store, repeats),
            PACKED_DELTA: measure(packed_delta_store, repeats),
        },
    )


def bench_stage_e(token_ids: list[int], repeats: int) -> StageResult:
    """Measure LOOKUP's uncached whole-sequence chunk hash.

    ``PrefetchLookupHandler`` hashes the request's whole sequence outside
    the session memoization. LOOKUP is what seeds that session, so it always
    carries every token: ``delta_token_ids`` cannot shrink it and both
    packed columns are the same measurement. Packing still helps, because
    blake3 can take the buffer as it arrives.

    Args:
        token_ids: The request's full token id list.
        repeats: Number of timed calls.

    Returns:
        StageResult: Whole-sequence hashing cost per variant.
    """
    hasher = TokenHasher(chunk_size=LMCACHE_CHUNK_SIZE)
    packed = pack_token_ids(token_ids)
    packed_timing = measure(lambda: hasher.compute_packed_chunk_hashes(packed), repeats)
    list_timing = measure(lambda: hasher.compute_chunk_hashes(list(token_ids)), repeats)
    return StageResult(
        stage=STAGE_E,
        context_len=len(token_ids),
        timings={
            LIST_FULL: list_timing,
            LIST_DELTA: list_timing,
            PACKED_FULL: packed_timing,
            PACKED_DELTA: packed_timing,
        },
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
    print(f"\n{results[0].stage}  (microseconds per op)")
    print(
        f"  {'ctx':>8} {LIST_FULL:>10} {LIST_DELTA:>11} {PACKED_FULL:>12} "
        f"{PACKED_DELTA:>13} {'speedup':>8} {'wire':>10}"
    )
    for result in results:
        print(
            f"  {result.context_len:>8,} "
            f"{result.median_us(LIST_FULL):>10.1f} "
            f"{result.median_us(LIST_DELTA):>11.1f} "
            f"{result.median_us(PACKED_FULL):>12.1f} "
            f"{result.median_us(PACKED_DELTA):>13.1f} "
            f"{result.speedup(PACKED_DELTA):>7.1f}x "
            f"{format_bytes(result.bytes_for(PACKED_DELTA)):>10}"
        )


def print_request_rollup(
    by_stage: dict[str, dict[int, StageResult]],
    context_len: int,
    tensor_parallel_size: int,
) -> None:
    """Project the per-op numbers onto one full ``context_len`` prefill.

    A chunked prefill takes ``ceil(context_len / PREFILL_CHUNK_TOKENS)``
    scheduler steps, and each step emits one store op. Scheduler-side stages
    run once per step; worker- and server-side stages run once per step per
    TP rank, because every rank submits its own store. LOOKUP runs once.

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
        f"  {'stage':<24} {'x':>5} {LIST_FULL:>10} {LIST_DELTA:>11} "
        f"{PACKED_FULL:>12} {PACKED_DELTA:>13}"
    )

    totals = dict.fromkeys(VARIANTS, 0.0)
    for stage, multiplier in multipliers.items():
        result = by_stage[stage][context_len]
        row = {n: multiplier * result.median_us(n) / 1_000.0 for n in VARIANTS}
        for name in VARIANTS:
            totals[name] += row[name]
        note = "  (unchanged by delta)" if stage == STAGE_E else ""
        print(
            f"  {stage:<24} {multiplier:>5} {row[LIST_FULL]:>10.1f} "
            f"{row[LIST_DELTA]:>11.1f} {row[PACKED_FULL]:>12.1f} "
            f"{row[PACKED_DELTA]:>13.1f}{note}"
        )
    print(
        f"  {'TOTAL CPU (ms)':<24} {'':>5} {totals[LIST_FULL]:>10.1f} "
        f"{totals[LIST_DELTA]:>11.1f} {totals[PACKED_FULL]:>12.1f} "
        f"{totals[PACKED_DELTA]:>13.1f}"
    )

    wire = {
        name: sum(
            multipliers[stage] * by_stage[stage][context_len].bytes_for(name)
            for stage in WIRE_STAGES
        )
        for name in VARIANTS
    }
    print(
        f"  {'bytes across processes':<24} {'':>5} "
        f"{format_bytes(wire[LIST_FULL]):>10} "
        f"{format_bytes(wire[LIST_DELTA]):>11} "
        f"{format_bytes(wire[PACKED_FULL]):>12} "
        f"{format_bytes(wire[PACKED_DELTA]):>13}"
    )
    # Each axis on its own, against the same baseline, plus the two
    # together -- they do not simply add, since whichever lands first
    # takes the bulk of a stage's cost with it.
    print(
        f"  -> delta alone saves {totals[LIST_FULL] - totals[LIST_DELTA]:.1f} ms, "
        f"packing alone {totals[LIST_FULL] - totals[PACKED_FULL]:.1f} ms, "
        f"both {totals[LIST_FULL] - totals[PACKED_DELTA]:.1f} ms"
    )

    rtt = by_stage[STAGE_Z][context_len]
    decode = by_stage[STAGE_C2][context_len]
    print(
        f"  cross-check: ZMQ round trip {rtt.median_us(LIST_FULL):.0f} us vs "
        f"in-process decode {decode.median_us(LIST_FULL):.0f} us -- transport "
        f"is minor next to decoding the token list"
    )


####
# Driver
####


def run(context_lengths: Sequence[int], repeats: int, num_requests: int) -> None:
    """Measure every stage at every context length and print the report.

    Args:
        context_lengths: Prompt lengths to sweep.
        repeats: Timed calls per measurement.
        num_requests: Concurrent requests sharing a scheduler step.
    """
    by_stage: dict[str, dict[int, StageResult]] = {}
    harness = ZmqHarness()
    try:
        for context_len in context_lengths:
            token_ids = make_token_ids(context_len)
            start, end = op_range(context_len)
            results = [
                bench_stage_a(token_ids, start, end, repeats),
                *bench_stage_b(token_ids, start, end, repeats, num_requests),
                *bench_stage_c(harness, token_ids, start, end, repeats),
                bench_stage_d(token_ids, start, end, repeats),
                bench_stage_e(token_ids, repeats),
            ]
            for result in results:
                by_stage.setdefault(result.stage, {})[context_len] = result
    finally:
        harness.close()

    print(
        f"\nOne store op per scheduler step; {num_requests} request(s) per step, "
        f"{repeats} repeats."
    )
    print("'speedup' is packed+delta over list+full; 'wire' is packed+delta's bytes.")
    for stage in (
        STAGE_A,
        STAGE_B1,
        STAGE_B2,
        STAGE_C1,
        STAGE_C2,
        STAGE_Z,
        STAGE_D,
        STAGE_E,
    ):
        print_stage_table([by_stage[stage][c] for c in context_lengths])

    print_request_rollup(by_stage, max(context_lengths), tensor_parallel_size=8)


def main() -> None:
    """Parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Sweep three context lengths with fewer repeats.",
    )
    parser.add_argument(
        "--repeats", type=int, default=50, help="Timed calls per measurement."
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=1,
        help="Concurrent requests sharing one scheduler step.",
    )
    args = parser.parse_args()
    run(
        QUICK_CONTEXT_LENGTHS if args.quick else CONTEXT_LENGTHS,
        repeats=10 if args.quick else args.repeats,
        num_requests=args.requests,
    )


if __name__ == "__main__":
    main()
