# SPDX-License-Identifier: Apache-2.0
"""Tests for the scheduler-side L0/L1 duplication meter.

The meter decides what L0 holds through a fake vLLM coordinator and asks a
real ``l1_residency`` module over a real CPU L1 what L1 commits.
"""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Literal, cast

# Third Party
import grpc
import pytest

# First Party
from lmcache.integration.vllm.l1_duplication import (
    MAX_MISSED_DEADLINES,
    SAMPLE_INTERVAL_S,
    DuplicationSample,
    L1DuplicationMeter,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.l1_residency import L1ResidencyModule
from lmcache.v1.multiprocess.protocols.l1_residency import L1_RESIDENCY_QUERY_METHOD
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallResponse,
    unknown_method_error,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from tests.v1.distributed.utils import (
    OBJECT_BYTES,
    commit,
    cpu_storage,
    object_key,
)

pytestmark = pytest.mark.no_shared_allocator

CHUNK = 4
HASH_BLOCK = 2
URL = "tcp://server"
TOKENS = list(range(12))  # three chunks
CHUNK_HASHES = TokenHasher(chunk_size=CHUNK).compute_chunk_hashes(TOKENS)

# What _Adapter answers: "module" for the real L1ResidencyModule's reply, a
# response, a transport exception stored on the future, or None for a future
# that never completes.
_Response = Literal["module"] | ServerModuleCallResponse | Exception | None


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _RpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode) -> None:
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


class _Coordinator:
    """vLLM's coordinator over a set of GPU-cached block hashes.

    Attention hits the longest run of cached blocks from the start; with
    EAGLE it then drops the last hit block. Mamba then needs a cached state
    at the hit's last block, else it falls back to the last earlier block
    with one, like vLLM's hybrid fixed point.
    """

    def __init__(
        self, attention: bool = True, mamba: bool = False, eagle: bool = False
    ) -> None:
        self.attention = attention
        self.mamba = mamba
        self.eagle = eagle
        self.attention_blocks: set[Any] = set()
        self.mamba_states: set[Any] = set()
        self.calls = 0

    def find_longest_cache_hit(
        self, block_hashes: list[Any], max_cache_hit_length: int
    ) -> tuple[list, int, int]:
        self.calls += 1
        num_blocks = min(len(block_hashes), max_cache_hit_length // HASH_BLOCK)
        if self.attention:
            for index in range(num_blocks):
                if block_hashes[index] not in self.attention_blocks:
                    num_blocks = index
                    break
            if self.eagle and num_blocks:
                num_blocks -= 1
        if self.mamba:
            while num_blocks and block_hashes[num_blocks - 1] not in self.mamba_states:
                num_blocks -= 1
        return [], num_blocks * HASH_BLOCK, 0


class _Adapter:
    model_name = "m"

    def __init__(
        self,
        store: StorageManager,
        world_size: int = 1,
        hasher: TokenHasher | None = None,
        response: _Response = "module",
    ) -> None:
        ctx = SimpleNamespace(
            storage_manager=store,
            token_hasher=hasher or TokenHasher(chunk_size=CHUNK),
            separate_object_groups=False,
        )
        self.module = L1ResidencyModule(cast(Any, ctx))
        self.world_size = world_size
        self.response = response
        self.calls = 0
        self.is_healthy = True

    def server_module_call(
        self, method: str, payload: bytes
    ) -> dict[str, MessagingFuture[ServerModuleCallResponse]]:
        self.calls += 1
        future: MessagingFuture[ServerModuleCallResponse] = MessagingFuture()
        response = self.response
        if response == "module":
            response = ServerModuleCallResponse(
                success=True, payload=self.module.query(payload), error=""
            )
        if isinstance(response, Exception):
            future.set_exception(response)
        elif isinstance(response, ServerModuleCallResponse):
            future.set_result(response)
        return {URL: future}


@pytest.fixture
def storage(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[L1Manager, StorageManager]]:
    l1, store = cpu_storage(monkeypatch)
    yield l1, store
    store.close()


def _block_hashes(tokens: list[int]) -> list[Any]:
    return [
        tuple(tokens[: end + HASH_BLOCK])
        for end in range(0, len(tokens) - HASH_BLOCK + 1, HASH_BLOCK)
    ]


def _request(
    request_id: str, tokens: list[int], cache_salt: str = ""
) -> tuple[Any, Any]:
    """A vLLM request and its tracker; both read the live ``tokens``."""
    request = SimpleNamespace(request_id=request_id, block_hashes=_block_hashes(tokens))
    tracker = SimpleNamespace(cache_salt=cache_salt, get_token_ids=lambda: tokens)
    return request, tracker


def _cache(coordinator: _Coordinator, tokens: list[int], start: int = 0) -> None:
    """Put the attention blocks of ``tokens[start:]`` on the GPU."""
    coordinator.attention_blocks.update(_block_hashes(tokens)[start // HASH_BLOCK :])


def _evict(coordinator: _Coordinator, tokens: list[int], start: int = 0) -> None:
    """Free the attention blocks of ``tokens[start:]``."""
    coordinator.attention_blocks.difference_update(
        _block_hashes(tokens)[start // HASH_BLOCK :]
    )


def _meter(
    adapter: _Adapter, coordinator: _Coordinator
) -> tuple[L1DuplicationMeter, _Clock, list[DuplicationSample]]:
    clock = _Clock()
    samples: list[DuplicationSample] = []
    meter = L1DuplicationMeter(
        cast(Any, adapter),
        cast(Any, coordinator),
        hash_block_size=HASH_BLOCK,
        chunk_size=CHUNK,
        hash_algorithm="blake3",
        on_sample=samples.append,
        clock=clock,
    )
    return meter, clock, samples


def _sample(meter: L1DuplicationMeter, clock: _Clock) -> None:
    """Run the step that sends a query and the step that reads its answer."""
    clock.now += SAMPLE_INTERVAL_S
    meter.on_scheduler_step()
    meter.on_scheduler_step()


def _matched(samples: list[DuplicationSample]) -> int:
    return samples[-1].matched_objects


def test_ratio_is_over_committed_l1_of_the_model_for_the_request_salt(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    c0, c1, _ = CHUNK_HASHES
    commit(l1, object_key(c0, cache_salt="u"))  # in L0 and L1
    commit(l1, object_key(c1))  # L0 holds chunk 1 for salt "u" only
    coordinator = _Coordinator()
    meter, clock, samples = _meter(_Adapter(store), coordinator)
    request, tracker = _request("r", TOKENS, cache_salt="u")
    meter.on_request_tracked(request, tracker)
    _cache(coordinator, TOKENS)

    _sample(meter, clock)

    assert samples == [
        DuplicationSample(
            matched_objects=1,
            matched_bytes=OBJECT_BYTES,
            model_objects=2,
            model_bytes=2 * OBJECT_BYTES,
        )
    ]
    assert samples[0].byte_ratio == pytest.approx(0.5)
    assert samples[0].object_ratio == pytest.approx(0.5)


def test_mamba_hit_holds_every_chunk_up_to_its_latest_state(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, *(object_key(c) for c in CHUNK_HASHES))
    coordinator = _Coordinator(attention=False, mamba=True)
    meter, clock, samples = _meter(_Adapter(store), coordinator)
    meter.on_request_tracked(*_request("r", TOKENS))
    # Mamba keeps only the latest state; earlier ones were freed.
    coordinator.mamba_states.add(_block_hashes(TOKENS)[-1])

    _sample(meter, clock)

    assert _matched(samples) == 3


def test_hybrid_gap_holds_only_the_prefix_vllm_hits(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, *(object_key(c) for c in CHUNK_HASHES))
    coordinator = _Coordinator(mamba=True)
    meter, clock, samples = _meter(_Adapter(store), coordinator)
    meter.on_request_tracked(*_request("r", TOKENS))
    # Attention holds all three chunks; Mamba has a state at chunk 0's end
    # only, so vLLM hits chunk 0 and recomputes the rest.
    _cache(coordinator, TOKENS)
    coordinator.mamba_states.add(_block_hashes(TOKENS)[CHUNK // HASH_BLOCK - 1])

    _sample(meter, clock)

    assert _matched(samples) == 1


def test_running_prefix_follows_new_blocks(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, *(object_key(c) for c in CHUNK_HASHES))
    coordinator = _Coordinator()
    meter, clock, samples = _meter(_Adapter(store), coordinator)
    tokens = TOKENS[: 2 * CHUNK]
    request, tracker = _request("r", tokens)
    meter.on_request_tracked(request, tracker)
    _cache(coordinator, tokens)
    _sample(meter, clock)

    tokens.extend(TOKENS[2 * CHUNK :])
    request.block_hashes.extend(_block_hashes(tokens)[len(request.block_hashes) :])
    _cache(coordinator, tokens)
    _sample(meter, clock)

    assert [s.matched_objects for s in samples] == [2, 3]


def test_finished_prefix_is_cut_to_what_l0_holds_then_dropped(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, *(object_key(c) for c in CHUNK_HASHES))
    coordinator = _Coordinator()
    meter, clock, samples = _meter(_Adapter(store), coordinator)
    meter.on_request_tracked(*_request("r", TOKENS))
    _cache(coordinator, TOKENS)
    meter.on_request_finished("r")
    _sample(meter, clock)

    _evict(coordinator, TOKENS, start=CHUNK)
    _sample(meter, clock)
    # A chunk cut from a finished prefix stays cut once it is back.
    _cache(coordinator, TOKENS)
    _sample(meter, clock)
    _evict(coordinator, TOKENS)
    _sample(meter, clock)
    _cache(coordinator, TOKENS)
    _sample(meter, clock)

    assert [s.matched_objects for s in samples] == [3, 1, 1, 0, 0]


@pytest.mark.parametrize(
    "coordinator_kwargs",
    [{"eagle": True}, {"attention": False, "mamba": True}],
    ids=["eagle", "mamba"],
)
def test_finished_prefix_keeps_what_vllm_still_hits_from_its_prompt(
    storage: tuple[L1Manager, StorageManager], coordinator_kwargs: dict[str, bool]
) -> None:
    l1, store = storage
    commit(l1, *(object_key(c) for c in CHUNK_HASHES))
    coordinator = _Coordinator(**coordinator_kwargs)
    meter, clock, samples = _meter(_Adapter(store), coordinator)
    # Two tokens past the last chunk: vLLM hits all three chunks from this
    # prompt, but EAGLE would hit fewer from a list cut at chunk 2's end, and
    # Mamba nothing, since its only state sits past the cut.
    tokens = [*TOKENS, 12, 13]
    _cache(coordinator, tokens)
    coordinator.mamba_states.add(_block_hashes(tokens)[-1])
    meter.on_request_tracked(*_request("r", tokens))
    meter.on_request_finished("r")

    for _ in range(3):
        _sample(meter, clock)

    assert [s.matched_objects for s in samples] == [3, 3, 3]


def test_finished_prefixes_ending_at_one_chunk_keep_the_later_one(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, *(object_key(c) for c in CHUNK_HASHES))
    coordinator = _Coordinator(eagle=True)
    meter, clock, samples = _meter(_Adapter(store), coordinator)
    # Same three chunks, then one more block each; EAGLE needs that block
    # to hit all three.
    older, newer = [*TOKENS, 12, 13], [*TOKENS, 14, 15]
    _cache(coordinator, older)
    _cache(coordinator, newer)
    for request_id, tokens in (("older", older), ("newer", newer)):
        meter.on_request_tracked(*_request(request_id, tokens))
        meter.on_request_finished(request_id)
    # vLLM evicts the older request's freed block first.
    _evict(coordinator, older, start=len(TOKENS))

    _sample(meter, clock)

    assert _matched(samples) == 3


def test_unhealthy_servers_skip_the_query_but_not_the_pruning(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, *(object_key(c) for c in CHUNK_HASHES))
    coordinator = _Coordinator()
    adapter = _Adapter(store)
    meter, clock, samples = _meter(adapter, coordinator)
    meter.on_request_tracked(*_request("r", TOKENS))
    _cache(coordinator, TOKENS)
    meter.on_request_finished("r")

    adapter.is_healthy = False
    _evict(coordinator, TOKENS)
    _sample(meter, clock)
    assert adapter.calls == 0

    adapter.is_healthy = True
    _cache(coordinator, TOKENS)
    _sample(meter, clock)
    assert _matched(samples) == 0


def test_earlier_turn_is_not_counted_twice_or_looked_up(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, *(object_key(c) for c in CHUNK_HASHES))
    coordinator = _Coordinator()
    meter, clock, samples = _meter(_Adapter(store), coordinator)
    turn = TOKENS[: 2 * CHUNK]
    meter.on_request_tracked(*_request("turn-1", turn))
    _cache(coordinator, TOKENS)
    meter.on_request_finished("turn-1")
    meter.on_request_tracked(*_request("turn-2", TOKENS))

    calls = coordinator.calls
    _sample(meter, clock)

    assert _matched(samples) == 3
    assert coordinator.calls == calls + 1


def _unknown_method() -> ServerModuleCallResponse:
    return ServerModuleCallResponse(
        success=False,
        payload=b"",
        error=unknown_method_error(L1_RESIDENCY_QUERY_METHOD),
    )


def _reseeded() -> TokenHasher:
    hasher = TokenHasher(chunk_size=CHUNK)
    hasher.none_hash = b"another PYTHONHASHSEED"
    return hasher


@pytest.mark.parametrize(
    ("response", "server_hasher", "calls"),
    [
        (_unknown_method(), None, 1),
        ("module", TokenHasher(chunk_size=CHUNK, hash_algorithm="sha256_cbor"), 1),
        ("module", _reseeded(), 1),
        (
            ServerModuleCallResponse(success=True, payload=b"\xc1", error=""),
            None,
            1,
        ),
        (_RpcError(grpc.StatusCode.UNIMPLEMENTED), None, 1),
        (None, None, MAX_MISSED_DEADLINES),
    ],
    ids=[
        "unknown_method",
        "other_algorithm",
        "other_seed",
        "undecodable_reply",
        "grpc_unimplemented",
        "missed_deadlines",
    ],
)
def test_stops_measuring(
    storage: tuple[L1Manager, StorageManager],
    response: _Response,
    server_hasher: TokenHasher | None,
    calls: int,
) -> None:
    adapter = _Adapter(storage[1], hasher=server_hasher, response=response)
    meter, clock, samples = _meter(adapter, _Coordinator())

    for _ in range(MAX_MISSED_DEADLINES + 1):
        _sample(meter, clock)
    adapter.response = "module"
    _sample(meter, clock)

    assert adapter.calls == calls
    assert samples == []


@pytest.mark.parametrize(
    "response",
    [
        ServerModuleCallResponse(success=False, payload=b"", error="boom"),
        RuntimeError("send failed"),
        _RpcError(grpc.StatusCode.UNAVAILABLE),
        None,
    ],
    ids=["handler_failure", "transport_error", "grpc_unavailable", "one_deadline"],
)
def test_drops_one_sample_and_retries(
    storage: tuple[L1Manager, StorageManager], response: _Response
) -> None:
    adapter = _Adapter(storage[1], response=response)
    meter, clock, samples = _meter(adapter, _Coordinator())

    _sample(meter, clock)
    assert samples == []

    adapter.response = "module"
    _sample(meter, clock)
    assert adapter.calls == 2
    assert len(samples) == 1


def test_no_query_before_the_interval(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    adapter = _Adapter(storage[1])
    meter, clock, _ = _meter(adapter, _Coordinator())

    clock.now = SAMPLE_INTERVAL_S - 1
    meter.on_scheduler_step()
    assert adapter.calls == 0
    clock.now = SAMPLE_INTERVAL_S
    meter.on_scheduler_step()
    assert adapter.calls == 1
