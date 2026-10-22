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
import torch

# First Party
from lmcache.integration.vllm.l1_duplication import (
    MAX_MISSED_DEADLINES,
    SAMPLE_INTERVAL_S,
    DuplicationSample,
    L1DuplicationMeter,
)
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
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
    with one, like vLLM's hybrid fixed point. It is also its own block pool:
    a cached block has ``ref_cnt`` 1 if its hash is in ``in_use``, else 0.
    """

    def __init__(
        self, attention: bool = True, mamba: bool = False, eagle: bool = False
    ) -> None:
        self.attention = attention
        self.mamba = mamba
        self.eagle = eagle
        self.attention_blocks: set[Any] = set()
        self.mamba_states: set[Any] = set()
        self.in_use: set[Any] = set()
        self.calls = 0
        specs: list[Any] = []
        if attention:
            specs.append(FullAttentionSpec())
        if mamba:
            specs.append(MambaSpec())
        self.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs]
        )
        self.single_type_managers = [SimpleNamespace(block_pool=self) for _ in specs]

    def get_cached_block(
        self, block_hash: Any, kv_cache_group_ids: list[int]
    ) -> list[Any] | None:
        (group_id,) = kv_cache_group_ids
        spec = self.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
        cached = (
            self.mamba_states if isinstance(spec, MambaSpec) else self.attention_blocks
        )
        if block_hash not in cached:
            return None
        return [SimpleNamespace(ref_cnt=int(block_hash in self.in_use))]

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


# Stand-ins for vLLM's KV cache specs; the meter tells them apart by class
# name, as for real specs.
class FullAttentionSpec:
    def __init__(self) -> None:
        self.block_size = HASH_BLOCK


class SlidingWindowSpec:
    def __init__(self, sliding_window: int) -> None:
        self.block_size = HASH_BLOCK
        self.sliding_window = sliding_window


class MambaSpec:
    def __init__(self) -> None:
        self.block_size = CHUNK
        self.mamba_cache_mode = "align"


class _BlockPool:
    """vLLM's block pool over cached (block hash, KV cache group id) pairs.

    A cached block has ``ref_cnt`` 1 if its pair is in ``in_use``, else 0.
    """

    def __init__(self) -> None:
        self.cached: set[tuple[Any, int]] = set()
        self.in_use: set[tuple[Any, int]] = set()
        self.enable_caching = True

    def get_cached_block(
        self, block_hash: Any, kv_cache_group_ids: list[int]
    ) -> list[Any] | None:
        if all((block_hash, g) in self.cached for g in kv_cache_group_ids):
            return [
                SimpleNamespace(ref_cnt=int((block_hash, g) in self.in_use))
                for g in kv_cache_group_ids
            ]
        return None


class _GroupCoordinator:
    """vLLM's hybrid coordinator over KV cache groups of the given specs.

    A full-attention group's own hit is its longest run of cached blocks
    from the start; the whole hit is the shortest of those.
    """

    def __init__(self, *specs: Any) -> None:
        self.block_pool = _BlockPool()
        self.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs]
        )
        self.single_type_managers = [
            SimpleNamespace(block_pool=self.block_pool) for _ in specs
        ]

    def find_longest_cache_hit_per_group(
        self, block_hashes: list[Any], max_cache_hit_length: int
    ) -> tuple[tuple[list, ...], tuple[int, ...]]:
        hits = []
        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            num_blocks = 0
            if isinstance(group.kv_cache_spec, FullAttentionSpec):
                while (
                    num_blocks < len(block_hashes)
                    and (block_hashes[num_blocks], group_id) in self.block_pool.cached
                ):
                    num_blocks += 1
            hits.append(num_blocks * HASH_BLOCK)
        return tuple([] for _ in hits), tuple(hits)

    def find_longest_cache_hit(
        self, block_hashes: list[Any], max_cache_hit_length: int
    ) -> tuple[list, int, int]:
        _, hits = self.find_longest_cache_hit_per_group(
            block_hashes, max_cache_hit_length
        )
        full = [
            hit
            for hit, group in zip(
                hits, self.kv_cache_config.kv_cache_groups, strict=True
            )
            if isinstance(group.kv_cache_spec, FullAttentionSpec)
        ]
        return [], min(full), 0


# The object groups of a server that separates them for an attention +
# Mamba model: full attention (id 0) and one-chunk recurrent state (id 1).
HYBRID = AttnWindowDesc(
    num_chunks_in_sw=[-1, 1], group_kinds=("attention", "recurrent")
)


class _Adapter:
    model_name = "m"

    def __init__(
        self,
        store: StorageManager,
        world_size: int = 1,
        hasher: TokenHasher | None = None,
        response: _Response = "module",
        attn_desc: AttnWindowDesc | None = None,
    ) -> None:
        # attn_desc: the object groups of a server that separates them,
        # registered for this model and world size; None for one that merges.
        registry = LayoutDescRegistry()
        if attn_desc is not None:
            layout = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
            registry.register(self.model_name, world_size, layout, attn_desc=attn_desc)
        ctx = SimpleNamespace(
            storage_manager=store,
            token_hasher=hasher or TokenHasher(chunk_size=CHUNK),
            separate_object_groups=attn_desc is not None,
            full_sw_kv=False,
            layout_desc_registry=registry,
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
    adapter: _Adapter, coordinator: "_Coordinator | _GroupCoordinator"
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
            matched_inuse_objects=0,
            matched_inuse_bytes=0,
        )
    ]
    assert samples[0].byte_ratio == pytest.approx(0.5)
    assert samples[0].object_ratio == pytest.approx(0.5)


def test_held_chunks_split_into_in_use_and_free_cached(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, *(object_key(c) for c in CHUNK_HASHES))
    coordinator = _Coordinator()
    meter, clock, samples = _meter(_Adapter(store), coordinator)
    meter.on_request_tracked(*_request("r", TOKENS))
    _cache(coordinator, TOKENS)
    # A running request holds the whole of chunk 0 and one block of chunk 1;
    # chunk 2's blocks sit only in the free queue.
    hashes = _block_hashes(TOKENS)
    coordinator.in_use.update({hashes[0], hashes[1], hashes[2]})

    _sample(meter, clock)

    assert samples[-1] == DuplicationSample(
        matched_objects=3,
        matched_bytes=3 * OBJECT_BYTES,
        model_objects=3,
        model_bytes=3 * OBJECT_BYTES,
        matched_inuse_objects=2,
        matched_inuse_bytes=2 * OBJECT_BYTES,
    )

    # Once every block is freed (ref_cnt 0, still cached), nothing is in use.
    coordinator.in_use.clear()
    _sample(meter, clock)

    assert samples[-1].matched_objects == 3
    assert samples[-1].matched_inuse_objects == 0
    assert samples[-1].matched_inuse_bytes == 0


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


def _cache_blocks(
    coordinator: _GroupCoordinator, group_id: int, blocks: list[int]
) -> None:
    """Cache group ``group_id``'s blocks of ``TOKENS``, by hash block index.

    Hash block ``2 * i + 1`` ends chunk ``i``; a Mamba state there is the
    state at the chunk's end.
    """
    hashes = _block_hashes(TOKENS)
    coordinator.block_pool.cached.update((hashes[b], group_id) for b in blocks)


def _uncache_blocks(
    coordinator: _GroupCoordinator, group_id: int, blocks: list[int]
) -> None:
    """Evict group ``group_id``'s blocks of ``TOKENS``, by hash block index."""
    hashes = _block_hashes(TOKENS)
    coordinator.block_pool.cached.difference_update(
        (hashes[b], group_id) for b in blocks
    )


def _commit_all_chunks(l1: L1Manager, *object_group_ids: int) -> None:
    """Commit every chunk of ``TOKENS`` in each of ``object_group_ids``."""
    commit(
        l1,
        *(
            object_key(c, object_group_id=g)
            for c in CHUNK_HASHES
            for g in object_group_ids
        ),
    )


def test_separated_object_groups_hold_chunks_per_kv_cache_group(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    _commit_all_chunks(l1, 0, 1)
    coordinator = _GroupCoordinator(FullAttentionSpec(), MambaSpec())
    adapter = _Adapter(store, attn_desc=HYBRID)
    meter, clock, samples = _meter(adapter, coordinator)
    meter.on_request_tracked(*_request("r", TOKENS))
    # Attention holds all three chunks; Mamba keeps states at the ends of
    # chunks 0 and 2 only.
    _cache_blocks(coordinator, 0, list(range(6)))
    _cache_blocks(coordinator, 1, [1, 5])

    # The first reply shows two object groups: that sample is dropped.
    _sample(meter, clock)
    _sample(meter, clock)

    assert adapter.calls == 2
    assert samples == [
        DuplicationSample(
            matched_objects=5,
            matched_bytes=5 * OBJECT_BYTES,
            model_objects=6,
            model_bytes=6 * OBJECT_BYTES,
            matched_inuse_objects=0,
            matched_inuse_bytes=0,
        )
    ]


def test_separated_object_groups_split_in_use_by_their_own_blocks(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    _commit_all_chunks(l1, 0, 1)
    coordinator = _GroupCoordinator(FullAttentionSpec(), MambaSpec())
    meter, clock, samples = _meter(_Adapter(store, attn_desc=HYBRID), coordinator)
    meter.on_request_tracked(*_request("r", TOKENS))
    _cache_blocks(coordinator, 0, list(range(6)))
    _cache_blocks(coordinator, 1, [1, 5])
    # A running request holds chunk 2's attention blocks and the Mamba state
    # at chunk 0's end; the other held objects are free-cached.
    hashes = _block_hashes(TOKENS)
    coordinator.block_pool.in_use.update(
        {(hashes[4], 0), (hashes[5], 0), (hashes[1], 1)}
    )

    _sample(meter, clock)
    _sample(meter, clock)

    assert (samples[-1].matched_objects, samples[-1].matched_inuse_objects) == (5, 2)
    assert samples[-1].matched_inuse_bytes == 2 * OBJECT_BYTES


def test_object_group_of_several_kv_cache_groups_needs_all_of_them(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    _commit_all_chunks(l1, 0)
    coordinator = _GroupCoordinator(
        FullAttentionSpec(), FullAttentionSpec(), MambaSpec()
    )
    meter, clock, samples = _meter(_Adapter(store, attn_desc=HYBRID), coordinator)
    meter.on_request_tracked(*_request("r", TOKENS))
    # Both attention groups store into object group 0; the second holds
    # chunk 0 only.
    _cache_blocks(coordinator, 0, list(range(6)))
    _cache_blocks(coordinator, 1, [0, 1])

    _sample(meter, clock)
    _sample(meter, clock)

    assert _matched(samples) == 1


def test_sliding_window_chunk_needs_the_window_ending_at_it(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    _commit_all_chunks(l1, 1)
    # A 4-token window needs vLLM's cdiv(4 - 1, 2) = 2 trailing blocks.
    coordinator = _GroupCoordinator(FullAttentionSpec(), SlidingWindowSpec(4))
    attn_desc = AttnWindowDesc(
        num_chunks_in_sw=[-1, 1], group_kinds=("attention", "attention")
    )
    meter, clock, samples = _meter(_Adapter(store, attn_desc=attn_desc), coordinator)
    meter.on_request_tracked(*_request("r", TOKENS))
    # Chunk 1's window is cached; chunk 2's only in part; chunk 0's not.
    _cache_blocks(coordinator, 1, [2, 3, 5])

    _sample(meter, clock)
    _sample(meter, clock)

    assert _matched(samples) == 1


def test_finished_prefix_keeps_each_object_group_its_held_chunks(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    _commit_all_chunks(l1, 0, 1)
    coordinator = _GroupCoordinator(FullAttentionSpec(), MambaSpec())
    meter, clock, samples = _meter(_Adapter(store, attn_desc=HYBRID), coordinator)
    meter.on_request_tracked(*_request("r", TOKENS))
    # Attention holds chunk 0; Mamba a state at chunk 2's end.
    _cache_blocks(coordinator, 0, [0, 1])
    _cache_blocks(coordinator, 1, [5])
    _sample(meter, clock)
    meter.on_request_finished("r")

    _sample(meter, clock)
    _uncache_blocks(coordinator, 1, [5])
    _sample(meter, clock)
    # A chunk cut from a finished prefix stays cut once it is back.
    _cache_blocks(coordinator, 1, [5])
    _sample(meter, clock)
    _uncache_blocks(coordinator, 0, [0, 1])
    _sample(meter, clock)
    _cache_blocks(coordinator, 0, [0, 1])
    _sample(meter, clock)

    assert [s.matched_objects for s in samples] == [2, 1, 1, 0, 0]


@pytest.mark.parametrize(
    ("attn_desc", "enable_caching"),
    [
        (
            AttnWindowDesc(
                num_chunks_in_sw=[-1, 2], group_kinds=("attention", "recurrent")
            ),
            True,
        ),
        (HYBRID, False),
    ],
    ids=["object_group_missing", "prefix_caching_disabled"],
)
def test_separated_object_groups_stop_measuring(
    storage: tuple[L1Manager, StorageManager],
    attn_desc: AttnWindowDesc,
    enable_caching: bool,
) -> None:
    coordinator = _GroupCoordinator(FullAttentionSpec(), MambaSpec())
    coordinator.block_pool.enable_caching = enable_caching
    adapter = _Adapter(storage[1], attn_desc=attn_desc)
    meter, clock, samples = _meter(adapter, coordinator)

    for _ in range(3):
        _sample(meter, clock)

    assert adapter.calls == 1
    assert samples == []
