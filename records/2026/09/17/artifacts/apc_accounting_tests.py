# SPDX-License-Identifier: Apache-2.0
"""APC-accounting tests cut from the lazy-offload PR as out of scope.

Extracted from tests/v1/lazy_offload/test_mp_connector.py at 9c59bc6e. They
cover eager-mode vLLM prefix-cache accounting in LMCacheMPConnector, not lazy
offload, so they were removed from the PR branch. Kept here for whoever writes
the APC-accounting PR: they need the _make_connector harness from that file.
"""

def test_lookup_miss_still_records_the_vllm_prefix_hit() -> None:
    harness = _make_connector()
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="F",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )

    need_to_load, is_async = harness.connector.get_num_new_matched_tokens(
        request, num_computed_tokens=3 * TOKENS_PER_BLOCK + 4
    )

    assert (need_to_load, is_async) == (0, False)
    tracker = harness.connector.request_trackers["F"]
    assert tracker.num_vllm_hit_tokens == 3 * TOKENS_PER_BLOCK
    assert tracker.num_lmcache_hit_tokens == 0


def test_eager_lookup_miss_backfills_the_apc_prefix_without_duplicates() -> None:
    """The mode-independent APC accounting fixes eager under-store."""
    harness = _make_connector()
    harness.connector.lazy_offload = False
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="E",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )

    need_to_load, is_async = harness.connector.get_num_new_matched_tokens(
        request, num_computed_tokens=3 * TOKENS_PER_BLOCK + 4
    )

    assert (need_to_load, is_async) == (0, False)
    tracker = harness.connector.request_trackers["E"]
    assert tracker.num_vllm_hit_tokens == 3 * TOKENS_PER_BLOCK
    assert tracker.num_lmcache_hit_tokens == 0
    tracker.allocated_block_ids = {0: [1, 2, 3, 4]}
    tracker.num_scheduled_tokens = 4

    backfill = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=TOKENS_PER_BLOCK,
        group_tokens_per_block=[TOKENS_PER_BLOCK],
    )

    assert backfill is not None
    assert (backfill.op.start, backfill.op.end) == (0, 3 * TOKENS_PER_BLOCK)
    assert tracker.num_stored_tokens == 3 * TOKENS_PER_BLOCK

    tracker.increase_num_scheduled_tokens(TOKENS_PER_BLOCK - 4)
    next_chunk = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=TOKENS_PER_BLOCK,
        group_tokens_per_block=[TOKENS_PER_BLOCK],
    )
    assert next_chunk is not None
    assert (next_chunk.op.start, next_chunk.op.end) == (
        3 * TOKENS_PER_BLOCK,
        4 * TOKENS_PER_BLOCK,
    )


def test_eager_apc_backfill_uses_the_existing_immediate_store_path() -> None:
    """The eager side effect changes coverage, not offload orchestration."""
    harness = _make_connector()
    harness.connector.lazy_offload = False
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="E-immediate",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )
    assert harness.connector.get_num_new_matched_tokens(
        request, num_computed_tokens=3 * TOKENS_PER_BLOCK + 4
    ) == (0, False)
    tracker = harness.connector.request_trackers["E-immediate"]
    tracker.allocated_block_ids = {0: [1, 2, 3, 4]}
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["E-immediate"],
            new_block_ids=[[]],
            resumed_req_ids={"E-immediate"},
        ),
        num_scheduled_tokens={"E-immediate": 4},
    )
    connector_metadata = LMCacheMPConnectorMetadata()

    harness.connector._process_cached_requests(
        scheduler_output,
        connector_metadata,  # type: ignore[arg-type]
    )

    assert len(connector_metadata.requests) == 1
    store = connector_metadata.requests[0]
    assert store.direction == "STORE"
    assert (store.op.start, store.op.end) == (0, 3 * TOKENS_PER_BLOCK)
    assert harness.manager.candidates == []


def test_eager_without_an_aligned_apc_hit_keeps_the_old_chunk_threshold() -> None:
    """The eager fix is inert when APC contributes no complete chunk."""
    harness = _make_connector()
    harness.connector.lazy_offload = False
    tokens = list(range(2 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="E-no-apc",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )

    assert harness.connector.get_num_new_matched_tokens(
        request, num_computed_tokens=4
    ) == (0, False)
    tracker = harness.connector.request_trackers["E-no-apc"]
    tracker.allocated_block_ids = {0: [1, 2]}
    tracker.num_scheduled_tokens = 4

    assert (
        LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker,
            lmcache_tokens_per_chunk=TOKENS_PER_BLOCK,
            group_tokens_per_block=[TOKENS_PER_BLOCK],
        )
        is None
    )


def test_eager_apc_backfill_excludes_the_prefix_already_in_lmcache() -> None:
    """APC and LMCache hits overlap; they are never summed or re-stored."""
    harness = _make_connector()
    harness.connector.lazy_offload = False
    harness.adapter.lookup_result = 2 * TOKENS_PER_BLOCK
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="E-partial",
        status=RequestStatus.WAITING,
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )

    assert harness.connector.get_num_new_matched_tokens(
        request, num_computed_tokens=3 * TOKENS_PER_BLOCK + 4
    ) == (0, False)
    tracker = harness.connector.request_trackers["E-partial"]
    assert tracker.num_lmcache_hit_tokens == 2 * TOKENS_PER_BLOCK
    assert tracker.num_vllm_hit_tokens == 3 * TOKENS_PER_BLOCK
    tracker.allocated_block_ids = {0: [1, 2, 3, 4]}
    tracker.num_scheduled_tokens = 4

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=TOKENS_PER_BLOCK,
        group_tokens_per_block=[TOKENS_PER_BLOCK],
    )

    assert metadata is not None
    assert (metadata.op.start, metadata.op.end) == (
        2 * TOKENS_PER_BLOCK,
        3 * TOKENS_PER_BLOCK,
    )


def test_store_metadata_covers_the_vllm_hit_tokens() -> None:
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    request = SimpleNamespace(
        request_id="F",
        cache_salt=None,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
    )
    tracker = LMCacheMPRequestTracker(request)  # type: ignore[arg-type]
    tracker.allocated_block_ids = {0: [1, 2, 3, 4]}
    tracker.num_scheduled_tokens = 2
    tracker.num_vllm_hit_tokens = 3 * TOKENS_PER_BLOCK

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=TOKENS_PER_BLOCK,
        group_tokens_per_block=[TOKENS_PER_BLOCK],
    )

    assert metadata is not None
    assert (metadata.op.start, metadata.op.end) == (0, 3 * TOKENS_PER_BLOCK)
