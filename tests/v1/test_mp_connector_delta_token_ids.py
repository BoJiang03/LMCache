# SPDX-License-Identifier: Apache-2.0
"""Tests for delta ``token_ids`` on MP connector store/retrieve ops.

Under ``TokenIdsTransport.DELTA`` an op carries only its own
``[start, end)`` range, at ``token_offset``, instead of the request's whole
sequence. That is only valid once the LMCache server holds everything before
``start`` for the request, so the tracker tracks how much it has sent and
falls back to the full sequence whenever it cannot prove the prefix is
there. These tests exercise that decision through the public tracker and
metadata interfaces.
"""

# Standard
from dataclasses import dataclass

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    TokenIdsTransport,
)
from lmcache.integration.vllm.utils import mm_hash_to_token_values  # noqa: E402
from lmcache.v1.multiprocess.token_codec import (  # noqa: E402
    unpack_token_ids,
)

CHUNK = 4
IMAGE_PLACEHOLDER_ID = 99


@dataclass
class _FakePlaceholder:
    offset: int
    length: int


@dataclass
class _FakeMMFeature:
    identifier: str
    mm_position: _FakePlaceholder


@dataclass
class _FakeSamplingParams:
    extra_args: dict[str, object] | None = None


class _FakeRequest:
    """Duck-typed vLLM Request carrying only what the tracker reads."""

    def __init__(
        self,
        prompt_token_ids: list[int],
        mm_features: list[_FakeMMFeature] | None = None,
    ) -> None:
        self.request_id = "req-delta"
        self.resumable = False
        self.cache_salt = ""
        self.prompt_token_ids = list(prompt_token_ids)
        self._live_token_ids = list(prompt_token_ids)
        self.all_token_ids = ConstantList(self._live_token_ids)
        self.mm_features = mm_features or []
        self.sampling_params = _FakeSamplingParams()
        self.block_hashes: list = []

    def append_decode_tokens(self, token_ids: list[int]) -> None:
        """Simulate vLLM appending decode tokens to the live token list."""
        self._live_token_ids.extend(token_ids)


def _storable_tracker(
    request: _FakeRequest, num_tokens: int
) -> LMCacheMPRequestTracker:
    """A tracker with enough scheduled tokens and blocks to store everything."""
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {0: list(range(num_tokens // CHUNK))}
    tracker.num_scheduled_tokens = num_tokens
    return tracker


def _store(
    tracker: LMCacheMPRequestTracker,
    transport: TokenIdsTransport,
) -> LMCacheMPRequestMetadata | None:
    return LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=CHUNK,
        group_tokens_per_block=[CHUNK],
        transport=transport,
    )


class TestGetTokenIdsSlice:
    def test_text_only(self) -> None:
        prompt = list(range(100, 116))
        tracker = LMCacheMPRequestTracker(_FakeRequest(prompt))
        assert tracker.get_token_ids_slice(4, 12) == prompt[4:12]

    def test_slice_inside_the_mm_prompt(self) -> None:
        prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 2 + [3, 4, 5, 6]
        tracker = LMCacheMPRequestTracker(
            _FakeRequest(prompt, [_FakeMMFeature("0xabcd", _FakePlaceholder(2, 2))])
        )
        assert tracker.get_token_ids_slice(0, 4) == tracker.get_token_ids()[0:4]

    def test_slice_spanning_prompt_and_decode(self) -> None:
        prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 2 + [3, 4, 5, 6]
        request = _FakeRequest(
            prompt, [_FakeMMFeature("0xabcd", _FakePlaceholder(2, 2))]
        )
        tracker = LMCacheMPRequestTracker(request)
        request.append_decode_tokens([500, 501])
        assert tracker.get_token_ids_slice(2, 10) == tracker.get_token_ids()[2:10]

    def test_slice_entirely_in_decode(self) -> None:
        prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 2 + [3, 4, 5, 6]
        request = _FakeRequest(
            prompt, [_FakeMMFeature("0xabcd", _FakePlaceholder(2, 2))]
        )
        tracker = LMCacheMPRequestTracker(request)
        request.append_decode_tokens([500, 501])
        assert tracker.get_token_ids_slice(8, 10) == [500, 501]

    def test_slice_agrees_with_full_everywhere(self) -> None:
        prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 3 + [3, 4, 5]
        request = _FakeRequest(
            prompt, [_FakeMMFeature("0xabcd", _FakePlaceholder(2, 3))]
        )
        tracker = LMCacheMPRequestTracker(request)
        request.append_decode_tokens([500, 501, 502])
        full = tracker.get_token_ids()
        for start in range(len(full) + 1):
            for end in range(start, len(full) + 1):
                assert tracker.get_token_ids_slice(start, end) == full[start:end]


class TestStoreTransport:
    def test_full_transport_sends_the_whole_sequence(self) -> None:
        prompt = list(range(100, 108))
        tracker = _storable_tracker(_FakeRequest(prompt), 8)
        tracker.note_tokens_at_server(8)

        metadata = _store(tracker, TokenIdsTransport.FULL)

        assert metadata is not None
        assert unpack_token_ids(metadata.op.token_bytes) == prompt
        assert metadata.op.token_offset == 0

    def test_delta_transport_sends_only_the_op_range(self) -> None:
        prompt = list(range(100, 116))
        tracker = _storable_tracker(_FakeRequest(prompt), 16)
        # The lookup already handed the whole prompt to the server.
        tracker.note_tokens_at_server(16)
        tracker.num_stored_tokens = 8

        metadata = _store(tracker, TokenIdsTransport.DELTA)

        assert metadata is not None
        assert metadata.op.start == 8
        assert metadata.op.end == 16
        assert metadata.op.token_offset == 8
        assert unpack_token_ids(metadata.op.token_bytes) == prompt[8:16]

    def test_delta_falls_back_when_the_server_has_no_prefix(self) -> None:
        """No lookup was submitted, so the server has nothing to chain onto."""
        prompt = list(range(100, 116))
        tracker = _storable_tracker(_FakeRequest(prompt), 16)
        tracker.num_stored_tokens = 8
        assert tracker.num_tokens_at_server == 0

        metadata = _store(tracker, TokenIdsTransport.DELTA)

        assert metadata is not None
        assert metadata.op.token_offset == 0
        assert unpack_token_ids(metadata.op.token_bytes) == prompt

    def test_an_op_at_zero_needs_no_prefix(self) -> None:
        """[0, end) chains onto nothing, so it is a delta even with no lookup."""
        prompt = list(range(100, 116))
        request = _FakeRequest(prompt)
        tracker = _storable_tracker(request, 8)
        assert tracker.num_tokens_at_server == 0

        first = _store(tracker, TokenIdsTransport.DELTA)
        assert first is not None
        assert first.op.token_offset == 0
        assert unpack_token_ids(first.op.token_bytes) == prompt[0:8]
        assert tracker.num_tokens_at_server == 8

        request.append_decode_tokens([500, 501, 502, 503])
        tracker.num_scheduled_tokens = 20
        tracker.allocated_block_ids = {0: list(range(5))}

        second = _store(tracker, TokenIdsTransport.DELTA)
        assert second is not None
        assert second.op.start == 8
        assert second.op.token_offset == 8

    def test_a_gap_falls_back_to_the_full_sequence(self) -> None:
        """A store starting past what the server holds must resend everything."""
        prompt = list(range(100, 116))
        tracker = _storable_tracker(_FakeRequest(prompt), 16)
        tracker.num_stored_tokens = 8
        assert tracker.num_tokens_at_server == 0

        metadata = _store(tracker, TokenIdsTransport.DELTA)

        assert metadata is not None
        assert metadata.op.start == 8
        assert metadata.op.token_offset == 0
        assert unpack_token_ids(metadata.op.token_bytes) == prompt

    def test_forget_forces_a_full_resend(self) -> None:
        prompt = list(range(100, 116))
        tracker = _storable_tracker(_FakeRequest(prompt), 16)
        tracker.note_tokens_at_server(16)
        tracker.num_stored_tokens = 8
        tracker.forget_tokens_at_server()

        metadata = _store(tracker, TokenIdsTransport.DELTA)

        assert metadata is not None
        assert metadata.op.token_offset == 0
        assert unpack_token_ids(metadata.op.token_bytes) == prompt

    def test_delta_and_full_describe_the_same_range(self) -> None:
        """Whatever the transport, the op's range and blocks are identical."""
        prompt = list(range(100, 116))

        full_tracker = _storable_tracker(_FakeRequest(prompt), 16)
        full_tracker.note_tokens_at_server(16)
        full_tracker.num_stored_tokens = 8
        full_meta = _store(full_tracker, TokenIdsTransport.FULL)

        delta_tracker = _storable_tracker(_FakeRequest(prompt), 16)
        delta_tracker.note_tokens_at_server(16)
        delta_tracker.num_stored_tokens = 8
        delta_meta = _store(delta_tracker, TokenIdsTransport.DELTA)

        assert full_meta is not None and delta_meta is not None
        assert full_meta.op.start == delta_meta.op.start
        assert full_meta.op.end == delta_meta.op.end
        assert full_meta.op.block_ids == delta_meta.op.block_ids
        assert unpack_token_ids(full_meta.op.token_bytes)[
            delta_meta.op.token_offset :
        ] == unpack_token_ids(delta_meta.op.token_bytes)

    def test_mm_adjusted_tokens_survive_the_delta(self) -> None:
        prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 2 + [3, 4, 5, 6]
        tracker = _storable_tracker(
            _FakeRequest(prompt, [_FakeMMFeature("0xabcd", _FakePlaceholder(2, 2))]),
            8,
        )
        tracker.note_tokens_at_server(8)

        metadata = _store(tracker, TokenIdsTransport.DELTA)

        assert metadata is not None
        v = list(mm_hash_to_token_values("0xabcd", 2))
        assert unpack_token_ids(metadata.op.token_bytes) == [1, 2, *v, 3, 4, 5, 6]


class TestRetrieveTransport:
    def _tracker(self, prompt: list[int]) -> LMCacheMPRequestTracker:
        tracker = LMCacheMPRequestTracker(_FakeRequest(prompt))
        tracker.allocated_block_ids = {0: list(range(len(prompt) // CHUNK))}
        tracker.num_lmcache_hit_tokens = len(prompt)
        tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD
        return tracker

    def _retrieve(
        self,
        tracker: LMCacheMPRequestTracker,
        transport: TokenIdsTransport,
    ) -> LMCacheMPRequestMetadata | None:
        return LMCacheMPRequestMetadata.GetRetrieveMetadata(
            tracker,
            lmcache_tokens_per_chunk=CHUNK,
            group_tokens_per_block=[CHUNK],
            transport=transport,
        )

    def test_delta_retrieve_sends_only_the_loaded_range(self) -> None:
        prompt = list(range(100, 116))
        tracker = self._tracker(prompt)
        tracker.note_tokens_at_server(16)
        tracker.num_vllm_hit_tokens = 8

        metadata = self._retrieve(tracker, TokenIdsTransport.DELTA)

        assert metadata is not None
        assert metadata.op.start == 8
        assert metadata.op.end == 16
        assert metadata.op.token_offset == 8
        assert unpack_token_ids(metadata.op.token_bytes) == prompt[8:16]

    def test_full_retrieve_unchanged(self) -> None:
        prompt = list(range(100, 116))
        tracker = self._tracker(prompt)
        tracker.num_vllm_hit_tokens = 8

        metadata = self._retrieve(tracker, TokenIdsTransport.FULL)

        assert metadata is not None
        assert metadata.op.token_offset == 0
        assert unpack_token_ids(metadata.op.token_bytes) == prompt

    def test_skip_first_n_tokens_is_transport_independent(self) -> None:
        prompt = list(range(100, 116))
        full_tracker = self._tracker(prompt)
        full_tracker.num_vllm_hit_tokens = 10
        delta_tracker = self._tracker(prompt)
        delta_tracker.note_tokens_at_server(16)
        delta_tracker.num_vllm_hit_tokens = 10

        full_meta = self._retrieve(full_tracker, TokenIdsTransport.FULL)
        delta_meta = self._retrieve(delta_tracker, TokenIdsTransport.DELTA)

        assert full_meta is not None and delta_meta is not None
        assert (
            full_meta.op.skip_first_n_tokens == delta_meta.op.skip_first_n_tokens == 2
        )
