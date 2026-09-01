# SPDX-License-Identifier: Apache-2.0
"""A preempted multimodal request must look up the tokens it stored.

``ReqMeta.from_request_tracker`` keys the save on the tracker's full token
list -- the prompt plus every token generated so far -- with the placeholder
spans overwritten by mm_hash-derived values. When ``save_decode_cache`` is
enabled, the chunks covering the generated tokens are written to the cache
under those keys.

After a preemption vLLM resets ``num_computed_tokens`` to 0 and calls
``get_num_new_matched_tokens`` again. For a text-only request the query is
built from ``request.all_token_ids``, so it covers those decode chunks. The
multimodal branch rebuilt the query from ``request.prompt_token_ids`` alone
and dropped everything past the prompt, leaving the chunks it had just
written unreachable: stored, never asked for.

These tests pin the round trip, and record the configuration under which
the divergence is unobservable.
"""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm", reason="the v1 adapter imports vLLM at module top")

# Third Party
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (  # noqa: E402
    LMCacheConnectorV1Impl,
    ReqMeta,
    RequestTracker,
)

IMAGE_PLACEHOLDER_ID = 151655
CHUNK_SIZE = 16
BLOCK_SIZE = 16
PROMPT_LEN = 32
IMAGE_OFFSET = 8
IMAGE_LENGTH = 8
NUM_DECODE_TOKENS = 16
MM_IDENTIFIER = "0f2b8c1d4e6a7b9c0d1e2f3a4b5c6d7e"


@dataclass
class _FakePlaceholder:
    offset: int
    length: int


@dataclass
class _FakeMMFeature:
    identifier: str
    mm_position: _FakePlaceholder


class _RecordingLookupClient:
    """Records the token ids each lookup is issued against."""

    def __init__(self) -> None:
        self.queried_token_ids: list[list[int]] = []

    def lookup_cache(self, lookup_id: str) -> int:
        # -1 means "no result cached", forcing a fresh lookup.
        return -1

    def lookup(
        self,
        token_ids: list[int],
        lookup_id: str,
        request_configs: dict | None = None,
    ) -> int:
        self.queried_token_ids.append(list(token_ids))
        return CHUNK_SIZE


class _FakeRequest:
    """Duck-typed vLLM Request carrying only what the adapter reads."""

    def __init__(self, prompt_token_ids: list[int], mm_features: list) -> None:
        self.request_id = "req-preempted"
        self.prompt_token_ids = list(prompt_token_ids)
        self._live_token_ids = list(prompt_token_ids)
        self.all_token_ids = ConstantList(self._live_token_ids)
        self.mm_features = mm_features
        self.sampling_params = None
        self.num_tokens = len(self._live_token_ids)

    def append_decode_tokens(self, token_ids: list[int]) -> None:
        """Simulate vLLM appending generated tokens to the live token list."""
        self._live_token_ids.extend(token_ids)
        self.num_tokens = len(self._live_token_ids)


def _make_prompt() -> list[int]:
    """A prompt with one image placeholder span in the middle."""
    prompt = list(range(1000, 1000 + PROMPT_LEN))
    for i in range(IMAGE_OFFSET, IMAGE_OFFSET + IMAGE_LENGTH):
        prompt[i] = IMAGE_PLACEHOLDER_ID
    return prompt


def _decode_tokens() -> list[int]:
    return list(range(2000, 2000 + NUM_DECODE_TOKENS))


def _make_connector(lookup_client: _RecordingLookupClient) -> LMCacheConnectorV1Impl:
    """A scheduler-role connector with only the fields the lookup path reads."""
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.kv_role = "kv_both"
    # ``lookup_client`` is a read-only property backed by ``self._manager``;
    # inject the recorder through the manager so the property resolves to it.
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lookup_client=lookup_client
    )
    connector._requests_priority = {}
    connector.skip_last_n_tokens = 0
    connector._max_tokens_per_load = 0
    connector._lmcache_chunk_size = CHUNK_SIZE
    connector.load_specs = {}
    connector.config = SimpleNamespace(min_retrieve_tokens=0)  # type: ignore[assignment]
    return connector


def _store_token_ids(
    tracker_token_ids: list[int],
    mm_features: list,
    save_decode_cache: bool,
) -> list[int] | None:
    """Return the token ids the save path would key this request on.

    ``None`` means the save was skipped, so nothing reaches the cache.
    """
    tracker = RequestTracker(
        req_id="req-preempted",
        prompt_len=PROMPT_LEN,
        token_ids=list(tracker_token_ids),
        allocated_block_ids=[0, 1, 2],
        mm_hashes=[f.identifier for f in mm_features],
        mm_positions=[f.mm_position for f in mm_features],
    )
    # The request has already emitted tokens past its prompt.
    tracker.is_decode_phase = True
    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=BLOCK_SIZE,
        lmcache_chunk_size=CHUNK_SIZE,
        save_decode_cache=save_decode_cache,
    )
    if req_meta is None:
        return None
    return list(req_meta.token_ids)


def test_preempted_mm_request_queries_the_tokens_it_stored() -> None:
    """The lookup after a preemption must cover the stored decode chunks.

    With ``save_decode_cache`` on, the save path writes chunks keyed on
    ``mm-adjusted prompt + decode tokens``. The lookup must be issued
    against the same list, or those chunks can never be hit.
    """
    prompt = _make_prompt()
    mm_features = [
        _FakeMMFeature(
            identifier=MM_IDENTIFIER,
            mm_position=_FakePlaceholder(offset=IMAGE_OFFSET, length=IMAGE_LENGTH),
        )
    ]

    stored = _store_token_ids(
        prompt + _decode_tokens(), mm_features, save_decode_cache=True
    )
    assert stored is not None, "save_decode_cache=True must not skip the save"
    assert len(stored) == PROMPT_LEN + NUM_DECODE_TOKENS

    request = _FakeRequest(prompt, mm_features)
    request.append_decode_tokens(_decode_tokens())

    lookup_client = _RecordingLookupClient()
    connector = _make_connector(lookup_client)
    # A preempted request comes back with num_computed_tokens reset to 0.
    connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

    assert len(lookup_client.queried_token_ids) == 1
    queried = lookup_client.queried_token_ids[0]

    assert queried == stored, (
        f"lookup asked for {len(queried)} tokens but the save path stored "
        f"{len(stored)}; the last {len(stored) - len(queried)} tokens' chunks "
        "are unreachable"
    )


def test_lookup_keeps_the_mm_substitution_on_the_prompt() -> None:
    """Appending decode tokens must not undo the placeholder substitution.

    The placeholder span still has to be replaced by mm_hash-derived values,
    otherwise two different images share the query's prompt chunks again.
    """
    prompt = _make_prompt()
    mm_features = [
        _FakeMMFeature(
            identifier=MM_IDENTIFIER,
            mm_position=_FakePlaceholder(offset=IMAGE_OFFSET, length=IMAGE_LENGTH),
        )
    ]
    request = _FakeRequest(prompt, mm_features)
    request.append_decode_tokens(_decode_tokens())

    lookup_client = _RecordingLookupClient()
    connector = _make_connector(lookup_client)
    connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

    queried = lookup_client.queried_token_ids[0]
    span = queried[IMAGE_OFFSET : IMAGE_OFFSET + IMAGE_LENGTH]
    assert IMAGE_PLACEHOLDER_ID not in span, (
        "placeholder ids survived into the lookup key"
    )
    assert len(set(span)) == IMAGE_LENGTH, "span values must be position-dependent"
    assert queried[-NUM_DECODE_TOKENS:] == _decode_tokens()


def test_text_only_request_already_covers_its_decode_tokens() -> None:
    """Control: the text-only path has queried ``all_token_ids`` since #2007.

    This is what makes the multimodal branch the outlier rather than the
    rule, and it passes with or without the multimodal fix.
    """
    prompt = list(range(1000, 1000 + PROMPT_LEN))
    request = _FakeRequest(prompt, mm_features=[])
    request.append_decode_tokens(_decode_tokens())

    lookup_client = _RecordingLookupClient()
    connector = _make_connector(lookup_client)
    connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

    queried = lookup_client.queried_token_ids[0]
    assert queried == prompt + _decode_tokens()


def test_default_config_stores_no_decode_chunks() -> None:
    """Scope: with ``save_decode_cache`` off there is nothing to miss.

    The save is skipped entirely in the decode phase, so the shorter query
    and the longer one hit exactly the same chunks. This is why the
    divergence is unobservable under the default configuration.
    """
    prompt = _make_prompt()
    mm_features = [
        _FakeMMFeature(
            identifier=MM_IDENTIFIER,
            mm_position=_FakePlaceholder(offset=IMAGE_OFFSET, length=IMAGE_LENGTH),
        )
    ]
    stored = _store_token_ids(
        prompt + _decode_tokens(), mm_features, save_decode_cache=False
    )
    assert stored is None
