# SPDX-License-Identifier: Apache-2.0
"""Tests for the delta + packed ``token_ids`` protocol.

Two changes to how a store or retrieve describes its tokens are exercised
here. A key carries only the tokens its own ``[start, end)`` range needs, at
``token_offset``, instead of the request's whole sequence; and it carries
them packed rather than as a list of ints. These tests pin the properties
that make both safe: hashes must be identical however the tokens arrive, and
a key whose prefix the server is missing must resolve to nothing at all.
"""

# Standard
from types import SimpleNamespace
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import (
    resolve_obj_keys_from_session,
)
from lmcache.v1.multiprocess.modules.lookup import resolve_prefetched_obj_keys
from lmcache.v1.multiprocess.session import Session, SessionManager
from lmcache.v1.multiprocess.token_codec import (
    TOKEN_STRIDE,
    num_packed_tokens,
    pack_token_ids,
    unpack_token_ids,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK_SIZE = 4


@pytest.fixture
def hasher() -> TokenHasher:
    """TokenHasher with a small chunk_size so ranges stay readable."""
    return TokenHasher(chunk_size=CHUNK_SIZE, hash_algorithm="blake3")


@pytest.fixture
def session(hasher: TokenHasher) -> Session:
    """A fresh Session instance."""
    return Session(request_id="req-delta", hasher=hasher)


class TestTokenCodec:
    def test_round_trip(self) -> None:
        tokens = [0, 1, 127, 128_255, 2**32 - 1]
        assert unpack_token_ids(pack_token_ids(tokens)) == tokens

    def test_empty_round_trip(self) -> None:
        assert pack_token_ids([]) == b""
        assert unpack_token_ids(b"") == []

    def test_stride(self) -> None:
        packed = pack_token_ids([1, 2, 3])
        assert len(packed) == 3 * TOKEN_STRIDE
        assert num_packed_tokens(packed) == 3

    def test_big_endian_uint32_layout(self) -> None:
        """The layout blake3 already hashes; changing it would rekey the cache."""
        assert pack_token_ids([1, 258]) == b"\x00\x00\x00\x01\x00\x00\x01\x02"

    def test_ragged_buffer_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a multiple"):
            unpack_token_ids(b"\x00\x00\x00")

    def test_token_too_wide_rejected(self) -> None:
        with pytest.raises(OverflowError):
            pack_token_ids([2**32])


class TestPackedHashingMatchesTokenList:
    """Packed hashing must be bit-identical to the token-list path."""

    def test_whole_sequence(self, hasher: TokenHasher) -> None:
        tokens = list(range(400, 432))
        assert hasher.compute_packed_chunk_hashes(
            pack_token_ids(tokens)
        ) == hasher.compute_chunk_hashes(tokens)

    def test_sub_range(self, hasher: TokenHasher) -> None:
        tokens = list(range(400, 432))
        assert hasher.compute_packed_chunk_hashes(
            pack_token_ids(tokens), start=8, end=24
        ) == hasher.compute_chunk_hashes(tokens, start=8, end=24)

    def test_single_chunk(self, hasher: TokenHasher) -> None:
        chunk = [7, 8, 9, 10]
        assert hasher.hash_packed_chunk(pack_token_ids(chunk)) == hasher.hash_tokens(
            chunk
        )

    def test_trailing_partial_chunk_is_dropped(self, hasher: TokenHasher) -> None:
        tokens = list(range(400, 410))  # 2 full chunks + 2 spare tokens
        assert len(hasher.compute_packed_chunk_hashes(pack_token_ids(tokens))) == 2


class TestAbsorbTokens:
    def test_seeds_an_empty_session(self, session: Session) -> None:
        assert session.absorb_tokens(0, pack_token_ids([1, 2, 3, 4]))
        assert session.tokens_in_range(0, 4) == [1, 2, 3, 4]

    def test_appends_a_contiguous_slice(self, session: Session) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3, 4]))
        assert session.absorb_tokens(4, pack_token_ids([5, 6]))
        assert session.tokens_in_range(0, 6) == [1, 2, 3, 4, 5, 6]

    def test_overlapping_slice_extends_without_truncating(
        self, session: Session
    ) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3, 4]))
        assert session.absorb_tokens(2, pack_token_ids([3, 4, 5, 6]))
        assert session.tokens_in_range(0, 6) == [1, 2, 3, 4, 5, 6]

    def test_fully_covered_slice_is_a_noop(self, session: Session) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3, 4, 5, 6]))
        assert session.absorb_tokens(2, pack_token_ids([3, 4]))
        assert session.tokens_in_range(0, 6) == [1, 2, 3, 4, 5, 6]

    def test_gap_is_rejected(self, session: Session) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3, 4]))
        assert not session.absorb_tokens(5, pack_token_ids([6, 7]))
        assert session.num_tokens == 4

    def test_negative_offset_raises(self, session: Session) -> None:
        with pytest.raises(ValueError, match="offset must be >= 0"):
            session.absorb_tokens(-1, pack_token_ids([1]))

    def test_memoized_hashes_survive_absorb(self, session: Session) -> None:
        """Extending must not invalidate hashes already computed."""
        session.set_tokens(pack_token_ids([1, 2, 3, 4]))
        first = list(session.get_hashes(0, 4))
        session.absorb_tokens(4, pack_token_ids([5, 6, 7, 8]))
        assert list(session.get_hashes(0, 4)) == first
        assert len(session.get_hashes(0, 8)) == 2


class TestDeltaMatchesFullSequence:
    """A slice spliced in chunk by chunk hashes like the whole sequence."""

    def test_incremental_absorb_matches_one_shot(self, hasher: TokenHasher) -> None:
        tokens = list(range(100, 116))

        whole = Session(request_id="whole", hasher=hasher)
        whole.set_tokens(pack_token_ids(tokens))
        expected = list(whole.get_hashes(0, 16))

        incremental = Session(request_id="incremental", hasher=hasher)
        for start in range(0, 16, CHUNK_SIZE):
            assert incremental.absorb_tokens(
                start, pack_token_ids(tokens[start : start + CHUNK_SIZE])
            )
            chunk_idx = start // CHUNK_SIZE
            assert (
                list(incremental.get_hashes(start, start + CHUNK_SIZE))
                == expected[chunk_idx : chunk_idx + 1]
            )

    def test_lookup_seeded_prefix_then_decode_deltas(self, hasher: TokenHasher) -> None:
        """The real shape: LOOKUP seeds the prompt, stores append past it."""
        prompt = list(range(200, 208))
        generated = list(range(300, 308))

        whole = Session(request_id="whole", hasher=hasher)
        whole.set_tokens(pack_token_ids(prompt + generated))
        expected = list(whole.get_hashes(0, 16))

        live = Session(request_id="live", hasher=hasher)
        live.set_tokens(pack_token_ids(prompt))
        assert list(live.get_hashes(0, 8)) == expected[:2]
        for i in range(0, 8, CHUNK_SIZE):
            assert live.absorb_tokens(
                8 + i, pack_token_ids(generated[i : i + CHUNK_SIZE])
            )
        assert list(live.get_hashes(0, 16)) == expected


class TestMatchesTokens:
    def test_matching_slice(self, session: Session) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3, 4, 5, 6]))
        assert session.matches_tokens(2, pack_token_ids([3, 4]))

    def test_differing_slice(self, session: Session) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3, 4, 5, 6]))
        assert not session.matches_tokens(2, pack_token_ids([3, 9]))

    def test_slice_past_the_end(self, session: Session) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3, 4]))
        assert not session.matches_tokens(2, pack_token_ids([3, 4, 5]))

    def test_full_sequence(self, session: Session) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3, 4]))
        assert session.matches_tokens(0, pack_token_ids([1, 2, 3, 4]))


class TestTokensInRange:
    def test_returns_absolute_range(self, session: Session) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3, 4, 5, 6]))
        assert session.tokens_in_range(2, 5) == [3, 4, 5]

    def test_clips_past_the_end(self, session: Session) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3, 4]))
        assert session.tokens_in_range(2, 99) == [3, 4]

    def test_packed_range_matches_unpacked(self, session: Session) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3, 4, 5, 6]))
        assert session.packed_tokens_in_range(2, 5) == pack_token_ids([3, 4, 5])

    def test_does_not_alias_session_state(self, session: Session) -> None:
        session.set_tokens(pack_token_ids([1, 2, 3, 4]))
        got = session.tokens_in_range(0, 4)
        got[0] = 999
        assert session.tokens_in_range(0, 1) == [1]


class TestKeyValidation:
    def _key(self, **overrides: object) -> IPCCacheServerKey:
        kwargs: dict[str, object] = {
            "model_name": "m",
            "world_size": 1,
            "worker_id": 0,
            "token_bytes": pack_token_ids([1, 2, 3, 4]),
            "start": 0,
            "end": 4,
            "request_id": "r",
        }
        kwargs.update(overrides)
        return IPCCacheServerKey(**kwargs)  # type: ignore[arg-type]

    def test_default_offset_is_zero(self) -> None:
        key = self._key()
        assert key.token_offset == 0
        assert key.carries_full_sequence
        assert key.num_tokens == 4
        assert key.tokens_end == 4

    def test_delta_key_reports_its_span(self) -> None:
        key = self._key(
            token_bytes=pack_token_ids([5, 6]), start=4, end=6, token_offset=4
        )
        assert not key.carries_full_sequence
        assert key.num_tokens == 2
        assert key.tokens_end == 6

    def test_negative_offset_rejected(self) -> None:
        with pytest.raises(ValueError, match="token_offset must be >= 0"):
            self._key(token_offset=-1)

    def test_offset_past_start_rejected(self) -> None:
        """A key that does not carry its own range is a protocol bug."""
        with pytest.raises(ValueError, match="must not exceed start"):
            self._key(
                token_bytes=pack_token_ids([5, 6]), start=4, end=6, token_offset=5
            )

    def test_ragged_token_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="whole number"):
            self._key(token_bytes=b"\x00\x00\x00")

    def test_offset_survives_no_worker_id_version(self) -> None:
        key = self._key(
            token_bytes=pack_token_ids([5, 6]), start=4, end=6, token_offset=4
        )
        assert key.no_worker_id_version().token_offset == 4

    def test_offset_is_part_of_identity(self) -> None:
        """The same tokens at a different offset are different content."""
        a = self._key(
            token_bytes=pack_token_ids([5, 6]), start=4, end=6, token_offset=4
        )
        b = self._key(
            token_bytes=pack_token_ids([5, 6]), start=0, end=2, token_offset=0
        )
        assert a != b

    def test_from_token_ids_packs(self) -> None:
        key = IPCCacheServerKey.from_token_ids(
            model_name="m", world_size=1, worker_id=0, token_ids=[1, 2, 3, 4]
        )
        assert key.token_bytes == pack_token_ids([1, 2, 3, 4])


class TestResolvedKeysAreTransportIndependent:
    """A delta key and a full key must resolve to the very same object keys."""

    def _ctx(self, hasher: TokenHasher) -> SimpleNamespace:
        """The slice of the server context that key resolution reads."""
        registry = SimpleNamespace(
            find_attn_desc=lambda model_name, world_size: AttnWindowDesc(
                num_chunks_in_sw=[-1]
            )
        )
        return SimpleNamespace(
            chunk_size=CHUNK_SIZE,
            token_hasher=hasher,
            layout_desc_registry=registry,
        )

    def _key(
        self, tokens: list[int], start: int, end: int, offset: int
    ) -> IPCCacheServerKey:
        return IPCCacheServerKey(
            model_name="m",
            world_size=1,
            worker_id=0,
            num_kv_readers=1,
            token_bytes=pack_token_ids(tokens[offset:end]),
            start=start,
            end=end,
            request_id="r",
            token_offset=offset,
        )

    def test_delta_and_full_resolve_identically(self, hasher: TokenHasher) -> None:
        tokens = list(range(400, 416))
        ctx = self._ctx(hasher)

        full_session = Session(request_id="full", hasher=hasher)
        full_session.set_tokens(pack_token_ids(tokens))
        full_keys = resolve_prefetched_obj_keys(
            ctx, full_session, self._key(tokens, 8, 16, 0), hit_chunks=4, locked_gids=()
        )

        # The delta key carries only [8, 16); the session already holds the
        # prefix, the way a LOOKUP would have left it.
        delta_session = Session(request_id="delta", hasher=hasher)
        delta_session.set_tokens(pack_token_ids(tokens[:8]))
        delta_keys = resolve_prefetched_obj_keys(
            ctx,
            delta_session,
            self._key(tokens, 8, 16, 8),
            hit_chunks=4,
            locked_gids=(),
        )

        assert full_keys
        assert delta_keys == full_keys

    def test_missing_prefix_resolves_to_nothing(self, hasher: TokenHasher) -> None:
        """Guessing keys here would decrement another request's read locks."""
        tokens = list(range(400, 416))
        empty_session = Session(request_id="empty", hasher=hasher)

        resolved = resolve_prefetched_obj_keys(
            self._ctx(hasher),
            empty_session,
            self._key(tokens, 8, 16, 8),
            hit_chunks=4,
            locked_gids=(),
        )

        assert resolved == []


class TestSessionTtlRunsFromLastUse:
    def test_absorb_keeps_a_session_alive(self, hasher: TokenHasher) -> None:
        """A long request keeps touching its session; expiry must not drop it."""
        manager = SessionManager(hasher, ttl=0.2, cleanup_interval=None)
        try:
            session = manager.get_or_create("long-request")
            session.set_tokens(pack_token_ids([1, 2, 3, 4]))
            for _ in range(4):
                time.sleep(0.1)
                assert session.absorb_tokens(
                    session.num_tokens, pack_token_ids([9, 9, 9, 9])
                )
                assert manager.cleanup_expired() == 0
            assert manager.get("long-request") is not None
        finally:
            manager.close()

    def test_idle_session_still_expires(self, hasher: TokenHasher) -> None:
        manager = SessionManager(hasher, ttl=0.05, cleanup_interval=None)
        try:
            manager.get_or_create("idle").set_tokens(pack_token_ids([1, 2, 3, 4]))
            time.sleep(0.1)
            assert manager.cleanup_expired() == 1
            assert manager.get("idle") is None
        finally:
            manager.close()


class TestStoreKeysAreTransportIndependent:
    """The property the whole change rests on, on the store/retrieve path.

    ``resolve_obj_keys_from_session`` is what a STORE or RETRIEVE resolves
    through. Whatever a key carried -- the whole sequence or just its own
    range -- it must name exactly the same objects, or a delta-sending
    client would write to keys a full-sending client could never read back.
    """

    def _key(
        self, tokens: list[int], start: int, end: int, offset: int
    ) -> IPCCacheServerKey:
        return IPCCacheServerKey(
            model_name="m",
            world_size=2,
            worker_id=1,
            num_kv_readers=1,
            token_bytes=pack_token_ids(tokens[offset:end]),
            start=start,
            end=end,
            request_id="r",
            token_offset=offset,
        )

    def test_delta_store_names_the_same_objects(self, hasher: TokenHasher) -> None:
        tokens = list(range(700, 716))

        full_session = Session(request_id="full", hasher=hasher)
        full_keys = resolve_obj_keys_from_session(
            full_session, self._key(tokens, 8, 16, 0), [0]
        )

        # The delta key's session holds only the LOOKUP-seeded prefix.
        delta_session = Session(request_id="delta", hasher=hasher)
        delta_session.set_tokens(pack_token_ids(tokens[:8]))
        delta_keys = resolve_obj_keys_from_session(
            delta_session, self._key(tokens, 8, 16, 8), [0]
        )

        assert full_keys[0]
        assert delta_keys == full_keys

    def test_successive_delta_stores_chain(self, hasher: TokenHasher) -> None:
        """Decode-phase stores append past the prompt, one range at a time."""
        tokens = list(range(700, 732))

        full_session = Session(request_id="full", hasher=hasher)
        expected = resolve_obj_keys_from_session(
            full_session, self._key(tokens, 0, 32, 0), [0]
        )[0]

        live = Session(request_id="live", hasher=hasher)
        resolved: list[object] = []
        for start in range(0, 32, 8):
            resolved.extend(
                resolve_obj_keys_from_session(
                    live, self._key(tokens, start, start + 8, start), [0]
                )[0]
            )
        assert resolved == expected

    def test_missing_prefix_yields_no_keys(self, hasher: TokenHasher) -> None:
        """Never mint keys across a gap: they would name unwritten content."""
        tokens = list(range(700, 716))
        empty = Session(request_id="empty", hasher=hasher)

        resolved = resolve_obj_keys_from_session(
            empty, self._key(tokens, 8, 16, 8), [0]
        )

        assert resolved == [[]]

    def test_worker_id_none_is_rejected(self, hasher: TokenHasher) -> None:
        key = self._key(list(range(700, 708)), 0, 8, 0)
        session = Session(request_id="r", hasher=hasher)
        with pytest.raises(ValueError, match="worker_id"):
            resolve_obj_keys_from_session(session, key.no_worker_id_version(), [0])
