# SPDX-License-Identifier: Apache-2.0
"""Session token deltas must derive the same chunk hashes as whole prefixes.

A STORE key carries only ``[start, end)`` plus ``token_offset``; the session
continues the rolling chunk hash it already holds instead of being handed the
request's whole prompt again. If that derivation ever diverged from the
whole-prefix one, every stored object would land under a different key and the
cache would silently stop hitting -- no error, just a permanent miss.
"""

# Standard
import random

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import SessionTokenGapError
from lmcache.v1.multiprocess.session import Session
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK = 64
N_CHUNKS = 12


@pytest.fixture
def hasher():
    return TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")


@pytest.fixture
def tokens():
    rng = random.Random(20260903)
    # A ragged tail: the last chunk is deliberately incomplete.
    return [rng.randrange(100_000) for _ in range(CHUNK * N_CHUNKS + 17)]


def _reference_hashes(hasher, tokens):
    """What the whole-prefix path produces, chunk by chunk."""
    session = Session(request_id="reference", hasher=hasher)
    session.set_tokens(list(tokens))
    return list(session.get_hashes(0))


def _ranges(first_start=0):
    return [(i * CHUNK, (i + 1) * CHUNK) for i in range(first_start // CHUNK, N_CHUNKS)]


@pytest.mark.parametrize("hit_chunks", [0, 1, 5])
def test_deltas_match_whole_prefix(hasher, tokens, hit_chunks):
    """Delta stores and whole-prefix stores produce identical chunk hashes.

    ``hit_chunks`` stands in for a lookup that already found a prefix, which is
    what makes the first store start at a non-zero offset.
    """
    old = Session(request_id="whole-prefix", hasher=hasher)
    new = Session(request_id="delta", hasher=hasher)
    if hit_chunks:
        # A lookup lands first with the whole prompt in both worlds.
        old.set_tokens(list(tokens))
        new.set_tokens(list(tokens))

    old_hashes, new_hashes = [], []
    for start, end in _ranges(hit_chunks * CHUNK):
        old.set_tokens(list(tokens[:end]))
        old_hashes += list(old.get_hashes(start, end))
        new.extend_tokens(list(tokens[start:end]), start)
        new_hashes += list(new.get_hashes(start, end))

    assert new_hashes == old_hashes
    # The spliced session's chain is also intact end to end, not just
    # per-range: a fresh whole-prompt session reproduces it exactly.
    assert list(new.chunk_hashes) == _reference_hashes(hasher, tokens)


def test_resending_a_delta_is_idempotent(hasher, tokens):
    session = Session(request_id="idempotent", hasher=hasher)
    session.extend_tokens(list(tokens[: 2 * CHUNK]), 0)
    first = list(session.get_hashes(CHUNK, 2 * CHUNK))
    session.extend_tokens(list(tokens[CHUNK : 2 * CHUNK]), CHUNK)
    assert list(session.get_hashes(CHUNK, 2 * CHUNK)) == first


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11])
@pytest.mark.parametrize("after_lookup", [True, False])
def test_interleaved_tp_ranks_do_not_truncate_each_other(
    hasher, tokens, seed, after_lookup
):
    """Every TP rank stores the same ranges against one shared session.

    The ranks run asynchronously, so a straggler's low-offset delta can arrive
    long after another rank has pushed the session further along. If splicing
    shortened the sequence, that straggler would strand every rank ahead of it
    behind a gap it could never close -- and each retry would re-truncate. This
    is the regression: the truncating version fails here in the hundreds.
    """
    ranges = _ranges()
    rng = random.Random(seed)
    pending = {rank: list(ranges) for rank in range(8)}
    order = []
    while any(pending.values()):
        rank = rng.choice([r for r, todo in pending.items() if todo])
        order.append(pending[rank].pop(0))

    session = Session(request_id="tp8", hasher=hasher)
    if after_lookup:
        session.set_tokens(list(tokens))

    reference = _reference_hashes(hasher, tokens)
    for start, end in order:
        session.extend_tokens(list(tokens[start:end]), start)
        assert (
            list(session.get_hashes(start, end))
            == (reference[start // CHUNK : end // CHUNK])
        )


def test_a_gap_raises_and_leaves_the_session_untouched(hasher, tokens):
    """A delta that does not reach back to what is held cannot be hashed across.

    The store path turns this into a terminal failure response rather than an
    unhandled handler exception, which would leave the client's future pending
    forever.
    """
    session = Session(request_id="gap", hasher=hasher)
    session.extend_tokens(list(tokens[:CHUNK]), 0)
    held = list(session.token_ids)

    with pytest.raises(SessionTokenGapError):
        session.extend_tokens(list(tokens[3 * CHUNK : 4 * CHUNK]), 3 * CHUNK)

    assert list(session.token_ids) == held
