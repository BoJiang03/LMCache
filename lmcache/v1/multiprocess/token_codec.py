# SPDX-License-Identifier: Apache-2.0
"""Packed binary representation of a token id sequence.

Token ids cross three process boundaries on every scheduler step (scheduler
-> worker via vLLM's shm ring, worker -> LMCache server via ZMQ). As a
``list[int]`` the wire size is fine but the *decode* is not: msgpack and
pickle both have to materialize one Python ``int`` object per token, which
at 200k tokens costs milliseconds per hop. As a packed buffer the same
payload decodes with a single memcpy.

The layout is big-endian ``uint32``, one word per token -- deliberately the
exact byte string
:func:`lmcache.v1.multiprocess.token_hasher._make_blake3_hash_func` already
feeds to blake3 (``struct.pack(f">{n}I", *tokens)``). Chunk hashes are
therefore bit-identical whether they are computed from a token list or
straight from this buffer, so switching representations does not invalidate
a single cached entry.
"""

# Standard
from collections.abc import Sequence
import array
import struct
import sys

TOKEN_STRIDE = 4
"""Bytes per packed token id (big-endian ``uint32``)."""

_NATIVE_U32 = array.array("I").itemsize == TOKEN_STRIDE
"""Whether ``array('I')`` is a 4-byte word here, enabling the fast path."""

_SWAP = sys.byteorder == "little"
"""Whether native words need byte-swapping to reach big-endian order."""


def pack_token_ids(token_ids: Sequence[int]) -> bytes:
    """Pack token ids into the big-endian ``uint32`` wire buffer.

    Args:
        token_ids: The token ids to pack. Each must fit in a ``uint32``,
            which every real vocabulary does.

    Returns:
        ``TOKEN_STRIDE * len(token_ids)`` bytes.

    Raises:
        OverflowError: If a token id does not fit in a ``uint32``.
    """
    if not _NATIVE_U32:
        return struct.pack(f">{len(token_ids)}I", *token_ids)
    # ~2x faster than struct.pack for long sequences: one bulk conversion
    # instead of unpacking the whole sequence as varargs.
    words = array.array("I", token_ids)
    if _SWAP:
        words.byteswap()
    return words.tobytes()


def unpack_token_ids(packed: bytes) -> list[int]:
    """Unpack a wire buffer back into token ids.

    Only callers that genuinely need Python ints should do this -- hashing
    and splicing both work directly on the packed form.

    Args:
        packed: A buffer whose length is a multiple of ``TOKEN_STRIDE``.

    Returns:
        The token ids it holds.

    Raises:
        ValueError: If ``packed`` is not a whole number of tokens.
    """
    if len(packed) % TOKEN_STRIDE:
        raise ValueError(
            f"packed token buffer of {len(packed)} byte(s) is not a multiple "
            f"of {TOKEN_STRIDE}"
        )
    if not _NATIVE_U32:
        return list(struct.unpack(f">{len(packed) // TOKEN_STRIDE}I", packed))
    words = array.array("I")
    words.frombytes(packed)
    if _SWAP:
        words.byteswap()
    return words.tolist()


def num_packed_tokens(packed: bytes) -> int:
    """Return how many token ids ``packed`` holds."""
    return len(packed) // TOKEN_STRIDE
