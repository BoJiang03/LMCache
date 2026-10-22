# SPDX-License-Identifier: Apache-2.0
"""Payloads of the ``l1_residency.query`` server-module method.

Residency here means committed residency: an object counts once its write
finished and it is not temporary, as in
:class:`~lmcache.v1.distributed.internal_api.L1CommittedUsage`.
"""

# Third Party
import msgspec

# First Party
from lmcache.v1.distributed.api import GroupKind
from lmcache.v1.multiprocess.token_hasher import TokenHasher

L1_RESIDENCY_QUERY_METHOD = "l1_residency.query"

_PROBE_TOKENS = [1, 2, 3, 4]


def compute_probe_hash(hasher: TokenHasher) -> bytes:
    """Hash the fixed probe tokens as the first chunk of a sequence.

    The hash goes through the same entry points as a real chunk, starting
    from the hasher's ``none_hash``, so two hashers agree on it only when
    both the hash function and its chain seed agree.

    Args:
        hasher: The hasher whose chunk hashes are being checked.

    Returns:
        The probe chunk's hash, as an ``ObjectKey.chunk_hash``.
    """
    return hasher.hash_to_bytes(hasher.hash_tokens(_PROBE_TOKENS))


class GroupWindow(msgspec.Struct, frozen=True):
    """What tells object groups apart under ``--separate-object-groups``.

    The server buckets its KV cache groups into object groups by kind and
    cross-chunk window, so a client names an object group by both, as
    :class:`~lmcache.v1.distributed.api.AttnWindowDesc` reports them.

    Args:
        kind: ``"attention"`` for attention KV, ``"recurrent"`` for
            Mamba/linear state pages, ``"aux"`` for a connector-private group.
        window_chunks: The window in LMCache chunks; ``-1`` for full
            attention.
    """

    kind: GroupKind
    window_chunks: int


class GroupHeldChunks(msgspec.Struct):
    """The chunks to count in the one object group named by ``window``.

    Args:
        window: The object group, as the server's ``object_groups`` lists it.
        chunk_hashes_by_salt: Per cache salt, distinct ``ObjectKey.chunk_hash``
            values to count in that object group.
    """

    window: GroupWindow
    chunk_hashes_by_salt: dict[str, list[bytes]]


class L1ResidencyQuery(msgspec.Struct):
    """Ask how much committed L1 the given chunks and their model hold.

    The server expands every chunk into its exact object keys: one per kv
    rank of ``world_size``, salted with the dict key, as a lookup of the same
    model and world size would. A server with one object group counts
    ``chunk_hashes_by_salt`` in it; a server with several counts each entry
    of ``chunk_hashes_by_group`` in the object group it names, and skips an
    entry naming none.

    Args:
        model_name: The model's ``ObjectKey.model_name``.
        world_size: The kv world size whose ranks the keys fan out to.
        chunk_hashes_by_salt: Per cache salt, distinct ``ObjectKey.chunk_hash``
            values to count, for a server with one object group.
        chunk_hashes_by_group: The chunks to count per object group, for a
            server with several.
        inuse_chunk_hashes_by_salt: The subset of ``chunk_hashes_by_salt``
            to count again as ``matched_inuse_*``.
        inuse_chunk_hashes_by_group: The subset of ``chunk_hashes_by_group``
            to count again as ``matched_inuse_*``.
    """

    model_name: str
    world_size: int
    chunk_hashes_by_salt: dict[str, list[bytes]]
    chunk_hashes_by_group: list[GroupHeldChunks] = msgspec.field(default_factory=list)
    inuse_chunk_hashes_by_salt: dict[str, list[bytes]] = msgspec.field(
        default_factory=dict
    )
    inuse_chunk_hashes_by_group: list[GroupHeldChunks] = msgspec.field(
        default_factory=list
    )


class L1ResidencyReply(msgspec.Struct):
    """Committed L1 usage of the queried chunks and of their whole model.

    The usage fields mirror
    :class:`~lmcache.v1.distributed.internal_api.L1CommittedUsage`.

    Args:
        probe_hash: The server's :func:`compute_probe_hash`. A client whose
            own probe hash differs hashes chunks differently and matches
            nothing.
        matched_objects: Committed objects among the queried chunks' keys.
        matched_bytes: Their bytes.
        model_objects: Committed objects of the queried model.
        model_bytes: Their bytes.
        object_groups: The server's object groups of the queried model and
            world size, by object group id. One entry means the server counted
            ``chunk_hashes_by_salt``; several mean it counted
            ``chunk_hashes_by_group``.
        matched_inuse_objects: Committed objects among the keys of the
            query's ``inuse_*`` chunks.
        matched_inuse_bytes: Their bytes.
    """

    probe_hash: bytes
    matched_objects: int
    matched_bytes: int
    model_objects: int
    model_bytes: int
    object_groups: list[GroupWindow]
    matched_inuse_objects: int
    matched_inuse_bytes: int
