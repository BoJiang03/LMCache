# SPDX-License-Identifier: Apache-2.0
"""Answer how much committed L1 a set of chunks holds.

A client, such as a scheduler measuring how much of L1 vLLM's GPU prefix
cache also holds, sends chunk hashes by cache salt, per object group when
the server separates object groups. The module expands them into exact
object keys and answers the committed L1 usage
(:class:`~lmcache.v1.distributed.internal_api.L1CommittedUsage`) of those
keys and of the whole model, with the server's probe hash so the client can
check that both sides hash tokens alike, and the server's object groups so
it can check that both sides name them alike.

The module lives outside the built-in ``modules`` package, so a server pays
nothing unless it measures. Start the MP server with::

    --server-module '{"module_path": "lmcache.v1.multiprocess.l1_residency"}'

Design: ``docs/design/v1/multiprocess/l1_residency.md``.
"""

# Third Party
import msgspec

# First Party
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.ext_server_module import server_module_handler
from lmcache.v1.multiprocess.protocols.l1_residency import (
    L1_RESIDENCY_QUERY_METHOD,
    GroupHeldChunks,
    GroupWindow,
    L1ResidencyQuery,
    L1ResidencyReply,
    compute_probe_hash,
)

# Without separated object groups every chunk is one object per kv rank, in
# one full-attention object group.
_MERGED_OBJECT_GROUPS = [GroupWindow(kind="attention", window_chunks=-1)]


def _salted_keys(
    model_name: str,
    world_size: int,
    chunk_hashes_by_salt: dict[str, list[bytes]],
    object_group_id: int,
) -> list[ObjectKey]:
    """Expand chunks into their object keys in one object group.

    Args:
        model_name: The model's ``ObjectKey.model_name``.
        world_size: The kv world size whose ranks the keys fan out to.
        chunk_hashes_by_salt: Per cache salt, the chunk hashes to expand.
        object_group_id: The object group of every key.

    Returns:
        One key per (chunk, kv rank), salted per chunk, as a lookup would.

    Raises:
        ValueError: If a cache salt violates the ``ObjectKey`` invariants.
    """
    keys: list[ObjectKey] = []
    for cache_salt, chunk_hashes in chunk_hashes_by_salt.items():
        # Only model, world size and salt select the keys; worker_id None
        # fans out to every kv rank, as for a lookup.
        ipc_key = IPCCacheServerKey(
            model_name=model_name,
            world_size=world_size,
            worker_id=None,
            token_ids=(),
            start=0,
            end=0,
            request_id="",
            cache_salt=cache_salt,
        )
        keys.extend(ipc_key_to_object_keys(ipc_key, chunk_hashes, [object_group_id])[0])
    return keys


def _query_keys(
    request: L1ResidencyQuery,
    object_groups: list[GroupWindow],
    chunk_hashes_by_salt: dict[str, list[bytes]],
    chunk_hashes_by_group: list[GroupHeldChunks],
) -> list[ObjectKey]:
    """Expand one set of a query's chunks into object keys.

    With one object group ``chunk_hashes_by_salt`` counts in it; with several,
    each ``chunk_hashes_by_group`` entry counts in the object group its window
    names, and an entry naming none counts nothing.

    Args:
        request: The query, for its model and world size.
        object_groups: The server's object groups of that model and world size.
        chunk_hashes_by_salt: The chunks to expand for one object group.
        chunk_hashes_by_group: The chunks to expand per object group.

    Returns:
        The object keys of the chunks.

    Raises:
        ValueError: If a cache salt violates the ``ObjectKey`` invariants.
    """
    if len(object_groups) == 1:
        return _salted_keys(
            request.model_name, request.world_size, chunk_hashes_by_salt, 0
        )
    keys: list[ObjectKey] = []
    for held in chunk_hashes_by_group:
        if held.window not in object_groups:
            continue
        keys.extend(
            _salted_keys(
                request.model_name,
                request.world_size,
                held.chunk_hashes_by_salt,
                object_groups.index(held.window),
            )
        )
    return keys


class L1ResidencyModule(DiscoverableModule):
    """Serve ``l1_residency.query`` from the storage manager's L1.

    Args:
        ctx: Engine context whose storage manager and, when it separates
            object groups, layout registry are read.

    Raises:
        ValueError: If the server separates object groups and caches full
            sliding-window KV: every attention group then reports a full
            window, so a window no longer names one object group.
    """

    module_name = "l1_residency"

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build the module over ``build_ctx``'s engine context.

        Args:
            build_ctx: The server's module build context.

        Returns:
            The module; never ``None``, since it needs no optional resource.
        """
        return cls(build_ctx.engine_context)

    def __init__(self, ctx: MPCacheServerContext) -> None:
        if ctx.separate_object_groups and ctx.full_sw_kv:
            raise ValueError(
                "lmcache.v1.multiprocess.l1_residency does not support "
                "separate_object_groups together with full_sw_kv: object "
                "groups are then not told apart by their windows"
            )
        self._ctx = ctx

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context."""
        return self._ctx

    def report_status(self) -> dict:
        """Return no status; the module keeps no state."""
        return {}

    def close(self) -> None:
        """Nothing to release; the module keeps no state."""

    @server_module_handler(L1_RESIDENCY_QUERY_METHOD)
    def query(self, payload: bytes) -> bytes:
        """Answer an encoded :class:`L1ResidencyQuery`.

        With one object group the query's ``chunk_hashes_by_salt`` count in
        it; with several, each ``chunk_hashes_by_group`` entry counts in the
        object group its window names, and an entry naming none counts
        nothing (the client sees that from the reply's ``object_groups``).

        Args:
            payload: The msgpack-encoded query.

        Returns:
            The msgpack-encoded :class:`L1ResidencyReply`.

        Raises:
            msgspec.DecodeError: If the payload is not an encoded query
                (its subclass ``msgspec.ValidationError`` when well-formed
                msgpack has the wrong shape).
            ValueError: If a cache salt violates the ``ObjectKey`` invariants,
                or the server separates object groups and the queried model
                and world size have no registered KV cache.
        """
        request = msgspec.msgpack.decode(payload, type=L1ResidencyQuery)
        object_groups = self._object_groups(request.model_name, request.world_size)
        keys = _query_keys(
            request,
            object_groups,
            request.chunk_hashes_by_salt,
            request.chunk_hashes_by_group,
        )
        inuse_keys = _query_keys(
            request,
            object_groups,
            request.inuse_chunk_hashes_by_salt,
            request.inuse_chunk_hashes_by_group,
        )
        usage = self._ctx.storage_manager.get_l1_committed_usage(
            keys, request.model_name
        )
        # A second lock pass (and model scan); objects committed or evicted
        # between the two can make the in-use count disagree by a few.
        inuse_usage = self._ctx.storage_manager.get_l1_committed_usage(
            inuse_keys, request.model_name
        )
        return msgspec.msgpack.encode(
            L1ResidencyReply(
                probe_hash=compute_probe_hash(self._ctx.token_hasher),
                matched_objects=usage.matched_objects,
                matched_bytes=usage.matched_bytes,
                model_objects=usage.model_objects,
                model_bytes=usage.model_bytes,
                object_groups=object_groups,
                matched_inuse_objects=inuse_usage.matched_objects,
                matched_inuse_bytes=inuse_usage.matched_bytes,
                matched_demoted_objects=usage.matched_demoted_objects,
                matched_demoted_bytes=usage.matched_demoted_bytes,
                matched_mru_objects={
                    c: n for c, n in usage.matched_mru_objects.items()
                },
                matched_mru_bytes={c: n for c, n in usage.matched_mru_bytes.items()},
                matched_inuse_demoted_bytes=inuse_usage.matched_demoted_bytes,
                matched_inuse_mru_bytes={
                    c: n for c, n in inuse_usage.matched_mru_bytes.items()
                },
            )
        )

    def _object_groups(self, model_name: str, world_size: int) -> list[GroupWindow]:
        """Return the object groups of a model and world size, by id.

        Args:
            model_name: The model name.
            world_size: The kv world size.

        Returns:
            One full-attention group without separated object groups, else
            each registered object group's kind and window.

        Raises:
            ValueError: If the server separates object groups and no worker
                of the pair registered its KV cache yet.
        """
        if not self._ctx.separate_object_groups:
            return _MERGED_OBJECT_GROUPS
        attn_desc = self._ctx.layout_desc_registry.find_attn_desc(
            model_name, world_size
        )
        # A producer that predates kinds labels none: every group is attention.
        kinds = attn_desc.group_kinds or ("attention",) * attn_desc.num_object_groups
        return [
            GroupWindow(kind=kind, window_chunks=window)
            for kind, window in zip(kinds, attn_desc.num_chunks_in_sw, strict=True)
        ]
