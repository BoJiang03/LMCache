# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
import enum

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.utils import ConstantList
import torch

# First Party
from lmcache.integration.vllm.utils import (
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    extract_request_configs_from_request,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp
from lmcache.v1.multiprocess.group_view import slice_block_ids_per_group
from lmcache.v1.multiprocess.token_codec import (
    TOKEN_STRIDE,
    num_packed_tokens,
    pack_token_ids,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.request import Request


class TokenIdsTransport(enum.Enum):
    """How much of a request's token sequence an op carries.

    ``FULL`` puts the whole sequence on every store and retrieve, so the
    per-step scheduler -> worker -> server payload grows with the context
    length even when the op only covers one chunk. ``DELTA`` puts only the
    op's own ``[start, end)`` range on the wire and lets the LMCache server
    chain it onto the prefix its session already holds.

    Either way the tokens travel packed (see
    :mod:`lmcache.v1.multiprocess.token_codec`); the transport choice is
    about how many of them ride along, not how they are encoded.

    ``DELTA`` is only safe once the server has that prefix, which the
    request's LOOKUP supplies; see
    :meth:`LMCacheMPRequestTracker.build_op_tokens`, which falls back to the
    full sequence whenever that cannot be established.
    """

    FULL = enum.auto()
    DELTA = enum.auto()


class LMCacheMPRequestState(enum.Enum):
    """
    State machine:
    PREFETCHING -- update_state_after_alloc --> WAITING_FOR_LOAD
    WAITING_FOR_LOAD -- process_loading_requests --> READY
    READY -- failed async load --> BYPASS_LMCACHE
    BYPASS_LMCACHE -- update_state_after_alloc --> READY
    """

    PREFETCHING = enum.auto()
    WAITING_FOR_LOAD = enum.auto()
    READY = enum.auto()
    BYPASS_LMCACHE = enum.auto()


@dataclass
class LMCacheMPRequestTracker:
    # NOTE: this class used vLLM data structures, should be part of
    # vLLM integration code

    request_id: str

    # Read-only list to track the token ids
    all_token_ids: ConstantList[int]

    # Block ids will be updated at update_states_after_alloc and
    # during generation. Keyed by engine_group_idx; non-HMA models use 0.
    allocated_block_ids: dict[int, list[int]] = field(default_factory=dict)

    # Number of scheduled tokens in this request. We keep tracking this to
    # avoid saving tokens whose KV has not been computed yet.
    num_scheduled_tokens: int = 0

    # Number of tokens stored will be initialized when lookup the external
    # hit tokens and will be updated when processing new requests and cached
    # requests.
    num_stored_tokens: int = 0

    # Staging load operation -- save vllm and lmcache hit tokens during lookup
    num_vllm_hit_tokens: int = 0
    num_lmcache_hit_tokens: int = 0

    # Main state
    state: LMCacheMPRequestState = LMCacheMPRequestState.PREFETCHING

    cache_salt: str = ""
    request_configs: dict[str, Any] | None = None
    max_offload_tokens: int | None = None
    lookup_started_at: float | None = None

    mm_adjusted_prompt_ids: list[int] = field(default_factory=list)

    # High-water mark of the token prefix the LMCache server is known to
    # hold for this request, in tokens. Seeded by the LOOKUP and advanced by
    # every op built for it; see `build_op_tokens`.
    num_tokens_at_server: int = 0

    # Packed prefix of `get_token_ids()`, grown in place as the request
    # generates. Packing is what every op and lookup ships, and a request
    # emits one op per scheduler step, so packing the whole sequence each
    # time would put the context length back on the scheduler's critical
    # path -- exactly what the delta transport removes from the wire.
    packed_tokens: bytearray = field(default_factory=bytearray)

    def __init__(self, request: "Request"):
        self.request_id = request.request_id
        self.cache_salt: str = request.cache_salt or ""
        self.request_configs = extract_request_configs_from_request(request)
        self.max_offload_tokens = (self.request_configs or {}).get(
            "lmcache.max_offload_tokens"
        )
        self.lookup_started_at = None
        self.all_token_ids = request.all_token_ids
        self.allocated_block_ids = {}
        self.num_stored_tokens = 0
        self.num_vllm_hit_tokens = 0
        self.num_lmcache_hit_tokens = 0
        self.state = LMCacheMPRequestState.PREFETCHING
        self.num_tokens_at_server = 0
        self.packed_tokens = bytearray()
        self.mm_adjusted_prompt_ids = []
        mm_hashes, mm_positions = extract_mm_features(request)
        if mm_hashes and mm_positions:
            prompt_ids = torch.tensor(request.prompt_token_ids)
            apply_mm_hashes_to_token_ids(prompt_ids, mm_hashes, mm_positions)
            self.mm_adjusted_prompt_ids = prompt_ids.tolist()

    ####
    # Check the state of the request
    ####
    def needs_retrieve(self) -> bool:
        """Check whether the current request needs retrieve, will be used
        update_stage_after_alloc"""
        return (
            self.num_lmcache_hit_tokens > self.num_vllm_hit_tokens
            and self.state
            not in (
                LMCacheMPRequestState.READY,
                LMCacheMPRequestState.BYPASS_LMCACHE,
            )
        )

    def is_ready_for_retrieving(self) -> bool:
        """Check whether the current request is ready for retrieving,
        will be used in process_loading_requests"""
        return (
            self.state == LMCacheMPRequestState.WAITING_FOR_LOAD
            and self.needs_retrieve()
        )

    ####
    # Update internal states
    ####
    def increase_num_scheduled_tokens(self, num_new_tokens: int):
        self.num_scheduled_tokens += num_new_tokens

    def increase_num_stored_tokens(self, num_new_tokens: int):
        """Increase the number of stored tokens for the current request
        This function will be called when processing the cached requests.
        """
        self.num_stored_tokens += num_new_tokens

    def append_block_ids(
        self,
        new_block_ids: tuple[list[int], ...],
    ):
        """Update the block ids for the current request
        This function will be called when processing the cached requests.
        """
        for engine_group_idx, group_block_ids in enumerate(new_block_ids):
            if group_block_ids:
                self.allocated_block_ids.setdefault(engine_group_idx, []).extend(
                    group_block_ids
                )

    def num_allocated_blocks(self) -> dict[int, int]:
        return {
            engine_group_idx: len(blocks)
            for engine_group_idx, blocks in self.allocated_block_ids.items()
        }

    def get_token_ids(self) -> list[int]:
        """Return the token ids to use for LMCache key derivation."""
        if not self.mm_adjusted_prompt_ids:
            return list(self.all_token_ids)
        num_prompt_tokens = len(self.mm_adjusted_prompt_ids)
        return self.mm_adjusted_prompt_ids + list(
            self.all_token_ids[num_prompt_tokens:]
        )

    def get_token_ids_slice(self, start: int, end: int) -> list[int]:
        """Return ``get_token_ids()[start:end]`` without building the whole list.

        Args:
            start: Absolute start position.
            end: Absolute end position (exclusive).

        Returns:
            The token ids in ``[start, end)``.
        """
        num_prompt_tokens = len(self.mm_adjusted_prompt_ids)
        if end <= num_prompt_tokens:
            return self.mm_adjusted_prompt_ids[start:end]
        if start >= num_prompt_tokens:
            return list(self.all_token_ids[start:end])
        return self.mm_adjusted_prompt_ids[start:] + list(
            self.all_token_ids[num_prompt_tokens:end]
        )

    def packed_token_ids(self) -> bytes:
        """Return the whole current token sequence, packed.

        Returns:
            Every token ``get_token_ids`` would return, in packed form.
        """
        return self.packed_token_slice(0, len(self.all_token_ids))

    def packed_token_slice(self, start: int, end: int) -> bytes:
        """Return the packed tokens at ``[start, end)``.

        Args:
            start: Absolute start position.
            end: Absolute end position (exclusive).

        Returns:
            The packed tokens in the (clipped) range.
        """
        self._pack_through(end)
        return bytes(self.packed_tokens[start * TOKEN_STRIDE : end * TOKEN_STRIDE])

    def _pack_through(self, end: int) -> None:
        """Extend the packed buffer so it covers up to ``end`` tokens.

        Args:
            end: Absolute position the buffer must reach, clipped to the
                tokens the request actually has.
        """
        packed_so_far = len(self.packed_tokens) // TOKEN_STRIDE
        if end <= packed_so_far:
            return
        self.packed_tokens += pack_token_ids(
            self.get_token_ids_slice(packed_so_far, end)
        )

    def note_tokens_at_server(self, num_tokens: int) -> None:
        """Record that the server holds at least ``num_tokens`` of the prefix.

        Args:
            num_tokens: Prefix length the server is known to hold. Smaller
                values are ignored, so a call that sent nothing can pass 0.
        """
        self.num_tokens_at_server = max(self.num_tokens_at_server, num_tokens)

    def forget_tokens_at_server(self) -> None:
        """Forget what the server holds, forcing the next op to resend it all.

        Called when the server may have lost this request's session, e.g.
        after it was seen unhealthy: a delta op chained onto a prefix that
        is no longer there resolves to nothing.
        """
        self.num_tokens_at_server = 0

    def build_op_tokens(
        self,
        transport: TokenIdsTransport,
        start: int,
        end: int,
    ) -> tuple[bytes, int]:
        """Build the packed token ids for an op covering ``[start, end)``.

        Under ``DELTA`` this is just the op's own range, provided the server
        already holds everything before ``start`` for this request. When it
        does not -- no lookup has been submitted yet, or the connector reset
        the mark after the server was seen unhealthy -- the whole sequence
        is sent instead, which re-seeds the server for later ops.

        Recording ``end`` as reached assumes the op is delivered, the same
        assumption ``GetStoreMetadata`` already makes when it advances
        ``num_stored_tokens``.

        Args:
            transport: How much of the sequence to put on the op.
            start: Absolute start position of the op's range.
            end: Absolute end position of the op's range (exclusive).

        Returns:
            The packed token ids for the op and the absolute position of
            their first token.
        """
        if transport is TokenIdsTransport.DELTA and start <= self.num_tokens_at_server:
            token_bytes, token_offset = self.packed_token_slice(start, end), start
        else:
            token_bytes, token_offset = self.packed_token_ids(), 0
        self.note_tokens_at_server(
            max(end, token_offset + num_packed_tokens(token_bytes))
        )
        return token_bytes, token_offset

    ####
    # For debugging
    ####
    def __repr__(self) -> str:
        return (
            f"LMCacheMPRequestTracker(request_id={self.request_id}, "
            f"num_tokens={len(self.all_token_ids)}, "
            f"num_allocated_blocks="
            f"{self.num_allocated_blocks()}, "
            f"num_stored_tokens={self.num_stored_tokens}, "
            f"vllm_hit_tokens={self.num_vllm_hit_tokens}, "
            f"lmcache_hit_tokens={self.num_lmcache_hit_tokens}, "
            f"state={self.state})"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class LMCacheMPRequestMetadata:
    request_id: str
    direction: Literal["STORE", "RETRIEVE"]
    op: LoadStoreOp
    cache_salt: str = ""
    request_configs: dict[str, Any] | None = None

    @staticmethod
    def GetStoreMetadata(
        tracker: LMCacheMPRequestTracker,
        lmcache_tokens_per_chunk: int,
        group_tokens_per_block: list[int],
        transport: TokenIdsTransport = TokenIdsTransport.FULL,
    ) -> "LMCacheMPRequestMetadata | None":
        """
        Generate the store metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            lmcache_tokens_per_chunk: the number of tokens in a LMCache data chunk
            group_tokens_per_block: per-engine-group tokens covered by one
                paged chunk (one block ID) of that group, i.e. the group's
                KV cache spec ``block_size``. Must each divide
                ``lmcache_tokens_per_chunk`` (hybrid models can mix different values).
            transport: How much of the token sequence the op should carry.
        """
        num_engine_groups = len(group_tokens_per_block)
        # NOTE: the invariant here is that `num_stored_tokens` should
        # always be a multiple of `lmcache_tokens_per_chunk`
        # TODO: This should be checked every time we update the num_stored_tokens
        #
        # Why computed_tokens uses max(num_vllm_hit_tokens, num_lmcache_hit_tokens):
        #
        # Both values represent a prefix of tokens whose KV data is already
        # available (either from vLLM APC or from LMCache), so they must NOT
        # be summed (that would double-count the overlapping prefix).
        #
        # * num_lmcache_hit_tokens: LMCache-hit tokens are already counted in
        #   num_stored_tokens (set during lookup), so they must be included
        #   here to keep the upper bound consistent.  They are NOT re-stored.
        # * num_vllm_hit_tokens: LMCache stores in units of chunks, so
        #   num_lmcache_hit_tokens is rounded DOWN to the nearest chunk
        #   boundary.  When vLLM APC hits more tokens than that rounded value
        #   (e.g. APC=704 tokens, LMCache=512 tokens after chunk alignment),
        #   using only num_lmcache_hit_tokens would set the upper bound too
        #   low and silently skip the APC-hit tokens that fall between the
        #   two values, causing under-storing.  Taking the max ensures we
        #   always use the tighter (larger) of the two hit counts.
        computed_tokens = tracker.num_scheduled_tokens + max(
            tracker.num_vllm_hit_tokens, tracker.num_lmcache_hit_tokens
        )
        # Each group covers ``len(block_ids) * tokens_per_block`` tokens; the
        # storable prefix is bounded by the least-covered group (e.g.
        # gemma-4 sliding: one 32-token ID covers 2x the tokens of a
        # 16-token full-attention ID).
        allocated_lengths = tracker.num_allocated_blocks()
        allocated_tokens = (
            min(
                allocated_lengths.get(engine_group_idx, 0)
                * group_tokens_per_block[engine_group_idx]
                for engine_group_idx in range(num_engine_groups)
            )
            if num_engine_groups > 0
            else 0
        )
        min_available_tokens = min(
            len(tracker.all_token_ids),
            allocated_tokens,
            computed_tokens,
        )
        if tracker.max_offload_tokens is not None:
            min_available_tokens = min(min_available_tokens, tracker.max_offload_tokens)
        num_staging_tokens = min_available_tokens - tracker.num_stored_tokens
        num_chunks = num_staging_tokens // lmcache_tokens_per_chunk

        if num_chunks >= 1:
            start_token_idx = tracker.num_stored_tokens
            end_token_idx = start_token_idx + num_chunks * lmcache_tokens_per_chunk
            block_ids = slice_block_ids_per_group(
                tracker.allocated_block_ids,
                group_tokens_per_block,
                start_token_idx,
                end_token_idx,
            )
            token_bytes, token_offset = tracker.build_op_tokens(
                transport, start_token_idx, end_token_idx
            )
            op = LoadStoreOp(
                token_bytes=token_bytes,
                block_ids=block_ids,
                start=start_token_idx,
                end=end_token_idx,
                token_offset=token_offset,
            )

            ret = LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction="STORE",
                op=op,
                cache_salt=tracker.cache_salt,
                request_configs=tracker.request_configs,
            )

            # Update the request tracker
            tracker.increase_num_stored_tokens(end_token_idx - start_token_idx)
            return ret

        return None

    @staticmethod
    def GetRetrieveMetadata(
        tracker: LMCacheMPRequestTracker,
        lmcache_tokens_per_chunk: int,
        group_tokens_per_block: list[int],
        transport: TokenIdsTransport = TokenIdsTransport.FULL,
    ) -> "LMCacheMPRequestMetadata | None":
        """
        Generate the retrieve metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            lmcache_tokens_per_chunk: the number of tokens in a LMCache data chunk
            group_tokens_per_block: per-engine-group tokens covered by one
                paged chunk (one block ID) of that group, i.e. the group's
                KV cache spec ``block_size``. Must each divide
                ``lmcache_tokens_per_chunk`` (hybrid models can mix different values).
            transport: How much of the token sequence the op should carry.
        """
        if not tracker.is_ready_for_retrieving():
            return None

        # |---------------------|-----------------|----------------|
        # | num_vllm_hit_tokens |
        # | lmcache chunk 1   | lmcache chunk 2   |
        #                     |  need to retrieve |

        start_token_idx = (
            tracker.num_vllm_hit_tokens
            // lmcache_tokens_per_chunk
            * lmcache_tokens_per_chunk
        )
        end_token_idx = tracker.num_lmcache_hit_tokens
        assert end_token_idx % lmcache_tokens_per_chunk == 0, (
            "The number of LMCache hit tokens should be a multiple of the "
            "LMCache chunk size. "
        )
        assert len(tracker.all_token_ids) >= end_token_idx, (
            "The number of tokens should be greater than or equal to the "
            "number of LMCache hit tokens. "
        )
        if end_token_idx > start_token_idx:
            block_ids = slice_block_ids_per_group(
                tracker.allocated_block_ids,
                group_tokens_per_block,
                start_token_idx,
                end_token_idx,
            )
            token_bytes, token_offset = tracker.build_op_tokens(
                transport, start_token_idx, end_token_idx
            )

            # Compute how many tokens at the start of the retrieve range
            # overlap with APC-shared blocks. The server must skip writing
            # to these positions to avoid a cross-stream data race: the
            # retrieve writes on the LMCache CUDA stream while concurrent
            # requests may read these APC-shared blocks on the vLLM stream.
            skip_first_n_tokens = tracker.num_vllm_hit_tokens - start_token_idx

            op = LoadStoreOp(
                token_bytes=token_bytes,
                block_ids=block_ids,
                start=start_token_idx,
                end=end_token_idx,
                skip_first_n_tokens=skip_first_n_tokens,
                token_offset=token_offset,
            )

            ret = LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction="RETRIEVE",
                op=op,
                cache_salt=tracker.cache_salt,
                request_configs=tracker.request_configs,
            )
            return ret

        return None


class LMCacheMPConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        super().__init__()
        self.requests: list[LMCacheMPRequestMetadata] = []
        self.need_flush_before_forward: bool = False

    def add_request_metadata(self, request_metadata: LMCacheMPRequestMetadata):
        self.requests.append(request_metadata)

    def __len__(self):
        return len(self.requests)

    # For debugging
    def __str__(self):
        request_strs = []
        for req_meta in self.requests:
            request_strs.append(
                f"RequestMetadata(request_id={req_meta.request_id}, "
                f"direction={req_meta.direction}, "
                f"num_blocks={len(req_meta.op.flat_block_ids)}, "
                f"block_ids={req_meta.op.block_ids})"
            )
        return (
            f"need_flush_before_forward={self.need_flush_before_forward}; ["
            + "\n".join(request_strs)
            + "]"
        )

    def __repr__(self):
        return self.__str__()


@dataclass
class LMCacheMPWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed store events.

    Each worker reports {req_id: 1} for newly completed stores.
    ``aggregate()`` sums counts across workers within a step.
    The scheduler-side manager accumulates across steps and processes
    a store completion only when count reaches ``world_size``.
    """

    completed_store_requests: dict[str, int]

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, LMCacheMPWorkerMetadata)
        merged = dict(self.completed_store_requests)
        for k, v in other.completed_store_requests.items():
            merged[k] = merged.get(k, 0) + v
        return LMCacheMPWorkerMetadata(completed_store_requests=merged)
