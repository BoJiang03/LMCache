# SPDX-License-Identifier: Apache-2.0
"""Build LMCache engine group infos from vLLM KV cache group metadata."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from functools import reduce
from typing import TYPE_CHECKING, Any
import math
import re

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.common.utils import LayoutHints

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


def gqa_kv_head_replicas(tp_size: int, total_num_kv_heads: int, dcp_size: int) -> int:
    """Return how many consecutive TP ranks hold the same GQA KV head.

    vLLM's ``QKVParallelLinear`` gives each rank one KV head once
    ``tp_size >= total_num_kv_heads``, so rank ``r`` holds head
    ``r // (tp_size // total_num_kv_heads)`` and that many consecutive ranks
    hold identical KV. Under DCP the replicas hold token shards instead, which
    this count does not describe, so it is ``1`` there.

    Args:
        tp_size: Tensor-parallel size of the engine.
        total_num_kv_heads: The model's KV head count before TP sharding.
        dcp_size: Decode-context-parallel size.

    Returns:
        The replica count, ``1`` when every rank holds distinct heads.
    """
    if dcp_size > 1 or total_num_kv_heads >= tp_size:
        return 1
    return tp_size // total_num_kv_heads


# KV head count fields of a text config, most specific first. A model whose
# sliding-window and full-attention layers hold different head counts names the
# odd one out (MiMo-V2 ``swa_num_key_value_heads``, Gemma 4
# ``num_global_key_value_heads``); every other layer reads ``num_key_value_heads``.
_SLIDING_KV_HEAD_FIELDS = (
    "swa_num_key_value_heads",
    "sliding_window_num_key_value_heads",
    "num_key_value_heads",
)
_FULL_KV_HEAD_FIELDS = ("num_global_key_value_heads", "num_key_value_heads")
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _config_kv_heads(text_config: Any, sliding: bool) -> int | None:
    """Return the total KV head count a text config declares for one layer type.

    Args:
        text_config: A Hugging Face text config, or ``None``.
        sliding: Whether the layer is a sliding-window attention layer.

    Returns:
        The first positive head count among the fields for that layer type, or
        ``None`` when the config declares none.
    """
    if text_config is None:
        return None
    fields = _SLIDING_KV_HEAD_FIELDS if sliding else _FULL_KV_HEAD_FIELDS
    for field in fields:
        value = getattr(text_config, field, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def resolve_gqa_replicas_by_layer(
    kv_cache_config: Any,
    target_text_config: Any,
    draft_text_config: Any,
    tp_size: int,
    dcp_size: int,
) -> dict[str, int]:
    """Resolve, per attention layer, how many consecutive TP ranks share its KV head.

    The model-wide KV head count describes only the layers that hold it. A
    hybrid such as MiMo-V2 gives its full-attention layers 4 KV heads and its
    sliding-window layers 8, so at TP8 the former put one head on two ranks
    while the latter put a distinct head on every rank. Reading each layer's
    own head count keeps the distinct heads from being folded into one object.

    A layer whose index lies past the target model's layers belongs to the
    speculative draft and reads the draft's text config. A layer whose head
    count cannot be resolved maps to ``1``: storing it per rank costs space,
    folding distinct heads would serve one rank's KV to another.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` (or ``None``).
        target_text_config: The target model's Hugging Face text config.
        draft_text_config: The draft model's text config, or ``None``.
        tp_size: Tensor-parallel size of the engine.
        dcp_size: Decode-context-parallel size.

    Returns:
        A mapping from layer name to replica count for every layer of every
        KV cache group.
    """
    replicas: dict[str, int] = {}
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or ()
    num_target_layers = getattr(target_text_config, "num_hidden_layers", None)
    for group in groups:
        spec = getattr(group, "kv_cache_spec", None)
        per_layer_specs = getattr(spec, "kv_cache_specs", None)
        for name in group.layer_names:
            layer_spec = per_layer_specs[name] if per_layer_specs else spec
            match = _LAYER_INDEX_RE.search(name)
            is_draft = (
                draft_text_config is not None
                and match is not None
                and isinstance(num_target_layers, int)
                and int(match.group(1)) >= num_target_layers
            )
            heads = _config_kv_heads(
                draft_text_config if is_draft else target_text_config,
                _is_sliding_window_spec(layer_spec),
            )
            replicas[name] = (
                1 if heads is None else gqa_kv_head_replicas(tp_size, heads, dcp_size)
            )
    return replicas


def _is_attention_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM attention spec.

    Checked by class name so this module stays importable without vLLM.
    ``UniformTypeKVCacheSpecs`` is unwrapped first (same-typed layers, one
    leaf suffices); it does not derive from ``AttentionSpec`` itself.
    """
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        spec = next(iter(inner.values()))
    return any(cls.__name__ == "AttentionSpec" for cls in type(spec).__mro__)


def _is_mla_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM MLA spec (one latent head,
    held in full on every TP rank). Checked by class name like
    :func:`_is_attention_spec`."""
    return any(cls.__name__ == "MLAAttentionSpec" for cls in type(spec).__mro__)


def is_scratch_spec(spec: Any) -> bool:
    """Return whether vLLM marks the spec as non-prefix-cacheable.

    vLLM never hashes or restores the blocks of such a scratch group, so they
    carry no token range LMCache could store (e.g. the per-request rings
    ``CircularBufferSpec`` and ``KpoolTailSpec``). ``prefix_cacheable`` takes
    precedence over the older ``participates_in_prefix_caching`` marker.
    Specs with neither marker are prefix-cacheable.
    ``UniformTypeKVCacheSpecs`` is unwrapped first.

    Args:
        spec: A vLLM KV cache spec, or a ``UniformTypeKVCacheSpecs`` container.

    Returns:
        ``True`` for a scratch spec, ``False`` for a prefix-cacheable one.
    """
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        spec = next(iter(inner.values()))
    if hasattr(spec, "prefix_cacheable"):
        return not spec.prefix_cacheable
    return not getattr(spec, "participates_in_prefix_caching", True)


def is_per_request_state_spec(spec: Any) -> bool:
    """Return whether the spec is a per-request ring.

    vLLM's ``CircularBufferSpec`` (``KpoolTailSpec`` on older vLLM) gives each
    request one block for its whole lifetime, written at ``position % capacity``
    and holding the raw keys of the compression group that is still open. It
    is a scratch spec (see :func:`is_scratch_spec`), but unlike the other
    scratch specs its content is needed to resume a request mid-group, e.g.
    from an unfull chunk. Checked by class name so this module stays
    importable without vLLM; ``UniformTypeKVCacheSpecs`` is unwrapped first.

    Args:
        spec: A vLLM KV cache spec, or a ``UniformTypeKVCacheSpecs`` container.

    Returns:
        ``True`` for a per-request ring spec.
    """
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        spec = next(iter(inner.values()))
    return is_scratch_spec(spec) and any(
        cls.__name__ in ("CircularBufferSpec", "KpoolTailSpec")
        for cls in type(spec).__mro__
    )


def get_tokens_per_block(kv_cache_spec: Any, dcp_size: int) -> int:
    """Global tokens covered by one block id of ``kv_cache_spec``.

    Attention blocks span ``block_size * dcp_size`` tokens under DCP
    (vLLM's ``resolve_kv_cache_block_sizes`` rule); recurrent state is
    replicated, not sharded, and stays at ``block_size``. Scratch specs
    (see :func:`is_scratch_spec`) cover no tokens and yield ``0``.
    """
    if is_scratch_spec(kv_cache_spec):
        return 0
    block_size = kv_cache_spec.block_size
    if dcp_size <= 1:
        return block_size
    if _is_attention_spec(kv_cache_spec):
        return block_size * dcp_size
    return block_size


def is_align_mode_mamba_spec(spec: Any) -> bool:
    """Return whether the spec is a Mamba/linear-attention spec in align mode.

    Align mode keeps state snapshots only at chosen boundaries and may move
    them between blocks, so a block-table position does not identify one;
    vLLM hands each snapshot's block to connectors instead
    (``SchedulerOutput.kv_connector_block_state.boundary_state_offloads``).
    Checked by class name so this module stays importable without vLLM.

    Args:
        spec: A vLLM KV cache spec.

    Returns:
        True for a ``MambaSpec`` with ``mamba_cache_mode == "align"``.
    """
    return (
        any(cls.__name__ == "MambaSpec" for cls in type(spec).__mro__)
        and getattr(spec, "mamba_cache_mode", "none") == "align"
    )


def _is_sliding_window_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM sliding-window spec.

    Checked by class name so this module stays importable without vLLM.
    Subclasses such as ``SlidingWindowMLASpec`` count.
    """
    return any(cls.__name__ == "SlidingWindowSpec" for cls in type(spec).__mro__)


def _is_cachable_mamba_spec(spec: Any) -> bool:
    """Return whether the spec is a snapshotting Mamba/linear-attention spec.

    Align mode snapshots only the last block; all mode snapshots every block
    boundary. Either way a restore consumes only the last matched block's
    page, so both behave like a cross-chunk sliding window of one block.
    Checked by class name (like :func:`_is_sliding_window_spec`) so this
    module stays importable without vLLM.
    """
    return any(cls.__name__ == "MambaSpec" for cls in type(spec).__mro__) and getattr(
        spec, "mamba_cache_mode", "none"
    ) in ("align", "all")


def _resolve_per_layer_sw_sizes(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
) -> list[int]:
    """Resolve the sliding window size in tokens for each registered KV tensor.

    Will resolve -1 for non-sliding-window layers.

    Args:
        vllm_groups: vLLM ``KVCacheGroupSpec`` instances.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.

    Returns:
        A list of length ``num_layers`` mapping each registered tensor index
        to its cross-chunk window size in tokens: the sliding window for
        sliding-window attention, the engine-side block size for align-mode
        Mamba/linear layers (a one-block window), or ``-1`` for full-attention
        layers.
    """
    per_layer_sw_size = [-1] * num_layers
    for group in vllm_groups:
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            continue
        # ``UniformTypeKVCacheSpecs`` carries per-layer specs in
        # ``kv_cache_specs``; other specs apply to all of the group's layers.
        per_layer_specs = getattr(spec, "kv_cache_specs", None)
        for name in group.layer_names:
            layer_spec = per_layer_specs[name] if per_layer_specs else spec
            if _is_sliding_window_spec(layer_spec):
                per_layer_sw_size[layer_to_idx[name]] = layer_spec.sliding_window
            elif _is_cachable_mamba_spec(layer_spec):
                per_layer_sw_size[layer_to_idx[name]] = layer_spec.block_size
    return per_layer_sw_size


#: Reserved layer-name prefix for CacheBlend fused-aux page pools:
#: ``cb.aux_pool.<tokens_per_block>[.<label>]`` — the first suffix part is
#: the logical block size; the optional label distinguishes multiple pools
#: sharing one block size (they then share an engine group).
CB_AUX_POOL_LAYER_PREFIX = "cb.aux_pool."


def cb_aux_pool_entries(kv_caches) -> "list[tuple[str, int]]":
    """CacheBlend fused-aux pool entries among the registered tensors.

    Presence-gated: any layer name starting with
    :data:`CB_AUX_POOL_LAYER_PREFIX` is a connector-owned aux page pool
    served by a synthetic engine group (one group per block size).

    Args:
        kv_caches: Registered tensors keyed by layer name.

    Returns:
        ``(layer_name, tokens_per_block)`` per aux pool, in registration
        order; empty for models without one.

    Raises:
        ValueError: If a marker name's block-size suffix does not parse.
    """
    entries: list[tuple[str, int]] = []
    for name in kv_caches:
        if not name.startswith(CB_AUX_POOL_LAYER_PREFIX):
            continue
        suffix = name[len(CB_AUX_POOL_LAYER_PREFIX) :].split(".", 1)[0]
        try:
            tokens_per_block = int(suffix)
        except ValueError as exc:
            raise ValueError(
                f"aux pool layer name {name!r}: block-size suffix "
                f"{suffix!r} is not an integer"
            ) from exc
        if tokens_per_block <= 0:
            raise ValueError(
                f"aux pool layer name {name!r}: tokens_per_block must be "
                f"positive, got {tokens_per_block}"
            )
        entries.append((name, tokens_per_block))
    return entries


def _resolve_per_layer_recurrent(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
) -> list[bool]:
    """Resolve whether each registered KV tensor holds recurrent state pages.

    Args:
        vllm_groups: vLLM ``KVCacheGroupSpec`` instances.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.

    Returns:
        A list of length ``num_layers``: ``True`` for Mamba/linear-attention
        layers in a snapshotting cache mode (see :func:`_is_cachable_mamba_spec`),
        ``False`` for attention layers.
    """
    per_layer_recurrent = [False] * num_layers
    for group in vllm_groups:
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            continue
        per_layer_specs = getattr(spec, "kv_cache_specs", None)
        for name in group.layer_names:
            layer_spec = per_layer_specs[name] if per_layer_specs else spec
            if _is_cachable_mamba_spec(layer_spec):
                per_layer_recurrent[layer_to_idx[name]] = True
    return per_layer_recurrent


def _resolve_per_layer_tp_replicas(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
    kv_tp_size: int,
    gqa_replicas: int,
    gqa_replicas_by_layer: Mapping[str, int] | None = None,
) -> list[int]:
    """Resolve how many consecutive kv ranks hold identical KV for each layer.

    A layer whose spec declares ``max_tp_shards == 1``, or an MLA spec that
    leaves it unset, is unsharded: every rank of the TP group holds the same
    KV (or the same token shard of it under DCP), so its count is
    ``kv_tp_size``. A GQA attention layer with one KV head per rank holds
    the head ``gqa_replicas`` consecutive ranks share (see
    :func:`gqa_kv_head_replicas`); the count is capped to what one server's
    rank block can fold. Every other layer counts ``1``.

    Args:
        vllm_groups: vLLM ``KVCacheGroupSpec`` instances.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.
        kv_tp_size: Tensor-parallel ranks attached to one LMCache server.
        gqa_replicas: Ranks sharing one GQA KV head across the whole TP group.
        gqa_replicas_by_layer: Per-layer replica counts from
            :func:`resolve_gqa_replicas_by_layer`; when given they replace
            ``gqa_replicas``, and a layer missing from it counts ``1``.

    Returns:
        A list of length ``num_layers`` with each layer's replica count.

    Note:
        ``gqa_replicas`` comes from the model-wide KV head count, so a model
        whose attention layers have different KV head counts is described
        correctly only for the layers matching that count; pass
        ``gqa_replicas_by_layer`` for such a model.
    """
    per_layer_tp_replicas = [1] * num_layers
    gqa_server_replicas = math.gcd(gqa_replicas, kv_tp_size)
    for group in vllm_groups:
        spec = getattr(group, "kv_cache_spec", None)
        per_layer_specs = getattr(spec, "kv_cache_specs", None)
        for name in group.layer_names:
            layer_spec = per_layer_specs[name] if per_layer_specs else spec
            max_tp_shards = getattr(layer_spec, "max_tp_shards", None)
            if max_tp_shards == 1 or (
                max_tp_shards is None and _is_mla_spec(layer_spec)
            ):
                per_layer_tp_replicas[layer_to_idx[name]] = kv_tp_size
            elif (
                max_tp_shards is None
                and _is_attention_spec(layer_spec)
                and getattr(layer_spec, "num_kv_heads", None) == 1
            ):
                per_layer_tp_replicas[layer_to_idx[name]] = (
                    gqa_server_replicas
                    if gqa_replicas_by_layer is None
                    else math.gcd(gqa_replicas_by_layer.get(name, 1), kv_tp_size)
                )
    return per_layer_tp_replicas


def _merge_layer_recurrent(per_layer_recurrent: list[bool], indices: list[int]) -> bool:
    """Merge the per-layer recurrent-state flags of one LMCache group.

    Args:
        per_layer_recurrent: Recurrent-state flag per registered tensor index.
        indices: Registered tensor indices of the group's layers.

    Returns:
        The group's common flag.

    Raises:
        ValueError: If the group mixes recurrent and attention layers (vLLM
            groups layers by KV cache spec, so a mix indicates inconsistent
            metadata).
    """
    flags = {per_layer_recurrent[idx] for idx in indices}
    if len(flags) != 1:
        raise ValueError(
            f"Layers with indices {indices} mix recurrent-state and attention "
            "layers in one group. This should not happen because vLLM only "
            "groups layers with the same KV cache spec."
        )
    return flags.pop()


def _merge_layer_sw_sizes(per_layer_sw_size: list[int], indices: list[int]) -> int:
    """Merge the per-layer sliding window sizes of one LMCache group.

    Args:
        per_layer_sw_size: Sliding window size per registered tensor index.
        indices: Registered tensor indices of the group's layers.

    Returns:
        The group's common sliding window size in tokens, or ``-1`` when the
        layers are not sliding-window attention.

    Raises:
        ValueError: If the layers have different non-negative sliding window sizes.
    """
    sw_sizes = {per_layer_sw_size[idx] for idx in indices}
    if len(sw_sizes) != 1:
        raise ValueError(
            f"Layers with indices {indices} have different sliding window sizes "
            f"{sw_sizes}, but they are in the same group. This should "
            "not happen because vLLM should only group layers with the same "
            "KV cache spec, but got inconsistent metadata or registered tensors."
        )
    return sw_sizes.pop()


def create_engine_group_infos_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    layout_hints: "LayoutHints | None" = None,
    dcp_size: int = 1,
    kv_tp_size: int = 1,
    gqa_replicas: int = 1,
    gqa_replicas_by_layer: Mapping[str, int] | None = None,
) -> list[EngineGroupInfo]:
    """Build the LMCache engine group infos from vLLM metadata and registered tensors.

    This is the single entry point for the vLLM -> LMCache conversion. It reads
    the vLLM-specific fields (``KVCacheConfig.kv_cache_groups`` and
    ``KVCacheGroupSpec.layer_names`` from the v1 KV cache interface), maps each
    engine KV cache group's layer names to registered tensor indices, then
    splits the layers by physical transfer identity using the real tensors (via
    the shared :func:`lmcache.v1.kv_layer_groups.group_layers_by_identity`).
    vLLM-specific field access is intentionally confined to this function.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` describing the engine KV cache
            groups (or ``None`` / no groups, which yields a single-group spec).
        kv_caches: Registered KV tensors keyed by layer name, in registration
            order. Keys provide the layer-name -> tensor-index mapping; values
            are inspected for physical shape and dtype.
        layout_hints: Optional engine-provided layout hints forwarded to format
            detection (e.g. ``NHD``/``HND`` and compression metadata).
        dcp_size: Decode context parallel size.
        kv_tp_size: Tensor-parallel ranks attached to one LMCache server. A
            group vLLM replicates on every TP rank (``max_tp_shards == 1``, or
            an MLA spec that leaves it unset) is tagged
            ``EngineGroupInfo.tp_replicas = kv_tp_size`` and
            ``dcp_shards = dcp_size``; every other group stays ``(1, 1)``.
        gqa_replicas: Consecutive TP ranks sharing one GQA KV head, from
            :func:`gqa_kv_head_replicas`. A group of one-head-per-rank GQA
            layers is tagged with it (capped to ``kv_tp_size``).
        gqa_replicas_by_layer: Per-layer replica counts from
            :func:`resolve_gqa_replicas_by_layer`, replacing ``gqa_replicas``
            for models whose layers hold different KV head counts.

    Note:
        Under DCP each attention group's ``tokens_per_block`` is scaled by
        ``dcp_size`` to stay in the scheduler's coordinate space; its ratio
        to the physical slot count is what sizes each rank's memory object.
        Mamba groups are replicated per rank and stay unscaled. Layers of
        scratch groups (see :func:`is_scratch_spec`) are excluded and never
        form an info.

    Returns:
        The list of ``EngineGroupInfo`` in protocol order, i.e. the LMCache group
        order used by store/retrieve block IDs.
    """
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.common.utils import (
        normalize_and_discover_per_layer_formats,
    )
    from lmcache.v1.kv_layer_groups import (
        EXCLUDED_ENGINE_GROUP,
        group_layers_by_identity,
    )

    # vLLM-specific field access (confined to this function): map each
    # registered KV tensor to its vLLM engine KV cache group index. vLLM places
    # every registered layer in exactly one group; layers in different groups
    # have disjoint block-id spaces and must not share an LMCache group. ``None``
    # means a single (non-hybrid) group, i.e. every layer shares one block-id
    # space.
    per_layer_discoverable_kv_caches = list(kv_caches.values())
    layer_to_idx = {name: idx for idx, name in enumerate(kv_caches.keys())}
    vllm_groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )

    # Scratch groups skip format discovery: their layers are excluded below and
    # never transferred, so a layout the transfer kernels reject must not fail
    # registration.
    layer_index_groups = [
        [layer_to_idx[name] for name in group.layer_names]
        for group in vllm_groups
        if not is_scratch_spec(group.kv_cache_spec)
    ]

    # CacheBlend fused-aux (presence-gated): the pool joins detection as
    # its own group so its rank-3 layout is classified independently.
    aux_entries = cb_aux_pool_entries(kv_caches)
    layer_index_groups += [[layer_to_idx[name]] for name, _ in aux_entries]
    normalized_kv_caches, engine_kv_formats = normalize_and_discover_per_layer_formats(
        per_layer_discoverable_kv_caches,
        layer_index_groups,
        EngineType.VLLM,
        layout_hints,
    )
    num_layers = len(engine_kv_formats)
    # Layers absent from every engine group's ``layer_names`` are cross-layer
    # KV-sharing layers (e.g. google/gemma-4-E4B-it): vLLM aliases them to a
    # target owner's KV tensor, so the owner's group already covers them. Tag
    # them EXCLUDED_ENGINE_GROUP so they form no group of their own (a
    # wrong-block-size group would corrupt the per-group block-id counts).
    # Scratch groups (tokens_per_block 0) are excluded the same way.
    per_layer_group_idx: list[int] | None = None
    group_tokens_per_block: dict[int, int] = {}
    per_layer_sw_size = [-1] * num_layers
    per_layer_recurrent = [False] * num_layers
    per_layer_tp_replicas = [1] * num_layers
    if vllm_groups:
        per_layer_group_idx = [EXCLUDED_ENGINE_GROUP] * num_layers
        for engine_group_id, group in enumerate(vllm_groups):
            # The spec's block_size is the logical tokens covered by one of
            # this group's paged chunks (block IDs); the physical slot count
            # per chunk is discovered later from the registered tensors.
            # Under DCP the two diverge (see get_tokens_per_block).
            group_tokens_per_block[engine_group_id] = get_tokens_per_block(
                group.kv_cache_spec, dcp_size
            )
            if group_tokens_per_block[engine_group_id] == 0:
                continue
            for name in group.layer_names:
                per_layer_group_idx[layer_to_idx[name]] = engine_group_id
        per_layer_sw_size = _resolve_per_layer_sw_sizes(
            vllm_groups, layer_to_idx, num_layers
        )
        per_layer_recurrent = _resolve_per_layer_recurrent(
            vllm_groups, layer_to_idx, num_layers
        )
        per_layer_tp_replicas = _resolve_per_layer_tp_replicas(
            vllm_groups,
            layer_to_idx,
            num_layers,
            kv_tp_size,
            gqa_replicas,
            gqa_replicas_by_layer,
        )

    # Aux pools form synthetic engine groups after the vLLM groups (an
    # unassigned marker layer would fall to EXCLUDED_ENGINE_GROUP and never
    # store), bucketed by tokens_per_block: pools sharing a block size share
    # one engine group — and thereby one kernel group when their tensor
    # identities also match. Tags are 1-based per bucket.
    aux_group_tags: dict[int, int] = {}
    if aux_entries:
        if per_layer_group_idx is None:
            # Non-hybrid engine config: all real layers share group 0.
            per_layer_group_idx = [0] * num_layers
        next_group_id = len(vllm_groups) if vllm_groups else 1
        group_by_tpb: dict[int, int] = {}
        for name, tokens_per_block in aux_entries:
            group_id = group_by_tpb.get(tokens_per_block)
            if group_id is None:
                group_id = next_group_id
                next_group_id += 1
                group_by_tpb[tokens_per_block] = group_id
                group_tokens_per_block[group_id] = tokens_per_block
                aux_group_tags[group_id] = len(group_by_tpb)
            per_layer_group_idx[layer_to_idx[name]] = group_id

    # Within one vLLM engine group, layers can have different hidden dimensions
    # (e.g. a different head count), which require different GPU copy kernels.
    # ``group_layers_by_identity`` splits each engine group further by physical
    # transfer identity (kv_size, num_heads, head_size, block_size,
    # engine_group_idx, dtype, engine_kv_format, block_stride_elems), so every
    # resulting LMCache group can be served by a single copy kernel. It is the
    # shared, engine-neutral primitive the server reuses to reproduce the same
    # grouping from the registered tensors.
    infos: list[EngineGroupInfo] = []
    for identity, indices in group_layers_by_identity(
        normalized_kv_caches,
        engine_kv_formats,
        per_layer_group_idx,
    ):
        # A group's object is shared only by ranks that agree on every layer
        # in it: one sharded layer makes the object differ per rank.
        tp_replicas = reduce(math.gcd, (per_layer_tp_replicas[i] for i in indices))
        infos.append(
            EngineGroupInfo(
                engine_group_id=identity.engine_group_idx,
                layer_indices=tuple(indices),
                tokens_per_block=group_tokens_per_block.get(
                    identity.engine_group_idx, 0
                ),
                sw_size_tokens=_merge_layer_sw_sizes(per_layer_sw_size, indices),
                # Connector-private pools bucket by tag under
                # --separate-object-groups, after the regular groups.
                extra_object_group_tag=aux_group_tags.get(identity.engine_group_idx, 0),
                recurrent_state=_merge_layer_recurrent(per_layer_recurrent, indices),
                tp_replicas=tp_replicas,
                # DCP splits a replicated group's tokens across its replicas.
                dcp_shards=dcp_size if tp_replicas > 1 else 1,
            )
        )
    return infos
