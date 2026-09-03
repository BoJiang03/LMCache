# SPDX-License-Identifier: Apache-2.0
"""A KV connector that does nothing, to bisect the ~9% connector tax.

WHY THIS EXISTS
---------------
Measured on this box, with `max_num_batched_tokens=8192` in every arm so a
forward step is a fixed number of tokens:

    no connector      85.3 ms/step
    LMCache MP        91.0 ms/step      +5.7
    LMCache IP        97.5 ms/step     +12.2

The +5.7 ms is paid by BOTH connectors, at both KV pool sizes, with
`Deferred == 0` for MP -- so it is the cost of *having a connector attached*,
not of anything LMCache is asked to store or fetch. Five candidate mechanisms
have been eliminated from logs (cudagraph mode, the store/D2H path, the prefix
hash chain, idle time, scheduler config) and a py-spy profile is not usable
here: with TP=8 the workers are lockstepped on NCCL collectives, so stopping any
one rank to read its stack stalls all eight, and sampling the tree at 30 Hz slowed
the run 3.75x.

So bisect instead of profile. This connector implements every abstract method of
KVConnectorBase_V1 and does nothing in all of them. vLLM still walks its entire
connector code path: `maybe_transfer_kv_layer` wraps all 36 attention layers,
`build_connector_meta` runs every scheduler step, the worker-side load/save hooks
fire every forward, the KVOutputAggregator is installed, and
`get_num_new_matched_tokens` is consulted for every waiting request.

    null connector ~= +5.7 ms/step  ->  the tax is vLLM's own connector plumbing,
                                        and LMCache is not the thing to fix.
    null connector ~= +0.0 ms/step  ->  the tax is inside LMCache, and the
                                        plumbing is free.

Either answer halves the search space with no profiler and no distortion, using
the ms/step measurement that reproduces to 0.1 ms across 14 blocks.

It deliberately does NOT subclass SupportsHMA, so it needs
--disable-hybrid-kv-cache-manager and lands at pool 13,724,416, the same as 1c
(no connector), 1e (MP) and 1b/1g (IP).
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class NullConnectorMetadata(KVConnectorMetadata):
    """Empty per-step metadata. vLLM still builds, ships and binds one."""


class NullConnector(KVConnectorBase_V1):
    """Implements the full KVConnectorBase_V1 surface and does nothing."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        logger.info("NullConnector attached (role=%s). It does nothing.", role)

    # ---- worker side -------------------------------------------------------
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        return

    # ---- scheduler side ----------------------------------------------------
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # (0 external tokens, not loading asynchronously).  Returning None here
        # would defer the request, which is the behaviour under test elsewhere;
        # this arm must never defer.
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        return

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        return NullConnectorMetadata()
