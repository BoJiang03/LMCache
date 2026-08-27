# SPDX-License-Identifier: Apache-2.0
"""
Controller protocol definitions for cache management and configuration.

This module defines the protocol for:
- CLEAR: Clear all caches in the server
- GET_CHUNK_SIZE: Get the chunk size configuration from the server
- GET_EXPERIMENTAL: Get the enabled experimental intermediate tensor transfer
- GET_L1_PRESSURE: Get an L1 capacity/eviction snapshot for rate estimation
"""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition


@dataclass
class L1PressureStats:
    """Snapshot of L1 capacity and cumulative eviction, for GET_L1_PRESSURE.

    Rate computation is the caller's job: two snapshots and their arrival
    times give an eviction byte rate, and ``total_bytes`` over that rate is
    the L1 residence time. The cumulative counters cover every key deletion
    (watermark eviction, store-failure cleanup, CLEAR), because any deletion
    shortens effective residence.
    """

    total_bytes: int
    """L1 capacity in bytes."""

    used_bytes: int
    """Bytes currently resident in L1."""

    evicted_bytes_total: int
    """Cumulative bytes freed by key deletion since server start."""

    evicted_chunks_total: int
    """Cumulative objects freed by key deletion since server start."""


# Define request names for this protocol group
REQUEST_NAMES = [
    "CLEAR",
    "GET_CHUNK_SIZE",
    "GET_EXPERIMENTAL",
    "PING",
    "GET_L1_PRESSURE",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for controller operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Clear all caches
        # Payload: None
        # Returns: None
        "CLEAR": ProtocolDefinition(
            payload_classes=[],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Get chunk size configuration
        # Payload: None
        # Returns: int - The chunk size value
        "GET_CHUNK_SIZE": ProtocolDefinition(
            payload_classes=[],
            response_class=int,
            handler_type=HandlerType.SYNC,
        ),
        # Ping
        # Payload: [instance_id] -- the sender's worker instance ID, or None
        #   for an untracked prober (the scheduler adapter).
        # Returns: bool - Always True
        # BLOCKING on the NORMAL pool: keeps PING off the MQ main loop (where a
        # slow SYNC REGISTER_KV_CACHE would stall it) and lets pool saturation
        # surface as worker degraded mode.
        "PING": ProtocolDefinition(
            payload_classes=[int | None],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        # Get the enabled experimental intermediate tensor transfer types
        # Payload: None
        # Returns: list[str]: the experimental intermediate tensor transfer
        # types the server was launched with (empty when none are enabled)
        "GET_EXPERIMENTAL": ProtocolDefinition(
            payload_classes=[],
            response_class=list[str],
            handler_type=HandlerType.SYNC,
        ),
        # Get an L1 capacity/eviction snapshot
        # Payload: None
        # Returns: L1PressureStats - capacity, usage, cumulative eviction
        "GET_L1_PRESSURE": ProtocolDefinition(
            payload_classes=[],
            response_class=L1PressureStats,
            handler_type=HandlerType.SYNC,
        ),
    }
