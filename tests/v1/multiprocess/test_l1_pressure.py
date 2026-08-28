# SPDX-License-Identifier: Apache-2.0
"""Tests for the GET_L1_PRESSURE endpoint.

Covers the two layers separately:

1. Protocol registration -- the request type resolves to an empty payload
   and an ``L1PressureStats`` response.
2. Server handler -- ``ManagementModule.get_l1_pressure`` pairs the storage
   manager's usage tuple with its deletion totals.
"""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.controller import L1PressureStats

# ============================================================================
# Protocol registration
# ============================================================================


def test_protocol_registration():
    """GET_L1_PRESSURE takes no payload and returns L1PressureStats."""
    assert get_payload_classes(RequestType.GET_L1_PRESSURE) == []
    assert get_response_class(RequestType.GET_L1_PRESSURE) is L1PressureStats


# ============================================================================
# Server handler
# ============================================================================


def test_handler_pairs_usage_with_deletion_totals():
    """The handler snapshots usage and cumulative deletion totals."""
    module = ManagementModule.__new__(ManagementModule)
    ctx = MagicMock()
    ctx.storage_manager.get_l1_usage.return_value = (700, 1000)
    ctx.storage_manager.get_l1_deletion_totals.return_value = (4096, 3)
    module._ctx = ctx

    stats = module.get_l1_pressure()

    assert stats == L1PressureStats(
        total_bytes=1000,
        used_bytes=700,
        evicted_bytes_total=4096,
        evicted_chunks_total=3,
    )
