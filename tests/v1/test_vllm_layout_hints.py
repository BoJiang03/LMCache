# SPDX-License-Identifier: Apache-2.0
"""KV cache layout discovery across vLLM generations.

vLLM before 0.28 exposes the layout as a process global through
``get_kv_cache_layout``; from 0.28 that function is gone and the engine core
records the resolved layout on ``CacheConfig.kv_cache_layout`` under a stride
permutation name. These tests pin both paths without importing vLLM.
"""

# Standard
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
import sys

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import utils as vllm_utils

_VLLM_UTILS_MODULE = "vllm.v1.attention.backends.utils"


@dataclass
class MockCacheConfig:
    """Stand-in for the fields of vLLM's ``CacheConfig`` read here."""

    kv_cache_layout: str | None


def _make_vllm_config(layout_name: str | None) -> SimpleNamespace:
    """Build a vLLM config double carrying only ``cache_config``."""
    return SimpleNamespace(cache_config=MockCacheConfig(kv_cache_layout=layout_name))


@pytest.fixture(autouse=True)
def reset_layout_memo(monkeypatch):
    """Clear the process-wide memo so tests do not leak into each other."""
    monkeypatch.setattr(vllm_utils, "_kv_cache_layout", None, raising=False)


@pytest.fixture
def vllm_without_layout_getter(monkeypatch):
    """Simulate vLLM >= 0.28, where ``get_kv_cache_layout`` no longer exists."""
    module = ModuleType(_VLLM_UTILS_MODULE)
    monkeypatch.setitem(sys.modules, _VLLM_UTILS_MODULE, module)


@pytest.fixture
def vllm_with_layout_getter(monkeypatch):
    """Simulate vLLM < 0.28, where the layout is a process global."""
    module = ModuleType(_VLLM_UTILS_MODULE)
    module.get_kv_cache_layout = lambda: "HND"
    monkeypatch.setitem(sys.modules, _VLLM_UTILS_MODULE, module)


def test_old_vllm_reads_the_process_global(vllm_with_layout_getter):
    """Before 0.28 the layout comes from vLLM and the config is not consulted."""
    assert vllm_utils.try_get_vllm_kv_cache_layout(_make_vllm_config("LBNHC")) == "HND"


@pytest.mark.parametrize(
    ("layout_name", "expected"),
    [("LBNHC", "NHD"), ("LBHNC", "HND")],
)
def test_new_vllm_maps_permutation_names(
    vllm_without_layout_getter, layout_name, expected
):
    """The two layer-outermost block-compact permutations are the legacy names."""
    config = _make_vllm_config(layout_name)
    assert vllm_utils.try_get_vllm_kv_cache_layout(config) == expected
    assert vllm_utils.vllm_layout_hints(config) == {"kv_layout": expected}


@pytest.mark.parametrize("layout_name", ["LHBNC", "BLHNC", "BHLNC"])
def test_new_vllm_rejects_inexpressible_permutations(
    vllm_without_layout_getter, layout_name
):
    """A permutation NHD/HND cannot describe yields no hint rather than a wrong one."""
    config = _make_vllm_config(layout_name)
    assert vllm_utils.try_get_vllm_kv_cache_layout(config) is None
    assert vllm_utils.vllm_layout_hints(config) == {}


def test_new_vllm_unresolved_layout_yields_no_hint(vllm_without_layout_getter):
    """Before the engine core resolves a layout there is nothing to report."""
    assert vllm_utils.try_get_vllm_kv_cache_layout(_make_vllm_config(None)) is None


def test_new_vllm_memoizes_for_call_sites_without_a_config(vllm_without_layout_getter):
    """Call sites holding no config reuse the layout an earlier call resolved."""
    assert vllm_utils.try_get_vllm_kv_cache_layout() is None

    vllm_utils.try_get_vllm_kv_cache_layout(_make_vllm_config("LBHNC"))

    assert vllm_utils.try_get_vllm_kv_cache_layout() == "HND"
    assert vllm_utils.vllm_layout_hints() == {"kv_layout": "HND"}


def test_missing_vllm_yields_no_hint(monkeypatch):
    """Without vLLM and without a config there is no layout to report."""
    monkeypatch.setitem(sys.modules, _VLLM_UTILS_MODULE, None)
    assert vllm_utils.try_get_vllm_kv_cache_layout() is None
    assert vllm_utils.vllm_layout_hints() == {}
