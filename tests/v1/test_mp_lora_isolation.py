# SPDX-License-Identifier: Apache-2.0
"""LoRA adapters must not share KV cache entries in MP mode."""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.utils import (
    LORA_TAG_KEY,
    extract_request_configs_from_request,
)
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey


def _request(
    lora_name: str | None, kv_transfer_params: dict[str, Any] | None = None
) -> Any:
    lora_request = None if lora_name is None else SimpleNamespace(lora_name=lora_name)
    extra_args = (
        None
        if kv_transfer_params is None
        else {"kv_transfer_params": kv_transfer_params}
    )
    return SimpleNamespace(
        lora_request=lora_request,
        sampling_params=SimpleNamespace(extra_args=extra_args),
    )


def _object_keys(request: Any) -> list[ObjectKey]:
    ipc_key = IPCCacheServerKey.from_token_ids(
        model_name="base-model",
        world_size=1,
        worker_id=0,
        token_ids=[1, 2, 3, 4],
        end=4,
        request_configs=extract_request_configs_from_request(request),
    )
    return ipc_key_to_object_keys(ipc_key, [b"same-chunk-hash"], [0])[0]


def test_lora_request_is_tagged_with_adapter_name() -> None:
    configs = extract_request_configs_from_request(_request("sql-adapter"))
    assert configs == {LORA_TAG_KEY: "sql-adapter"}


def test_base_request_has_no_lora_tag() -> None:
    assert extract_request_configs_from_request(_request(None)) is None


@pytest.mark.parametrize("lora_name", [None, "own-adapter"])
def test_client_cannot_set_lora_tag(lora_name: str | None) -> None:
    params = {LORA_TAG_KEY: "victim-adapter", "lmcache.tag.user": "alice"}
    configs = extract_request_configs_from_request(_request(lora_name, params))
    assert configs is not None
    assert configs.get(LORA_TAG_KEY) == lora_name
    assert configs["lmcache.tag.user"] == "alice"


def test_same_tokens_under_different_adapters_get_different_keys() -> None:
    adapter_a = _object_keys(_request("adapter-a"))
    assert _object_keys(_request("adapter-a")) == adapter_a
    assert _object_keys(_request("adapter-b")) != adapter_a
    assert _object_keys(_request(None)) != adapter_a
