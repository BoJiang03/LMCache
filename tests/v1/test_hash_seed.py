# SPDX-License-Identifier: Apache-2.0
"""Tests for the seed of LMCache's prefix-hash chain.

The seed decides every chunk hash, so a seed that differs between the
scheduler process and the worker processes makes every cross-process lookup
miss while stores keep succeeding, i.e. a silently write-only cache.
"""

# Standard
from types import ModuleType
from typing import Any, List
import os
import subprocess
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.hash_seed import derive_none_hash, resolve_none_hash

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Prints the seed and the chunk hashes of a fixed token sequence, so that two
# independent interpreters can be compared.
_CHILD_SCRIPT = """
from lmcache.v1.config import LMCacheEngineConfig
import lmcache.v1.token_database as token_database

db = token_database.ChunkedTokenDatabase(
    LMCacheEngineConfig.from_defaults(chunk_size=16), None
)
hashes = [h for _, _, h in db.process_tokens(tokens=list(range(64)), make_key=False)]
print("RESULT", token_database.NONE_HASH, hashes)
"""


def _fake_vllm_kv_cache_utils(
    monkeypatch: pytest.MonkeyPatch, seeds: List[Any], calls: List[Any]
) -> None:
    """Install a fake ``vllm.v1.core`` whose init_none_hash pops from `seeds`.

    :param pytest.MonkeyPatch monkeypatch: The fixture used to install the
        module and to remove it again afterwards.
    :param List[Any] seeds: The values ``init_none_hash`` assigns to NONE_HASH,
        in order, mimicking vLLM's per-process randomness.
    :param List[Any] calls: Receives one entry per ``init_none_hash`` call, so
        a test can assert that vLLM was or was not consulted.
    """
    kv_cache_utils = ModuleType("vllm.v1.core.kv_cache_utils")
    kv_cache_utils.NONE_HASH = None

    def init_none_hash(hash_fn: Any) -> None:
        calls.append(hash_fn)
        kv_cache_utils.NONE_HASH = seeds.pop(0)

    kv_cache_utils.init_none_hash = init_none_hash
    core = ModuleType("vllm.v1.core")
    core.kv_cache_utils = kv_cache_utils
    monkeypatch.setitem(sys.modules, "vllm.v1.core", core)


def _int_hash(args: Any) -> int:
    """A stand-in hash function with the (prefix, tokens, extra) convention."""
    return hash(args)


def test_derive_none_hash_is_a_pure_function_of_the_hash_function() -> None:
    assert derive_none_hash(_int_hash) == derive_none_hash(_int_hash)
    assert derive_none_hash(lambda _: b"digest") == b"digest"


def test_resolve_ignores_vllms_random_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without PYTHONHASHSEED, vLLM's seed is random and must not be adopted."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    calls: List[Any] = []
    _fake_vllm_kv_cache_utils(monkeypatch, [b"random-1", b"random-2"], calls)

    first = resolve_none_hash(_int_hash)
    second = resolve_none_hash(_int_hash)

    assert first == second == derive_none_hash(_int_hash)
    # vLLM's global is left alone: re-seeding it mid-process would also
    # invalidate the block hashes vLLM has already computed for itself.
    assert calls == []


def test_resolve_adopts_vllms_seed_when_pythonhashseed_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With PYTHONHASHSEED set, vLLM's seed is deterministic, so it is kept."""
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    calls: List[Any] = []
    _fake_vllm_kv_cache_utils(monkeypatch, [b"from-seed", b"from-seed"], calls)

    assert resolve_none_hash(_int_hash) == b"from-seed"
    assert calls == [_int_hash]


def test_resolve_falls_back_when_vllm_is_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vLLM that cannot be imported or seeded must not break key derivation."""
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    core = ModuleType("vllm.v1.core")

    def raising_init_none_hash(hash_fn: Any) -> None:
        raise RuntimeError("already a kernel registered")

    kv_cache_utils = ModuleType("vllm.v1.core.kv_cache_utils")
    kv_cache_utils.init_none_hash = raising_init_none_hash
    core.kv_cache_utils = kv_cache_utils
    monkeypatch.setitem(sys.modules, "vllm.v1.core", core)

    assert resolve_none_hash(_int_hash) == derive_none_hash(_int_hash)


def test_chunk_hashes_agree_across_processes() -> None:
    """The regression test for the write-only cache.

    vLLM randomizes its NONE_HASH per process when PYTHONHASHSEED is unset, so
    before the fix two interpreters derived different keys for the same tokens.
    The scheduler looks up what the workers stored, so the cache never hit.
    """
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    results = set()
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            pytest.fail(f"child process failed:\n{proc.stderr}")
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
        assert len(line) == 1, f"unexpected child output:\n{proc.stdout}"
        results.add(line[0])

    assert len(results) == 1, f"keys differ between processes: {results}"
