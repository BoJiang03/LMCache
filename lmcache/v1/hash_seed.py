# SPDX-License-Identifier: Apache-2.0
"""Seed resolution for LMCache's prefix-hash chain.

Both :class:`lmcache.v1.token_database.TokenDatabase` and
:class:`lmcache.v1.multiprocess.token_hasher.TokenHasher` start their rolling
prefix hash from a seed value ("NONE_HASH").  The seed has to be identical in
every process that derives cache keys for the same cache, so it is resolved
here in one place rather than in each of them.
"""

# Standard
from typing import Any, Callable, Final, Tuple, Union
import os

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Input to the hash function used to derive a deterministic seed.  It mirrors
# the (prefix_hash, tokens, extra_keys) shape that every chunk hash uses, so a
# seed stays inside the domain the hash function is called with elsewhere.
_SEED_HASH_INPUT: Final[Tuple[int, Tuple[int, ...], Tuple[Any, ...]]] = (0, (), ())


def resolve_none_hash(
    hash_func: Callable[[Any], Union[int, bytes]],
) -> Union[int, bytes]:
    """Resolve the seed of the prefix-hash chain for the current process.

    vLLM seeds its own prefix-cache hash chain from
    ``vllm.v1.core.kv_cache_utils.NONE_HASH``.  When ``PYTHONHASHSEED`` is unset
    vLLM derives that value from :func:`os.urandom`, so it differs in every
    process and after every restart.  That is deliberate for vLLM, whose prefix
    cache never outlives a single process, but LMCache keys must agree between
    the scheduler process, every worker process, and across restarts.  Adopting
    a per-process random seed silently turns a shared or persistent cache
    write-only: stores succeed, and every lookup issued by a different process
    computes different chunk hashes and therefore misses.

    When ``PYTHONHASHSEED`` is set, vLLM's seed is a deterministic function of
    it, so it is adopted unchanged -- deployments that already follow that
    documented advice keep their existing keys.  Otherwise the seed is derived
    from ``hash_func`` alone, which is deterministic by construction and still
    distinguishes the supported hash algorithms from each other.

    :param Callable[[Any], Union[int, bytes]] hash_func: The hash function used
        for the prefix-hash chain, as returned by the hash-function loaders in
        :mod:`lmcache.v1.token_database` and
        :mod:`lmcache.v1.multiprocess.token_hasher`.

    :return: The seed, as an int or a digest, matching what ``hash_func``
        returns.  Callers that store the seed as an int normalize it
        themselves.
    """
    if os.getenv("PYTHONHASHSEED") is not None:
        try:
            # Third Party
            from vllm.v1.core import kv_cache_utils

            if hasattr(kv_cache_utils, "init_none_hash"):
                # Deterministic in this branch: vLLM hashes PYTHONHASHSEED.
                kv_cache_utils.init_none_hash(hash_func)
                none_hash = kv_cache_utils.NONE_HASH
                logger.info(
                    "Initialized NONE_HASH=%s from vLLM (PYTHONHASHSEED is set)",
                    none_hash,
                )
                return none_hash
        except (ImportError, AttributeError, ValueError, RuntimeError, AssertionError):
            # vLLM absent, too old, or unusable on this platform (importing
            # torch._dynamo.device_interface asserts on non-CUDA platforms).
            pass

    return derive_none_hash(hash_func)


def derive_none_hash(
    hash_func: Callable[[Any], Union[int, bytes]],
) -> Union[int, bytes]:
    """Derive the seed from ``hash_func`` alone, without consulting vLLM.

    Used directly by callers whose hash function vLLM cannot drive (blake3),
    and as the branch of :func:`resolve_none_hash` that keeps keys stable when
    vLLM's own seed is random.

    :param Callable[[Any], Union[int, bytes]] hash_func: The hash function used
        for the prefix-hash chain.

    :return: The seed, as an int or a digest, matching what ``hash_func``
        returns.
    """
    none_hash = hash_func(_SEED_HASH_INPUT)
    logger.info(
        "Computed NONE_HASH=%s from the hash function.  vLLM's NONE_HASH is "
        "random per process unless PYTHONHASHSEED is set, which would make "
        "cache keys disagree between the scheduler and the worker processes.",
        none_hash,
    )
    return none_hash
