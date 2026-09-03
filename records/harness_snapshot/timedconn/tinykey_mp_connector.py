"""TimedMPConnector with the STORE key carrying only the chunk it stores.

THE QUESTION
------------
`nostore` proved the whole +5.70 ms/step at TP=8 is downstream of
`batched_submit_store_requests`, and chain23's cProfile diff named the single
biggest item inside it:

    _create_key            0.243 ms/step/rank   tuple(op.token_ids)
    msgspec msgpack_encode 1.013 ms/step/rank   ~255 KB of it on the wire
    zmq send/recv/poll     0.32  ms/step/rank

`IPCCacheServerKey.token_ids` is not the chunk being stored.  It is
`RequestTracker.token_ids`, seeded at vllm_v1_adapter.py:198 as
`prompt_token_ids[:num_tokens_to_compute].copy()` and grown as the request
advances; `start` / `end` index into it.  With 60,000-token prompts that is a
60k-int tuple rebuilt and msgpack-encoded every step, in every one of the 8
ranks, byte-identical across them.  Measured on the real dataclass:

       N tokens   tuple() ms   encode ms      bytes
           8192        0.031       0.085      35669
          60000        0.248       0.688     260915

The profiled 0.243 ms/step lands on the 60000 row.  Direct cost ~0.94 ms/step
per rank -- but the loss is 5.70.  The rest is ATTRIBUTED, not proved: the
ranks desynchronise and TP lockstep charges the whole step, visible as 61%
more `sched_yield` spin iterations in vLLM's shm queue.  Whether that
amplification collapses when the key shrinks is exactly what is untested, and
it is the difference between this fix being worth ~1.1% and worth ~6.7%.

WHAT IS CHANGED, EXACTLY
------------------------
For STORE requests only, the key is built from the slice actually being
stored:

    _create_key(token_ids,           start, end)      # stock
    _create_key(token_ids[start:end], 0,   end-start) # here

`submit_retrieve_request` shares `_create_key`, so the truncation is gated on
a flag set only for the duration of `batched_submit_store_requests`; the
retrieve path keeps the stock key.

WHY THIS IS SAFE, AND WHY IT LOSES NOTHING OBSERVABLE
-----------------------------------------------------
The server resolves a store key by `session.set_tokens(list(key.token_ids))`
then `session.get_hashes(key.start, key.end)` (engine_context.py:298-304), and
`get_hashes` asserts only that `start` is a multiple of chunk_size.  Chunk
hashes are prefix-chained, so a truncated token list yields DIFFERENT but
perfectly valid hashes, and the same NUMBER of chunks -- `len(slice)` equals
`end - start`.  `gpu_block_ids` is untouched, so `num_chunks` still matches
the block coverage and the all-or-nothing check in
`lmcache_driven_transfer.store` still passes.  The same KV bytes are copied to
differently-named keys.

The cold pass is 60,000 random tokens per prompt with unique ids, so every
lookup misses and nothing is ever served from the cache.  Renaming the keys
therefore changes no output and no hit count -- only the work.  This is the
same argument `nostore` rests on.

The guard: `get_hashes` asserts `start % chunk_size == 0`, and the slice is
only valid when `end - start` is a whole number of chunks too.  Any call that
does not satisfy both, or where `end > len(token_ids)`, falls back to the
stock key and is counted.  A run with FELLBACK large is not measuring what it
claims to.

THIS IS A DIAGNOSTIC, NOT THE FIX
---------------------------------
Truncating changes which key the KV lands under, which is fine only because
every lookup misses here.  The real fix keeps the key's identity and makes the
ENCODING cheap (raw buffer instead of a msgpack int array) or ships the delta
instead of the prefix.  This arm exists to size the prize before that work is
done.

FAILURE MODES, PRE-REGISTERED
-----------------------------
    ms/step -> ~86        the key IS the loss; the amplification collapses
                          with it, and the real fix is worth ~6.7%.
    ms/step -> ~90        the key is ~1 ms of a 5.7 ms problem; the cost is
                          mostly elsewhere in the store path and the fix is
                          worth ~1%.  Go back and keep digging.
    TRUNCATED == 0        the patch never fired; VOID.
    FELLBACK >> 0         the alignment guard rejected the real calls; VOID.
"""

import os

from timedconn.timed_mp_connector import TimedMPConnector, logger

# [calls to the wrapper, keys truncated, keys that fell back to stock,
#  tokens in the stock key, tokens in the truncated key]
STAT = [0, 0, 0, 0, 0]
IN_STORE = [False]


def _install_tiny_key(conn, chunk_size: int) -> bool:
    adapter = getattr(conn, "worker_adapter", None)
    if adapter is None:
        logger.warning("TINYKEY: no worker_adapter; nothing patched.")
        return False

    orig_create = adapter._create_key
    orig_batched = adapter.batched_submit_store_requests

    def tiny_create_key(token_ids, start, end, request_id, cache_salt=""):
        if IN_STORE[0]:
            n = len(token_ids)
            if (start % chunk_size == 0 and (end - start) % chunk_size == 0
                    and 0 <= start < end <= n):
                STAT[1] += 1
                STAT[3] += n
                STAT[4] += end - start
                return orig_create(token_ids[start:end], 0, end - start,
                                   request_id, cache_salt)
            STAT[2] += 1
        return orig_create(token_ids, start, end, request_id, cache_salt)

    def batched(request_ids, ops, event, cache_salts=None):
        STAT[0] += 1
        IN_STORE[0] = True
        try:
            return orig_batched(request_ids, ops, event, cache_salts)
        finally:
            IN_STORE[0] = False

    adapter._create_key = tiny_create_key
    adapter.batched_submit_store_requests = batched
    return True


class TinyKeyMPConnector(TimedMPConnector):
    """LMCacheMPConnector, timed, with the store key cut to its own chunk."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        from timedconn.timed_mp_connector import _ROLE

        chunk = int(os.environ.get("TINYKEY_CHUNK_SIZE", "8192"))
        patched = _install_tiny_key(self, chunk) if _ROLE[0] == "WORKER" else False
        logger.info(
            "TinyKeyMPConnector attached pid=%d role=%s tiny_key=%s chunk=%d",
            os.getpid(), _ROLE[0], patched, chunk,
        )

    def wait_for_save(self):
        r = super().wait_for_save()
        if STAT[0] and STAT[0] % 500 == 0:
            logger.info(
                "TINYKEY pid=%d batches=%d TRUNCATED=%d FELLBACK=%d "
                "mean_stock_tokens=%.0f mean_tiny_tokens=%.0f",
                os.getpid(), STAT[0], STAT[1], STAT[2],
                STAT[3] / max(STAT[1], 1), STAT[4] / max(STAT[1], 1),
            )
        return r
