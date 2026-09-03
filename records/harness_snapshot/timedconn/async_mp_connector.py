"""TimedMPConnector with the blocking LOOKUP round trip taken off the critical path.

THE QUESTION THIS ARM ANSWERS
-----------------------------
1j/1k established where the common +5.7 ms/step goes:

    SCHEDULER get_num_new_matched_tokens   7.51 ms/step  (~0.1 calls/step)
      sub_submit_lookup   7.357 ms/step wall, 0.083 ms/step thread CPU
      sub_check_result    0.424 ms/step wall, 0.015 ms/step thread CPU

98.9% of that is the vLLM EngineCore thread sitting blocked on a round trip to
the LMCache server, once per admitted request.  Two facts say the wait looks
removable: the value waited for is discarded (`fut.result(...)` in
`maybe_submit_lookup_request` is not bound to anything; the loop exists only to
catch TimeoutError), and vLLM's API already has a defer contract --
`get_num_new_matched_tokens` may return None for "ask me again later".

One fact says it may not pay: the IP connector ALREADY runs this lookup
asynchronously, through LMCacheAsyncLookupClient, and sits at 97.5 ms/step
against MP's 91.0.  Whatever IP's async path costs, it exceeds the 5.7 ms it
saves, and arm 1g already ruled out the backoff sleep as that cost.  So this
arm is a test, not a fix, and it is written to be falsifiable either way.

WHAT IS CHANGED, EXACTLY
------------------------
Nothing except when the scheduler thread waits.  The same LOOKUP is sent to the
same servers with the same key at the same point in the request's life.  The
only difference:

    stock:  send LOOKUP -> BLOCK for the ack -> send QUERY -> BLOCK -> answer
    here:   send LOOKUP -> return None (defer) -> ... later step ...
            ack already arrived? -> send QUERY -> BLOCK -> answer

QUERY is never sent before the LOOKUP ack has been observed, so the server-side
ordering the stock code relies on (LOOKUP locks the chunks, QUERY reports them)
is preserved.  That is the one correctness risk in the naive version of this
change and it is designed out rather than hoped away.

Cost of the change: one extra engine step of latency per admitted request, and
one extra QUERY per deferral.  At ~0.1 admitted requests/step that is small,
but it is not zero, and `sub_async_defer` counts it so the record can say so.

FAILURE MODE, AND HOW IT ANNOUNCES ITSELF
-----------------------------------------
If the futures never resolve, every request defers forever and the run stalls
at zero throughput -- loud, not silent.  If deferral is merely expensive, the
vLLM stat line's `Deferred:` counter goes from MP's 0 to positive and ms/step
does not improve.  Either way the log says which.

Everything else is inherited from TimedMPConnector: the same 20 hook timers,
the same sub-timers, the same descriptor-safety guard.  Numbers from this arm
are directly comparable to 1j and 1k.
"""

import os
import time

from lmcache.integration.vllm.vllm_multi_process_adapter import send_lmcache_request
from lmcache.v1.multiprocess.protocol import RequestType

from timedconn.timed_mp_connector import (
    STATS,
    _ROLE,
    TimedMPConnector,
    logger,
)


def _acc(key: str, t0: float, c0: float) -> None:
    """Fold one call into the same accumulators the timed connector reports.

    Names start with sub_ so _report() emits the thread-CPU twin (cpusub_*) and
    keeps them out of hook_total -- they nest inside get_num_new_matched_tokens.
    """
    a = STATS[key]
    a[0] += 1
    a[1] += time.perf_counter() - t0
    a[2] += time.thread_time() - c0


def _install_async_lookup(conn) -> bool:
    """Replace the adapter's blocking submit with a fire-and-collect pair.

    Instance attributes shadow class attributes, so LMCache's own
    `self.scheduler_adapter.maybe_submit_lookup_request(...)` call inside
    get_num_new_matched_tokens picks these up with no source edit -- the same
    trick 1k used for its sub-timers, and the reason lmcache/ stays untouched.

    Wraps the ALREADY-TIMED bound methods, so sub_check_result still measures
    exactly what it measured in 1k: the real QUERY round trip, delegated to
    only once the LOOKUP ack is in hand.
    """
    adapter = getattr(conn, "scheduler_adapter", None)
    if adapter is None:
        logger.warning("ASYNC: no scheduler_adapter; nothing patched.")
        return False

    timed_check = adapter.check_lookup_result
    timed_cleanup = adapter.cleanup_lookup_result

    # request_id -> (futures by url, token_ids, cache_salt).  Membership here
    # means "LOOKUP sent, ack not yet observed", a state stock LMCache has no
    # name for because it never returns while in it.
    inflight: dict = {}

    def async_submit(request_id, token_ids, cache_salt=""):
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            adapter._ensure_heartbeat_started()
            if not adapter.is_healthy:
                return
            if request_id in adapter._pending_lookups or request_id in inflight:
                return
            aligned_end = (
                len(token_ids) // adapter.lmcache_tokens_per_chunk
            ) * adapter.lmcache_tokens_per_chunk
            key = adapter._create_key(
                token_ids,
                start=0,
                end=aligned_end,
                request_id=request_id,
                cache_salt=cache_salt,
            ).no_worker_id_version()
            inflight[request_id] = (
                {
                    url: send_lmcache_request(
                        adapter.mq_clients[url],
                        RequestType.LOOKUP,
                        [key, adapter.tp_size],
                    )
                    for url in adapter._server_urls
                },
                token_ids,
                cache_salt,
            )
        finally:
            _acc("sub_async_submit", t0, c0)

    def async_check(request_id):
        entry = inflight.get(request_id)
        if entry is not None:
            t0, c0 = time.perf_counter(), time.thread_time()
            futures, token_ids, cache_salt = entry
            # query() is the non-blocking done check on MessagingFuture.
            if not all(f.query() for f in futures.values()):
                _acc("sub_async_defer", t0, c0)
                return None  # vLLM's "ask me again later"
            try:
                for url, fut in futures.items():
                    try:
                        # Already resolved; this cannot block.  The stock code
                        # discards this value too -- the call exists only so a
                        # timeout marks the server unhealthy.
                        fut.result(timeout=adapter._mq_timeout)
                    except TimeoutError:
                        logger.warning(
                            "ASYNC: LOOKUP to %s timed out. Marking unhealthy.", url
                        )
                        adapter._health_events[url].clear()
                        inflight.pop(request_id, None)
                        return 0
                inflight.pop(request_id, None)
                # Exactly the bookkeeping the stock submit does on success.
                adapter._pending_lookups.add(request_id)
                adapter._lookup_params[request_id] = (token_ids, cache_salt)
            finally:
                _acc("sub_async_collect", t0, c0)
        return timed_check(request_id)

    def async_cleanup(request_id):
        inflight.pop(request_id, None)
        return timed_cleanup(request_id)

    adapter.maybe_submit_lookup_request = async_submit
    adapter.check_lookup_result = async_check
    adapter.cleanup_lookup_result = async_cleanup
    return True


class AsyncLookupMPConnector(TimedMPConnector):
    """TimedMPConnector; the LOOKUP ack is polled instead of waited on."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        patched = _install_async_lookup(self) if _ROLE[0] == "SCHEDULER" else False
        logger.info(
            "AsyncLookupMPConnector attached pid=%d role=%s async_lookup=%s",
            os.getpid(), _ROLE[0], patched,
        )
