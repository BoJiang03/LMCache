"""The candidate fix: submit the LOOKUP early, never block on its acknowledgement.

WHAT IS WRONG TODAY
-------------------
`LMCacheMPSchedulerAdapter.maybe_submit_lookup_request` sends a LOOKUP to every
server and then does, per server:

    fut.result(timeout=self._mq_timeout)

The returned value is **not bound to anything**.  The call exists only so that a
TimeoutError can mark the server unhealthy.  The vLLM EngineCore thread
therefore stops dead for a full round trip, once per admitted request, inside
`get_num_new_matched_tokens`, for an acknowledgement it discards.  Measured at
TP=8: 7.357 ms/step of wall against 0.083 ms/step of thread CPU, ~73 ms per
admitted request.

WHY THE OBVIOUS FIX DID NOT WORK (arm 1l)
-----------------------------------------
1l made the hook return None -- vLLM's "ask me again later" -- while the ack was
outstanding.  That removed 15.5 s of blocking and added 16.16 s of scheduler
thread CPU, because vLLM re-asks every waiting request on every step: the hook
was entered 31,171 times for 1,000 requests instead of 1,000.  Net zero.  The
deferral contract is the wrong tool when the waiting queue is deep.

WHAT THIS DOES INSTEAD
----------------------
Two changes that only make sense together:

1. **Submit early.**  `lmcache.mp.eager_prefetch` (already in LMCache, default
   off) moves the submit into `on_new_request`, which vLLM calls once when the
   request enters the waiting queue -- long before the scheduler asks what it
   matched.  On its own this only relocates the block, which is why the option
   does not help today.

2. **Do not block on the ack.**  The submit records the futures and returns.
   The ack is collected in `check_lookup_result`, the one place that genuinely
   needs the LOOKUP to have been processed, because QUERY_PREFETCH_STATUS reads
   the job the LOOKUP registered.  Server-side ordering is therefore preserved
   exactly as in the stock code -- the QUERY is still sent strictly after the
   ack is in hand -- but the wait now happens after the request has sat in the
   queue, so in steady state it has already completed and costs nothing.

No deferral, so no re-polling: the hook is still entered once per request.  The
wait is not moved off the critical path by pretending it is not needed; it is
moved to a point in the request's life where it has already finished.

FALLBACK BEHAVIOUR IS THE STOCK BEHAVIOUR
-----------------------------------------
If the ack has NOT arrived by the time check_lookup_result runs, this blocks on
it there, with the same timeout and the same unhealthy-marking as the stock
code.  Worst case is exactly today's cost, paid one call later.  There is no
regime in which this is slower than stock except by the cost of one dict lookup.

PRE-REGISTERED READING
----------------------
    ms/step -> ~85.3   the blocking LOOKUP was the tax, and this is the fix.
    ms/step -> 91.0    the blocking is not on the critical path; then the
                       nolookup arm must also read 91.0, and if it does not,
                       something other than the wait (the server's work, the
                       messaging threads) is the cost.
    Deferred > 0       BUG: this arm must never defer.  It returns None only
                       where the stock connector would.
"""

import os
import time

from lmcache.integration.vllm.vllm_multi_process_adapter import send_lmcache_request
from lmcache.v1.multiprocess.protocol import RequestType

from timedconn.timed_mp_connector import STATS, TimedMPConnector, logger


def _acc(key: str, t0: float, c0: float) -> None:
    a = STATS[key]
    a[0] += 1
    a[1] += time.perf_counter() - t0
    a[2] += time.thread_time() - c0


def _install(conn) -> bool:
    adapter = getattr(conn, "scheduler_adapter", None)
    if adapter is None:
        logger.warning("NOWAIT: no scheduler_adapter; nothing patched.")
        return False

    timed_check = adapter.check_lookup_result
    timed_cleanup = adapter.cleanup_lookup_result
    # request_id -> {url: future} for LOOKUPs whose ack has not been collected.
    acks: dict = {}

    def submit(request_id, token_ids, cache_salt=""):
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            adapter._ensure_heartbeat_started()
            if not adapter.is_healthy:
                return
            if request_id in adapter._pending_lookups:
                return
            aligned_end = (
                len(token_ids) // adapter.lmcache_tokens_per_chunk
            ) * adapter.lmcache_tokens_per_chunk
            key = adapter._create_key(
                token_ids, start=0, end=aligned_end,
                request_id=request_id, cache_salt=cache_salt,
            ).no_worker_id_version()
            acks[request_id] = {
                url: send_lmcache_request(
                    adapter.mq_clients[url], RequestType.LOOKUP,
                    [key, adapter.tp_size])
                for url in adapter._server_urls
            }
            # Exactly the bookkeeping the stock submit does on success.  Doing
            # it here rather than after the ack is what makes the hook return an
            # answer on its first call instead of deferring.
            adapter._pending_lookups.add(request_id)
            adapter._lookup_params[request_id] = (token_ids, cache_salt)
        finally:
            _acc("sub_nowait_submit", t0, c0)

    def check(request_id):
        futures = acks.pop(request_id, None)
        if futures:
            t0, c0 = time.perf_counter(), time.thread_time()
            ready = all(f.query() for f in futures.values())
            try:
                for url, fut in futures.items():
                    try:
                        # Discarded, exactly as in the stock code: the call is
                        # here only so a timeout marks the server unhealthy.
                        fut.result(timeout=adapter._mq_timeout)
                    except TimeoutError:
                        logger.warning(
                            "NOWAIT: LOOKUP to %s timed out. Marking unhealthy.", url)
                        adapter._health_events[url].clear()
                        adapter._pending_lookups.discard(request_id)
                        return 0
            finally:
                # Split so the record can say how often the ack had already
                # landed.  If sub_nowait_late is ~0 calls, the early submit did
                # its job; if it is most of them, eager_prefetch is not on.
                _acc("sub_nowait_ready" if ready else "sub_nowait_late", t0, c0)
        return timed_check(request_id)

    def cleanup(request_id):
        acks.pop(request_id, None)
        return timed_cleanup(request_id)

    adapter.maybe_submit_lookup_request = submit
    adapter.check_lookup_result = check
    adapter.cleanup_lookup_result = cleanup
    return True


class NoWaitMPConnector(TimedMPConnector):
    """LMCacheMPConnector, timed, with the discarded LOOKUP ack not waited on."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        from timedconn.timed_mp_connector import _ROLE

        patched = _install(self) if _ROLE[0] == "SCHEDULER" else False
        logger.info(
            "NoWaitMPConnector attached pid=%d role=%s nowait=%s eager_prefetch=%s",
            os.getpid(), _ROLE[0], patched, getattr(self, "_eager_prefetch", None),
        )
