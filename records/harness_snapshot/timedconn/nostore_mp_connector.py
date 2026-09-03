"""TimedMPConnector with the STORE path removed, and nothing else.

THE QUESTION
------------
The common +5.7 ms/step is not in any KVConnector hook.  1j/1k timed all 20 of
them: the scheduler's are 8.5 ms/step and the workers' are 0.85 ms/step across
all eight processes, and 1l proved the scheduler's big one is free -- deleting
2.1 ms/step of its blocking moved the step rate by 0.001%.  So the cost is
OFF-HOOK: work LMCache does on threads and processes vLLM never calls into.

LMCache has exactly two jobs, and this arm switches one off:

    store   push each prefilled request's KV out of vLLM's paged buffer
    lookup  ask the server whether a prefix is already cached

The store is the one with a physical mechanism to spare.  Every step, the
worker submits the freshly computed KV for copy-out; with --max-gpu-workers 8
the LMCache server workers read it from the vLLM buffer over IPC.  That
contends for copy engines, memory bandwidth and -- if it lands on NVLink --
the same fabric the TP=N all-reduces need.  None of it appears in a hook,
because the hook only submits.

WHAT IS CHANGED, EXACTLY
------------------------
One instance attribute on the worker adapter:

    worker_adapter.batched_submit_store_requests -> no-op

Everything else runs untouched: the scheduler still builds STORE metadata, the
worker still reads it, still calls create_recorded_event() and still records the
CUDA event, still dispatches wait_for_save.  Only the copy is not submitted.
That makes the arm a one-variable experiment rather than "LMCache off".

WHY THIS IS SAFE, AND WHY IT LOSES NOTHING OBSERVABLE
-----------------------------------------------------
The cold pass is 60,000 random tokens per prompt with unique token ids, so
every lookup misses and no request is ever served from the cache.  Storing or
not storing therefore changes no output and no hit count -- only the work.

No hang: _process_finished_stores() reports a finished request immediately when
it is in neither finished_stores nor store_futures, which is exactly the state a
never-submitted store leaves it in.  Blocks are freed on the same step.

FAILURE MODES, PRE-REGISTERED
-----------------------------
    ms/step falls to the no-connector baseline  -> the tax IS the store path.
    ms/step unchanged                           -> the store path is free too,
                                                   and the tax is the mere
                                                   presence of LMCache's
                                                   processes and threads.
    store_submits > 0 in the log                -> the patch did not take; VOID.
"""

import os

from timedconn.timed_mp_connector import TimedMPConnector, logger

# Counts what the stock code would have submitted.  A nonzero count with the
# patch installed is the proof that the arm actually removed work rather than
# measuring a path that was already empty -- if the workers never submit a
# store, "store off" and "store on" are the same run and the comparison is void.
SKIPPED = [0, 0]  # [calls to the no-op, requests whose store was skipped]


def _install_no_store(conn) -> bool:
    adapter = getattr(conn, "worker_adapter", None)
    if adapter is None:
        logger.warning("NOSTORE: no worker_adapter; nothing patched.")
        return False

    def no_store(request_ids, ops, event, cache_salts=None):
        SKIPPED[0] += 1
        SKIPPED[1] += len(request_ids)
        return

    adapter.batched_submit_store_requests = no_store
    return True


class NoStoreMPConnector(TimedMPConnector):
    """LMCacheMPConnector, timed, with the KV copy-out never submitted."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        from timedconn.timed_mp_connector import _ROLE

        patched = _install_no_store(self) if _ROLE[0] == "WORKER" else False
        logger.info(
            "NoStoreMPConnector attached pid=%d role=%s no_store=%s",
            os.getpid(), _ROLE[0], patched,
        )

    def wait_for_save(self):
        # Same body as the timed base; the count is logged here because the
        # worker has no other periodic hook that is cheap to piggyback on.
        r = super().wait_for_save()
        if SKIPPED[0] and SKIPPED[0] % 500 == 0:
            logger.info("NOSTORE pid=%d skipped_batches=%d skipped_requests=%d",
                        os.getpid(), SKIPPED[0], SKIPPED[1])
        return r
