"""Split the worker's store submission into its parts, to locate the blocking.

WHAT IS ALREADY KNOWN
---------------------
On the TP=4 lane, pool pinned identically in every arm (ms/step, per worker,
steady state, from sitecustomize's probe):

    arm        loop     exec     cpu   exec-cpu
    none      131.78   81.82   81.82     0.00
    nostore   131.79   78.02   78.01     0.01
    mp        132.87   81.16   79.56     1.60
    nolookup  133.06   83.63   82.07     1.56

The baseline's exec_wall EQUALS its exec_cpu: the worker's main thread never
blocks inside execute_model.  With LMCache attached it blocks 1.60 ms/step, and
no-op'ing ONE call -- batched_submit_store_requests -- takes both the blocking
and the whole step-time cost back to baseline.  The lookup path is not it.

And the stores being submitted are almost all rejected: in that same `mp` run
the server logged 56 successful stores against 8,344
"Failed to batched allocate ... no enough memory".  The worker is blocking to
submit work the server then throws away.

WHAT THIS ARM ADDS
------------------
`blocked` says the thread waited; it does not say on what.  wait_for_save does
two separable things per step, with two different fixes:

    create_recorded_event()          record a CUDA event and export an IPC
                                     handle for it.  cudaIpcGetEventHandle is a
                                     driver call and can serialise against the
                                     work already queued on the device.
    batched_submit_store_requests()  per request: _create_key over the whole
                                     60k-token prompt, then submit_store, which
                                     exports the event handle again and does a
                                     ZMQ send.

Each is wrapped with wall AND thread CPU, so "waited" and "computed" stay
separated at this level too.  Reading:

    sub_store_event blocked   -> the cost is the CUDA IPC event export, and the
                                 fix is to stop minting a fresh IPC event every
                                 step (reuse a pool, or record one event for
                                 the whole batch)
    sub_store_submit blocked  -> the cost is in the send path, and the fix is
                                 backpressure: a server with no room should say
                                 so once instead of being asked every step
    sub_store_key CPU-heavy   -> tuple(token_ids) over 60k tokens per store;
                                 that would show as CPU, not as blocking, and
                                 the probe says the tax is blocking

Nothing is disabled: this arm stores exactly what stock MP stores, so its
ms/step must reproduce `mp`'s.  If it does not, the sub-timers perturbed the
thing they measure and the split is void.
"""

import os

from timedconn.timed_mp_connector import TimedMPConnector, _wrap_instance, logger


def _install(conn) -> list[str]:
    adapter = getattr(conn, "worker_adapter", None)
    if adapter is None:
        logger.warning("STOREPROBE: no worker_adapter; nothing timed.")
        return []
    done = []
    for attr, key in (
        ("create_recorded_event", "sub_store_event"),
        ("batched_submit_store_requests", "sub_store_submit"),
        ("submit_store_request", "sub_store_one"),
        ("_create_key", "sub_store_key"),
        ("batched_submit_retrieve_requests", "sub_retrieve_submit"),
        ("get_finished", "sub_get_finished"),
    ):
        if _wrap_instance(adapter, attr, key):
            done.append(key)
        else:
            logger.warning("STOREPROBE: %s absent on the worker adapter.", attr)
    return done


class StoreProbeMPConnector(TimedMPConnector):
    """LMCacheMPConnector, timed, with the store submission split into parts."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        from timedconn.timed_mp_connector import _ROLE

        installed = _install(self) if _ROLE[0] == "WORKER" else []
        logger.info(
            "StoreProbeMPConnector attached pid=%d role=%s store_timers=%s",
            os.getpid(), _ROLE[0], installed,
        )
