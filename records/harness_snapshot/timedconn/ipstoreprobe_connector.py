"""Split the IP connector's wait_for_save, which is the whole IP tax.

WHAT IS ALREADY KNOWN (TP=4 lane, pool pinned identically in every arm)
----------------------------------------------------------------------
Per worker, ms/step, steady state, from sitecustomize's probe:

    arm        loop     exec     cpu   exec-cpu
    none      131.78   81.82   81.82     0.00
    mp        132.87   81.16   79.56     1.60
    timedip   143.28  140.39  140.18     0.21

IP adds +58.6 ms/step of CPU inside execute_model and almost no blocking, and
the hook timers say where all of it is:

    wait_for_save         2,195 calls  162.66 s wall  162.66 s CPU  73.938 ms/step
    save_kv_layer        79,020 calls    0.70 s        0.71 s        0.317
    wait_for_layer_load  79,020 calls    0.38 s        0.31 s        0.173
    (every other worker hook < 0.01 ms/step)

wall EQUALS CPU to two decimals, so this is not waiting on a device or a
socket, it is the calling thread doing work.  MP's wait_for_save measured
0.53 ms/step; IP's is 74.  And this is the GPU-only configuration -- local_cpu
is false and there is no remote backend, so there is nowhere for the data to
go.

The body is a loop over the step's requests calling
`self.lmcache_engine.store(token_ids, mask, kvcaches, slot_mapping, ...)`.
At TP=4 a step is 8192 tokens x 18,432 B per rank = 151 MB, and 151 MB in
73.9 ms is 2.0 GB/s -- roughly what a synchronous copy to PAGEABLE host memory
achieves, and an order of magnitude off a pinned transfer.

WHAT THIS ARM ADDS
------------------
Timers on the engine's own entry points, so the record can say whether the 74
ms is inside store() (and therefore LMCache's offload path) or spread across
the surrounding bookkeeping:

    sub_ip_store     lmcache_engine.store  -- the offload itself
    sub_ip_unpin     lookup_unpin          -- called once per request per step
    sub_ip_lookup    lookup                -- scheduler side, for symmetry

Nothing is disabled, so this arm must reproduce timedip's ms/step; if it does
not, the timers perturbed what they measure and the split is void.
"""

import os

from timedconn.timed_ip_connector import TimedIPConnector
from timedconn.timed_mp_connector import _ROLE, _wrap_instance, logger


def _install(conn) -> list[str]:
    impl = getattr(conn, "_lmcache_engine", None)
    if impl is None:
        logger.warning("IPSTOREPROBE: no _lmcache_engine on the connector.")
        return []
    engine = getattr(impl, "lmcache_engine", None)
    if engine is None:
        logger.warning("IPSTOREPROBE: the impl has no lmcache_engine yet.")
        return []
    done = []
    for attr, key in (("store", "sub_ip_store"),
                      # store_layer is the layerwise entry point.  Without it
                      # the layerwise arm reports sub_ip_store=0 calls and the
                      # store cost vanishes from the timers entirely.
                      ("store_layer", "sub_ip_store_layer"),
                      ("lookup_unpin", "sub_ip_unpin"),
                      ("lookup", "sub_ip_lookup"),
                      ("retrieve", "sub_ip_retrieve")):
        if _wrap_instance(engine, attr, key):
            done.append(key)
    return done


class IPStoreProbeConnector(TimedIPConnector):
    """TimedIPConnector plus timers on the engine calls wait_for_save makes."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        installed = _install(self)
        logger.info("IPStoreProbeConnector attached pid=%d role=%s engine_timers=%s",
                    os.getpid(), _ROLE[0], installed)
