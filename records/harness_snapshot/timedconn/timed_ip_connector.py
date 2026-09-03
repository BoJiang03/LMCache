"""LMCacheConnectorV1 (the in-process "IP" connector) with every hook timed.

WHY
---
IP is the second of the two losses.  Per-step decomposition at TP=8 put it at
97.5 ms/step against MP's 91.0 and 85.3 with no connector: it pays the same
+5.7 as MP plus another +6.5 that nothing so far explains.  Arm 1g ruled out
the async-lookup backoff sleep.

The MP side has been mapped hook by hook (1j/1k) and the IP side has not, so
"IP is slower" is still a black box while MP is a measured one.  This is the
same instrument pointed at the other connector, so the two can be subtracted.

The structural difference that makes IP interesting: MP moves the KV offload
into a separate server process, IP does it inside the vLLM worker process.  If
the tax is CPU contention -- LMCache's threads competing for the GIL with the
Python that launches each forward step -- then IP should pay more of it than
MP for exactly that reason, and the sitecustomize step probe should show both
connectors leaving GPU time per step unchanged while stretching the wall clock.
That prediction is written down here before the run.

Reuses timed_mp_connector's accumulators verbatim, so an IP report and an MP
report are the same numbers in the same units and can sit in one table.
"""

import inspect
import os
import types

# vLLM's factory maps the name "LMCacheConnectorV1" to ITS OWN shim class,
# not to LMCache's lmcache_connector_v1 module -- the shim is what decides
# between the in-tree and the dev LMCache impl and then delegates every hook
# to LMCacheConnectorV1Impl.  Subclassing LMCache's class instead would time
# a connector vLLM never builds.
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)

from timedconn.timed_mp_connector import (
    HOOKS,
    STATS,
    STEP_HOOKS,
    _ROLE,
    _report,
    _STEPS,
    logger,
)
import time


def _wrap(name):
    fn = getattr(LMCacheConnectorV1, name)
    is_step = name in STEP_HOOKS
    acc = STATS[name]

    def wrapper(self, *args, **kwargs):
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            return fn(self, *args, **kwargs)
        finally:
            acc[0] += 1
            acc[1] += time.perf_counter() - t0
            acc[2] += time.thread_time() - c0
            if is_step:
                _STEPS[0] += 1
                if _STEPS[0] % int(os.environ.get("LMC_TIMER_REPORT_EVERY", "200")) == 0:
                    _report()

    wrapper.__name__ = name
    wrapper.__qualname__ = f"TimedIPConnector.{name}"
    wrapper.__doc__ = fn.__doc__
    return wrapper


class TimedIPConnector(LMCacheConnectorV1):
    """LMCacheConnectorV1, byte for byte, plus a wall clock on every hook."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        try:
            super().__init__(vllm_config, role, kv_cache_config)
        except TypeError:
            # The IP connector predates the 3-argument constructor; vLLM passes
            # kv_cache_config only to connectors that accept it.
            super().__init__(vllm_config, role)
        _ROLE[0] = getattr(role, "name", str(role))
        logger.info("TimedIPConnector attached pid=%d role=%s", os.getpid(), _ROLE[0])


_wrapped_ip, _skipped_ip = [], []
for _h in HOOKS:
    if not hasattr(LMCacheConnectorV1, _h):
        _skipped_ip.append((_h, "absent"))
        continue
    # getattr_static returns the descriptor, so a property stays visibly a
    # property instead of being invoked -- the mistake that killed all eight
    # workers the first time this wrapper was written for MP.
    _attr = inspect.getattr_static(LMCacheConnectorV1, _h)
    if not isinstance(_attr, types.FunctionType):
        _skipped_ip.append((_h, type(_attr).__name__))
        continue
    setattr(TimedIPConnector, _h, _wrap(_h))
    _wrapped_ip.append(_h)
