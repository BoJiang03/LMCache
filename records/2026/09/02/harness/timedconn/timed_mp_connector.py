"""LMCacheMPConnector with every KVConnector hook wall-clock timed.

WHY THIS EXISTS
---------------
Per-step decomposition (scripts/engine_rate.py, 23 blocks over 5 sessions) puts
the arms at three non-overlapping levels:

    85.3 ms/step   no connector, and the do-nothing NullConnector (arm 1i)
    91.0 ms/step   LMCache MP
    97.5 ms/step   LMCache IP

1i proves vLLM's entire connector path -- metadata build, ship, bind, the
per-layer maybe_transfer_kv_layer decorator on all 36 layers, and the
KVOutputAggregator's 16-way collective_rpc gather -- costs 0.0 ms/step.  So the
+5.7 ms/step that MP and IP pay in common is inside LMCache's own hook bodies
(or in threads LMCache starts).  Three mechanisms read out of the source and
sized by hand have now been refuted by experiment, 0 for 3.  Guessing a fourth
is not the move; measuring is.

WHY A SUBCLASS AND NOT A PATCH
------------------------------
The venv installs LMCache editable onto the vast_repro worktree, so editing
lmcache/ to add timers would dirty the repo that carries the deliverables.  A
subclass registered through kv_connector_module_path gets the same treatment
from vLLM's factory (proved by arm 1i) and touches nothing.  Inherited from
LMCacheMPConnector, unchanged: SupportsHMA, get_required_kvcache_layout,
requires_piecewise_for_cudagraph, and every hook body.  Only wall clock is added.

VALIDITY CHECK BUILT IN
-----------------------
Two perf_counter() calls per hook is ~100 ns against a 91 ms step.  If the
instrumented run does not itself land at 91.0 ms/step in engine_rate.py, the
instrument perturbed the thing it measures and the attribution is void.  Check
that first, before reading any attribution below it.

WHAT IT CANNOT SEE
------------------
Time spent by LMCache's background threads is not inside any hook.  That is why
each report also carries this process's rusage CPU delta over the same window:
if the hooks account for little but process CPU is high, the cost is off-hook
and the next instrument has to be different.

Output: one REPORT line per REPORT_EVERY steps per process, tagged with the pid
and the role, plus a final JSON dump per process into $LMC_TIMER_DIR.
"""

import atexit
import inspect
import json
import os
import resource
import time
import types
from collections import defaultdict

from vllm.logger import init_logger

from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

# vLLM only installs handlers for the "vllm" logger namespace.  A logger named
# after this module would be silently dropped -- that mistake aborted a good run
# on 2026-09-02 (arm 1i's first launch).  Sit inside the namespace instead.
logger = init_logger("vllm.timed_mp_connector")

REPORT_EVERY = int(os.environ.get("LMC_TIMER_REPORT_EVERY", "200"))
DUMP_DIR = os.environ.get("LMC_TIMER_DIR", "")

# name -> [call_count, wall_seconds, thread_cpu_seconds]
#
# thread_time() is CPU burned by the CALLING THREAD only, so wall-minus-cpu on
# the same interval separates "this call computed" from "this call waited".
# That distinction is the whole question for the lookup path: process-wide
# CPU (process_time, or the cpu_busy field) cannot answer it, because other
# threads keep running while the scheduler thread blocks on the server.
STATS: dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0])
_LAST = {"steps": 0, "wall": time.perf_counter(), "cpu": 0.0, "snap": {}}
_STEPS = [0]
_ROLE = ["?"]


def _cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _report() -> None:
    now = time.perf_counter()
    cpu = _cpu_seconds()
    dw = now - _LAST["wall"]
    dc = cpu - _LAST["cpu"]
    dsteps = _STEPS[0] - _LAST["steps"]
    if dsteps <= 0 or dw <= 0:
        return
    parts, hook_total = [], 0.0
    for name in sorted(STATS):
        cnt, tot, cpu_t = STATS[name]
        pc, pt, pcpu = _LAST["snap"].get(name, (0, 0.0, 0.0))
        d_cnt, d_tot, d_cpu = cnt - pc, tot - pt, cpu_t - pcpu
        if d_cnt == 0:
            continue
        # sub_* timers are nested INSIDE a hook; adding them to the hook total
        # would count the same microseconds twice.
        if not name.startswith("sub_"):
            hook_total += d_tot
        # ms per ENGINE STEP, which is the unit the 5.7 ms gap is quoted in --
        # not ms per call, which hides that some hooks run 36x per step.
        parts.append(f"{name}={1000 * d_tot / dsteps:.3f}({d_cnt / dsteps:.1f}x)")
        if name.startswith("sub_"):
            parts.append(f"cpu{name}={1000 * d_cpu / dsteps:.3f}({d_cnt / dsteps:.1f}x)")
    logger.info(
        "TIMER pid=%d role=%s steps=%d wall_ms/step=%.2f hooks_ms/step=%.3f "
        "cpu_busy=%.2f | %s",
        os.getpid(), _ROLE[0], dsteps, 1000 * dw / dsteps,
        1000 * hook_total / dsteps, dc / dw, " ".join(parts),
    )
    _LAST.update(steps=_STEPS[0], wall=now, cpu=cpu,
                 snap={k: (v[0], v[1], v[2]) for k, v in STATS.items()})
    # Dump every window, not only at exit.  vLLM tears the EngineCore down with
    # a signal it does not survive, so atexit never ran there and 1k lost the
    # exact call counts for the one process that mattered -- leaving per-call
    # cost as a range instead of a number.  A dump per window costs nothing and
    # cannot be lost.
    _dump()


def _dump() -> None:
    if not DUMP_DIR:
        return
    try:
        os.makedirs(DUMP_DIR, exist_ok=True)
        path = os.path.join(DUMP_DIR, f"timer_{_ROLE[0]}_{os.getpid()}.json")
        with open(path, "w") as f:
            json.dump({"pid": os.getpid(), "role": _ROLE[0],
                       "steps": _STEPS[0], "cpu_seconds": _cpu_seconds(),
                       "hooks": {k: {"calls": v[0], "seconds": v[1],
                                     "thread_cpu_seconds": v[2]}
                                 for k, v in STATS.items()}}, f, indent=2)
    except Exception as e:  # a dying process must not be made to die louder
        logger.warning("TIMER dump failed: %s", e)


atexit.register(_dump)

# The two hooks vLLM calls exactly once per engine step, one per role.  They are
# the step counter, so every number above is per-step and directly comparable to
# the 5.7 ms.
STEP_HOOKS = {"build_connector_meta", "start_load_kv"}

# Everything the connector overrides that can run per step.  One-shot setup
# (register_kv_caches, bind_gpu_block_pool, shutdown) and the classmethod
# get_required_kvcache_layout are left alone: wrapping them measures nothing and
# the classmethod would not survive this wrapper.
HOOKS = [
    "build_connector_meta", "build_connector_worker_meta",
    "get_num_new_matched_tokens", "update_state_after_alloc",
    "on_new_request", "request_finished", "request_finished_all_groups",
    "update_connector_output", "get_finished_count", "handle_preemptions",
    "take_events", "get_kv_connector_stats", "build_kv_connector_stats",
    "start_load_kv", "wait_for_layer_load", "save_kv_layer", "wait_for_save",
    "get_finished", "get_block_ids_with_load_errors",
    # base-class bookkeeping, wrapped to prove it stays at zero
    "bind_connector_metadata", "clear_connector_metadata",
]

# NOT wrapped, and the guard below enforces it: anything that is not a plain
# function.  transfer_intermediate_tensors is a @property returning False from
# config; the first version of this file wrapped it into a method, and a bound
# method is always truthy, so the connector asked the LMCache server for the
# TRANSFER_QUERY feature the server does not advertise and every worker died at
# init with "Connector enables transfer_query but server does not."  A property,
# classmethod or staticmethod turned into a function changes behaviour, which is
# exactly what an instrument must never do.


def _wrap(name):
    fn = getattr(LMCacheMPConnector, name)
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
                if _STEPS[0] % REPORT_EVERY == 0:
                    _report()

    wrapper.__name__ = name
    wrapper.__qualname__ = f"TimedMPConnector.{name}"
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _wrap_instance(obj, attr: str, key: str) -> bool:
    """Time one bound method on one live object, without touching its class.

    Instance attributes shadow class attributes, so an internal `self.foo(...)`
    call inside LMCache picks this up with no source change.  Used only for
    private helpers that live on the scheduler adapter, which is not a class we
    subclass.
    """
    try:
        orig = getattr(obj, attr)
    except AttributeError:
        return False
    acc = STATS[key]

    def w(*args, **kwargs):
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            return orig(*args, **kwargs)
        finally:
            acc[0] += 1
            acc[1] += time.perf_counter() - t0
            acc[2] += time.thread_time() - c0

    setattr(obj, attr, w)
    return True


def _install_sub_timers(conn) -> list[str]:
    """Split get_num_new_matched_tokens into its parts.

    1j put 7.5 of the scheduler's 8.5 ms/step inside this one hook, at ~0.1
    calls/step -- tens of ms per admitted request, on the EngineCore critical
    path.  The hook does four things worth separating, because they have four
    different fixes:

      sub_get_tracker    per-request bookkeeping in the scheduler process
      sub_create_key     builds tuple(token_ids) over the WHOLE 60k prompt
      sub_submit_lookup  the blocking LOOKUP round trip (create_key is INSIDE
                         it, so the round trip alone is submit - create_key)
      sub_check_result   the blocking QUERY_PREFETCH_STATUS round trip

    Each carries thread CPU next to wall, so "computing" and "waiting on the
    server" are told apart rather than argued about.
    """
    done = []
    if _ROLE[0] != "SCHEDULER":
        return done
    if _wrap_instance(conn, "_get_or_create_request_tracker", "sub_get_tracker"):
        done.append("sub_get_tracker")
    adapter = getattr(conn, "scheduler_adapter", None)
    if adapter is None:
        logger.warning("TIMER: no scheduler_adapter; the lookup path is untimed.")
        return done
    for attr, key in (("_create_key", "sub_create_key"),
                      ("maybe_submit_lookup_request", "sub_submit_lookup"),
                      ("check_lookup_result", "sub_check_result")):
        if _wrap_instance(adapter, attr, key):
            done.append(key)
        else:
            logger.warning("TIMER: %s absent on the scheduler adapter.", attr)
    return done


class TimedMPConnector(LMCacheMPConnector):
    """LMCacheMPConnector, byte-for-byte, plus a wall clock on every hook."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        _ROLE[0] = getattr(role, "name", str(role))
        installed = _install_sub_timers(self)
        logger.info("TimedMPConnector attached pid=%d role=%s report_every=%d "
                    "sub_timers=%s", os.getpid(), _ROLE[0], REPORT_EVERY, installed)


_wrapped, _skipped = [], []
for _h in HOOKS:
    if not hasattr(LMCacheMPConnector, _h):
        _skipped.append((_h, "absent"))
        continue
    # getattr_static, not getattr: it returns the descriptor itself, so a
    # property is visibly a property instead of being invoked.
    _attr = inspect.getattr_static(LMCacheMPConnector, _h)
    if not isinstance(_attr, types.FunctionType):
        _skipped.append((_h, type(_attr).__name__))
        continue
    setattr(TimedMPConnector, _h, _wrap(_h))
    _wrapped.append(_h)
