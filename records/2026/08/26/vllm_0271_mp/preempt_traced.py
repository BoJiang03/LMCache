"""Instrumented preemption scenario: why does the scheduler never progress?

Same experiment as scenario_wrapper.py + isolated_cases.run_preemption, with
a per-step trace of the scheduler's own bookkeeping written to <trace.log>:

  step, #running/#waiting/#skipped, free GPU blocks, in-flight prefill
  reservation, tokens scheduled this step, who was preempted, and each
  request's status / num_computed_tokens / num_preemptions.

Also records every connector scheduler-side decision
(get_num_new_matched_tokens -> (tokens, async?)) and every _preempt_request.

Bails out with os._exit after STEP_CAP steps so a livelocked run still
leaves a complete trace (the caller kills the process group to reap the
MP cache server).

Usage: preempt_traced.py <e2e_dir> <scenario> <model_key> <out.json> <trace.log>
"""
import faulthandler
import os
import runpy
import signal
import sys
import time

e2e_dir, scenario, model_key, out_json, trace_path = sys.argv[1:6]
sys.path.insert(0, e2e_dir)

STEP_CAP = int(os.environ.get("TRACE_STEP_CAP", "900"))
VERBOSE_STEPS = int(os.environ.get("TRACE_VERBOSE_STEPS", "60"))

trace = open(trace_path, "w", buffering=1)


def emit(msg: str) -> None:
    trace.write(f"{time.time():.3f} {msg}\n")


def install_trace() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    orig_schedule = Scheduler.schedule
    orig_preempt = Scheduler._preempt_request
    state = {"step": 0}

    def req_line(sched, rid):
        r = sched.requests.get(rid)
        if r is None:
            return f"{rid}:gone"
        try:
            nblk = sum(len(g) for g in sched.kv_cache_manager.get_blocks(rid).get_block_ids())
        except Exception:
            nblk = -1
        return (f"{rid}:{r.status.name}:c{r.num_computed_tokens}/{r.num_tokens}"
                f":p{r.num_preemptions}:b{nblk}")

    def snapshot(sched, out, step):
        try:
            free = sched.kv_cache_manager.block_pool.get_num_free_blocks()
        except Exception as e:
            free = f"err({e})"
        try:
            reserved = sched._inflight_prefill_reserved_blocks()
            inflight = len(sched._inflight_prefills)
        except Exception:
            reserved, inflight = "n/a", "n/a"
        try:
            pending = getattr(sched, "deferred_frees", ())
            deferred = f"{len(pending)}entries/{sum(len(b) for _, b in pending)}blocks"
            fences = sorted({f for f, _ in pending})[:4]
        except Exception:
            deferred, fences = "n/a", []
        reqs = " ".join(req_line(sched, rid) for rid in sorted(sched.requests))
        emit(
            f"step={step} run={len(sched.running)} wait={len(sched.waiting)} "
            f"skip={len(sched.skipped_waiting)} free={free} reserved={reserved} "
            f"inflight_prefills={inflight} deferred={deferred} fences={fences} "
            f"sched_seq={getattr(sched, 'sched_step_seq', -1)}/"
            f"proc_seq={getattr(sched, 'processed_step_seq', -1)} "
            f"sched_tok={out.total_num_scheduled_tokens} "
            f"new={len(out.scheduled_new_reqs)} preempted={sorted(out.preempted_req_ids)} "
            f"finished={sorted(out.finished_req_ids)} | {reqs}"
        )

    def patched_schedule(self, *a, **kw):
        out = orig_schedule(self, *a, **kw)
        step = state["step"]
        state["step"] += 1
        if step < VERBOSE_STEPS or step % 50 == 0:
            snapshot(self, out, step)
        if step >= STEP_CAP:
            emit(f"STEP_CAP {STEP_CAP} reached -- final snapshot then exit")
            snapshot(self, out, step)
            faulthandler.dump_traceback(file=trace, all_threads=True)
            trace.flush()
            os._exit(97)
        return out

    def patched_preempt(self, request, timestamp, *a, **kw):
        emit(
            f"PREEMPT step={state['step']} {request.request_id} "
            f"status={request.status.name} computed={request.num_computed_tokens} "
            f"npreempt={request.num_preemptions} inflight={request in self._inflight_prefills}"
        )
        return orig_preempt(self, request, timestamp, *a, **kw)

    orig_init = Scheduler.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        emit(
            f"scheduler init defer_block_free={self.defer_block_free} "
            f"requires_kv_delivery={self.requires_kv_delivery} "
            f"max_concurrent_batches={self.vllm_config.max_concurrent_batches} "
            f"async_scheduling={self.vllm_config.scheduler_config.async_scheduling} "
            f"watermark_blocks={self.kv_cache_manager.watermark_blocks} "
            f"is_kv_consumer={self.vllm_config.kv_transfer_config.is_kv_consumer}"
        )
        if os.environ.get("TRACE_DEFER_OFF") == "1":
            self.defer_block_free = False
            emit("defer_block_free forced OFF")

    Scheduler.__init__ = patched_init
    Scheduler.schedule = patched_schedule
    Scheduler._preempt_request = patched_preempt
    emit("scheduler instrumented")

    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    orig_match = LMCacheMPConnector.get_num_new_matched_tokens
    orig_alloc = LMCacheMPConnector.update_state_after_alloc

    def patched_match(self, request, num_computed_tokens):
        ret = orig_match(self, request, num_computed_tokens)
        if state["step"] < VERBOSE_STEPS or state["step"] % 50 == 0:
            emit(
                f"  match step={state['step']} {request.request_id} "
                f"status={request.status.name} vllm_computed={num_computed_tokens} -> {ret}"
            )
        return ret

    def patched_alloc(self, request, blocks, num_external_tokens):
        if state["step"] < VERBOSE_STEPS or state["step"] % 50 == 0:
            emit(
                f"  alloc step={state['step']} {request.request_id} "
                f"ext={num_external_tokens}"
            )
        return orig_alloc(self, request, blocks, num_external_tokens)

    LMCacheMPConnector.get_num_new_matched_tokens = patched_match
    LMCacheMPConnector.update_state_after_alloc = patched_alloc
    emit("connector instrumented")


if os.environ.get("TRACE_ASYNC_OFF") == "1":
    import vllm  # noqa: E402

    _orig_llm_init = vllm.LLM.__init__

    def _llm_init(self, *a, **kw):
        kw.setdefault("async_scheduling", False)
        return _orig_llm_init(self, *a, **kw)

    vllm.LLM.__init__ = _llm_init

from harness import configure_environment  # noqa: E402
from specs import MODEL_SPECS  # noqa: E402

spec = MODEL_SPECS[model_key]
if spec.hybrid_block_tokens:
    raise SystemExit("hybrid prompt shape not replicated here; run via pytest")
if not spec.supports_system_role:
    os.environ["LMCACHE_MM_E2E_NO_SYSTEM_ROLE"] = "1"
if spec.media_first_template:
    os.environ["LMCACHE_MM_E2E_MEDIA_FIRST"] = "1"
configure_environment()
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
install_trace()
print(f"[traced] pid={os.getpid()} scenario={scenario} model={model_key}", flush=True)
sys.argv = ["isolated_cases.py", scenario, model_key, out_json]
runpy.run_path(os.path.join(e2e_dir, "isolated_cases.py"), run_name="__main__")
