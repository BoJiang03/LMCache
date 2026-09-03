import io, re, sys, os

P = "lmcache/integration/vllm/vllm_v1_adapter.py"
src = open(P).read()

if "SLOTPROBE" in src:
    print("already patched"); sys.exit(0)

HELPER = '''
# ---------------------------------------------------------------------------
# SLOTPROBE -- diagnostic only, env-gated, NOT for the PR branch.
#
# `slot_mapping.to(self.device)` in wait_for_save profiled at 33.7 ms/call for
# a 480 KB int64 tensor.  A pageable H2D copy is stream-ordered AND
# host-blocking, so that number is the sum of two very different things:
#
#     (a) draining the ~80 ms of forward-pass kernels already queued, and
#     (b) the 480 KB DMA, which should be ~50 us.
#
# Whether the fix is "make the copy async" or "stop synchronising here at all"
# depends entirely on the split.  Synchronising the current stream immediately
# BEFORE the copy changes no semantics -- the copy already waits for exactly
# that -- so this probe cannot perturb what it measures.
# ---------------------------------------------------------------------------
_SLOTPROBE = os.environ.get("LMC_SLOTPROBE", "0") == "1"
_SLOTPROBE_EVERY = int(os.environ.get("LMC_SLOTPROBE_EVERY", "200"))
_slotprobe_state = {
    "n": 0, "sync": 0.0, "copy": 0.0, "store": 0.0, "reqs": 0, "toks": 0,
    "n_store": 0,
}


def _slotprobe_report() -> None:
    s = _slotprobe_state
    n = s["n"]
    if not n:
        return
    logger.info(
        "SLOTPROBE pid=%d calls=%d sync_ms/call=%.3f copy_ms/call=%.3f "
        "store_ms/call=%.3f n_store=%d reqs/call=%.2f toks/req=%.0f",
        os.getpid(), n,
        s["sync"] * 1000.0 / n,
        s["copy"] * 1000.0 / n,
        s["store"] * 1000.0 / n,
        s["n_store"],
        s["reqs"] / n,
        s["toks"] / max(s["reqs"], 1),
    )

'''

# 1. helper after logger definition
src = src.replace("logger = init_logger(__name__)\n",
                  "logger = init_logger(__name__)\n" + HELPER, 1)

# 2. split the wait_for_save copy.  Target the SECOND occurrence (the
#    non-layerwise wait_for_save body); the first is the layerwise path.
OLD = """            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.to(self.device)
"""
NEW = """            # TODO: have a pre-allocated buffer to hold the slot_mappings
            if _SLOTPROBE:
                _t0 = time.perf_counter()
                torch.cuda.current_stream(self.device).synchronize()
                _t1 = time.perf_counter()
                slot_mapping = slot_mapping.to(self.device)
                _t2 = time.perf_counter()
                _slotprobe_state["n"] += 1
                _slotprobe_state["sync"] += _t1 - _t0
                _slotprobe_state["copy"] += _t2 - _t1
                _slotprobe_state["reqs"] += 1
                _slotprobe_state["toks"] += len(token_ids)
            else:
                slot_mapping = slot_mapping.to(self.device)
"""
assert src.count(OLD) == 1, f"expected 1 non-layerwise copy site, found {src.count(OLD)}"
src = src.replace(OLD, NEW, 1)

# 3. time the store() call that follows it, in the same body
OLD_STORE = """            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                transfer_spec=request.disagg_spec,
                request_configs=request.request_configs,
                req_id=request.req_id,
            )
"""
NEW_STORE = """            _ts0 = time.perf_counter() if _SLOTPROBE else 0.0
            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                transfer_spec=request.disagg_spec,
                request_configs=request.request_configs,
                req_id=request.req_id,
            )
            if _SLOTPROBE:
                _slotprobe_state["store"] += time.perf_counter() - _ts0
                _slotprobe_state["n_store"] += 1
                if _slotprobe_state["n"] % _SLOTPROBE_EVERY == 0:
                    _slotprobe_report()
"""
assert src.count(OLD_STORE) == 1, f"expected 1 store site, found {src.count(OLD_STORE)}"
src = src.replace(OLD_STORE, NEW_STORE, 1)

# 4. import time
if not re.search(r"^import time$", src, re.M):
    src = src.replace("import sys\n", "import sys\nimport time\n", 1)

open(P, "w").write(src)
print("patched")
