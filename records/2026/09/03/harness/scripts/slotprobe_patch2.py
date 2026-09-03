import sys
P = "lmcache/integration/vllm/vllm_v1_adapter.py"
src = open(P).read()
if "_SLOTPROBE_LW" in src:
    print("already patched"); sys.exit(0)

# The layerwise site: save_kv_layer creates the storer at the FIRST layer, so
# this .to() drains only what layer 0 has queued, not the whole forward.  That
# is exactly the difference the layerwise arm is testing, so it needs its own
# split.  Counters are shared with the non-layerwise site -- the two branches
# are mutually exclusive within a run.
OLD = """                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping = slot_mapping.to(self.device)
"""
NEW = """                # TODO: have a pre-allocated buffer to hold the slot_mappings
                if _SLOTPROBE:
                    _SLOTPROBE_LW[0] = True
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
                    if _slotprobe_state["n"] % _SLOTPROBE_EVERY == 0:
                        _slotprobe_report()
                else:
                    slot_mapping = slot_mapping.to(self.device)
"""
assert src.count(OLD) == 1, f"expected 1 layerwise copy site, found {src.count(OLD)}"
src = src.replace(OLD, NEW, 1)

src = src.replace(
    '_slotprobe_state = {',
    '_SLOTPROBE_LW = [False]\n_slotprobe_state = {', 1)
src = src.replace(
    '"SLOTPROBE pid=%d calls=%d sync_ms/call=%.3f copy_ms/call=%.3f "\n'
    '        "store_ms/call=%.3f n_store=%d reqs/call=%.2f toks/req=%.0f",\n'
    '        os.getpid(), n,',
    '"SLOTPROBE%s pid=%d calls=%d sync_ms/call=%.3f copy_ms/call=%.3f "\n'
    '        "store_ms/call=%.3f n_store=%d reqs/call=%.2f toks/req=%.0f",\n'
    '        "-LAYERWISE" if _SLOTPROBE_LW[0] else "", os.getpid(), n,', 1)
assert "SLOTPROBE%s pid" in src, "report-line rewrite missed"
open(P, "w").write(src)
print("patched")
