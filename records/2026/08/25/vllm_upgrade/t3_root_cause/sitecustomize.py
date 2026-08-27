"""T3 debug instrumentation (v2), loaded via PYTHONPATH sitecustomize.

Active only when LMCACHE_T3DBG=1. Meta-path hook patches, at import time:
  - lmcache.v1.distributed.l1_manager (server): reservation outcomes,
    removal paths, and CRC32 of chunk bytes at finish_write / unsafe_read.
  - lmcache.integration.vllm.vllm_multi_process_adapter (engine): store /
    retrieve submissions and completion reporting, with timestamps.
"""
import os

if os.environ.get("LMCACHE_T3DBG") == "1":
    import importlib.abc
    import importlib.util
    import sys
    import time
    import traceback
    import zlib

    def _short(key):
        return f"{hash(key) & 0xFFFFFFFFFF:010x}"

    def _emit(msg):
        sys.stderr.write(f"[T3DBG {time.time():.3f} pid={os.getpid()}] {msg}\n")
        sys.stderr.flush()

    def _keys_arg(a, kw):
        return a[0] if a else kw.get("keys", [])

    def _crc(mo):
        try:
            return f"{zlib.crc32(bytes(mo.byte_array)) & 0xFFFFFFFF:08x}"
        except Exception as e:
            return f"ERR:{type(e).__name__}"

    def _patch_l1(mod):
        L1 = mod.L1Manager

        orig_rr = L1.reserve_read
        def reserve_read(self, *a, **kw):
            keys = _keys_arg(a, kw)
            ret = orig_rr(self, *a, **kw)
            fails = [(i, ret[k][0].name, _short(k))
                     for i, k in enumerate(keys) if ret[k][0].name != "SUCCESS"]
            if fails:
                _emit(f"RR n={len(keys)} nfail={len(fails)} fails={fails[:48]}")
            else:
                first = _short(keys[0]) if keys else "-"
                last = _short(keys[-1]) if keys else "-"
                _emit(f"RR n={len(keys)} all-ok first={first} last={last}")
            return ret
        L1.reserve_read = reserve_read

        orig_ur = L1.unsafe_read
        def unsafe_read(self, *a, **kw):
            keys = _keys_arg(a, kw)
            ret = orig_ur(self, *a, **kw)
            crcs = [(_short(k), _crc(ret[k][1])) for k in keys
                    if ret[k][0].name == "SUCCESS" and ret[k][1] is not None]
            fails = [(_short(k), ret[k][0].name) for k in keys
                     if ret[k][0].name != "SUCCESS"]
            _emit(f"UR n={len(keys)} crcs={crcs[:48]} fails={fails[:48]}")
            return ret
        L1.unsafe_read = unsafe_read

        orig_rw = L1.reserve_write
        def reserve_write(self, *a, **kw):
            keys = _keys_arg(a, kw)
            mode = kw.get("mode", a[3] if len(a) > 3 else "all")
            ret = orig_rw(self, *a, **kw)
            ok = [_short(k) for k in keys if ret[k][0].name == "SUCCESS"]
            nfail = len(keys) - len(ok)
            _emit(f"RW mode={mode} n={len(keys)} nok={len(ok)} ok={ok[:64]} "
                  f"nfail={nfail}")
            return ret
        L1.reserve_write = reserve_write

        orig_fw = L1.finish_write
        def finish_write(self, *a, **kw):
            keys = _keys_arg(a, kw)
            ret = orig_fw(self, *a, **kw)
            crcs = []
            for k in keys:
                if ret[k].name == "SUCCESS":
                    ent = self._objects.get(k)
                    crcs.append((_short(k), _crc(ent.memory_obj) if ent else "GONE"))
            fails = [(_short(k), ret[k].name) for k in keys
                     if ret[k].name != "SUCCESS"]
            _emit(f"FW n={len(keys)} crcs={crcs[:64]} fails={fails[:48]}")
            return ret
        L1.finish_write = finish_write

        orig_fr = L1.finish_read
        def finish_read(self, *a, **kw):
            keys = _keys_arg(a, kw)
            ret = orig_fr(self, *a, **kw)
            gone = [_short(k) for k in keys if k not in self._objects]
            _emit(f"FR n={len(keys)} deleted={gone[:48]}")
            return ret
        L1.finish_read = finish_read

        orig_del = L1.delete
        def delete(self, *a, **kw):
            keys = _keys_arg(a, kw)
            stack = "".join(traceback.format_stack(limit=8))
            _emit(f"DEL n={len(keys)} keys={[_short(k) for k in keys][:64]}\n"
                  f"STACK:\n{stack}")
            return orig_del(self, *a, **kw)
        L1.delete = delete

        orig_clear = L1.clear
        def clear(self, *a, **kw):
            stack = "".join(traceback.format_stack(limit=8))
            _emit(f"CLEAR args={a} {kw}\nSTACK:\n{stack}")
            return orig_clear(self, *a, **kw)
        L1.clear = clear

        _emit(f"patched L1Manager from {mod.__file__}")

    def _patch_adapter(mod):
        W = mod.LMCacheMPWorkerAdapter

        def _op_desc(op):
            try:
                fb = list(op.flat_block_ids)
                return (f"start={op.start} end={op.end} "
                        f"skip={getattr(op, 'skip_first_n_tokens', '?')} "
                        f"nblk={len(fb)} blk[:4]={fb[:4]}")
            except Exception as e:
                return f"opdesc ERR:{type(e).__name__}"

        orig_ss = W.submit_store_request
        def submit_store_request(self, request_id, op, event, cache_salt=""):
            _emit(f"ENG.SUBMIT_STORE req={request_id} {_op_desc(op)}")
            return orig_ss(self, request_id, op, event, cache_salt=cache_salt)
        W.submit_store_request = submit_store_request

        orig_sr = W.submit_retrieve_request
        def submit_retrieve_request(self, request_id, op, event, cache_salt=""):
            _emit(f"ENG.SUBMIT_RETRIEVE req={request_id} {_op_desc(op)}")
            return orig_sr(self, request_id, op, event, cache_salt=cache_salt)
        W.submit_retrieve_request = submit_retrieve_request

        orig_cfr = W._collect_finished_retrieves
        def _collect_finished_retrieves(self):
            done = orig_cfr(self)
            if done:
                _emit(f"ENG.RETRIEVE_DONE reqs={sorted(done)}")
            return done
        W._collect_finished_retrieves = _collect_finished_retrieves

        orig_gf = W.get_finished
        def get_finished(self, finished_req_ids_from_engine):
            ret = orig_gf(self, finished_req_ids_from_engine)
            stores, retrieves = ret
            if stores or retrieves:
                _emit(f"ENG.GET_FINISHED stores={sorted(stores or ())} "
                      f"retrieves={sorted(retrieves or ())}")
            return ret
        W.get_finished = get_finished

        _emit(f"patched LMCacheMPWorkerAdapter from {mod.__file__}")

    def _patch_torch_ops(mod):
        orig_memcpy = mod.lmcache_memcpy_async

        def lmcache_memcpy_async(*a, **kw):
            # Causal-proof experiment: order the synchronous default-stream
            # cudaMemcpy behind the non-blocking transfer stream's queued
            # kernels before it reads/writes the shared temp buffer.
            try:
                import torch
                if torch.cuda.is_initialized():
                    torch.cuda.current_stream().synchronize()
            except Exception:
                pass
            return orig_memcpy(*a, **kw)

        mod.lmcache_memcpy_async = lmcache_memcpy_async
        _emit(f"patched torch_ops.lmcache_memcpy_async (sync-before-copy) from {mod.__file__}")

    TARGETS = {
        "lmcache.v1.distributed.l1_manager": _patch_l1,
        "lmcache.integration.vllm.vllm_multi_process_adapter": _patch_adapter,
        "lmcache.v1.platform.torch_ops": _patch_torch_ops,
    }

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner, patcher):
            self._inner = inner
            self._patcher = patcher

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            try:
                self._patcher(module)
            except Exception:
                traceback.print_exc()

    class _Finder(importlib.abc.MetaPathFinder):
        _busy = False

        def find_spec(self, name, path=None, target=None):
            if name not in TARGETS or _Finder._busy:
                return None
            _Finder._busy = True
            try:
                spec = importlib.util.find_spec(name)
            finally:
                _Finder._busy = False
            if spec is None or spec.loader is None:
                return None
            spec.loader = _Loader(spec.loader, TARGETS[name])
            return spec

    sys.meta_path.insert(0, _Finder())
