"""Dump the KV cache geometry LMCache is handed, per vLLM version.

The hit path loads the right NUMBER of tokens and the wrong CONTENT on
0.27.1, which is what a layout/format mismatch looks like. This prints the
raw tensors vLLM registers with the connector plus vLLM's own declared
layout, so the two versions can be compared field by field.
"""
import json, os, sys, traceback

os.environ.setdefault("HF_HUB_CACHE", "/raid/data/hub")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["LMCACHE_CHUNK_SIZE"] = "16"
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"

out = sys.argv[1]
res = {"stage": "start"}


def dump(**kw):
    res.update(kw)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


try:
    import torch, vllm
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    res["vllm_version"] = vllm.__version__
    res["torch_version"] = torch.__version__

    captured = {}

    # Record what LMCache DECIDES the layout is, not just what vLLM handed it.
    import lmcache.v1.gpu_connector.utils as gcu
    _disc = gcu.normalize_and_discover_per_layer_formats

    def disc(*a, **kw):
        out = _disc(*a, **kw)
        try:
            fmts = out[1]
            captured["discovered_formats"] = sorted({str(f) for f in fmts})
        except Exception as e:
            captured["discovered_formats"] = f"ERR {type(e).__name__}"
        return out

    gcu.normalize_and_discover_per_layer_formats = disc
    import lmcache.integration.vllm.kv_cache_groups as kcg
    if hasattr(kcg, "normalize_and_discover_per_layer_formats"):
        kcg.normalize_and_discover_per_layer_formats = disc

    orig = LMCacheConnectorV1Impl.register_kv_caches

    def patched(self, kv_caches):
        if not captured:
            names = list(kv_caches)[:2]
            captured["n_layers"] = len(kv_caches)
            captured["layers"] = [
                {
                    "name": n,
                    "shape": list(kv_caches[n].shape),
                    "stride": list(kv_caches[n].stride()),
                    "dtype": str(kv_caches[n].dtype),
                    "contiguous": bool(kv_caches[n].is_contiguous()),
                }
                for n in names
            ]
            try:
                from vllm.v1.attention.backends.utils import get_kv_cache_layout
                captured["vllm_kv_cache_layout"] = get_kv_cache_layout()
            except Exception as e:
                captured["vllm_kv_cache_layout"] = f"ERR {type(e).__name__}"
            try:
                from lmcache.integration.vllm.utils import vllm_layout_hints
                captured["lmcache_layout_hints"] = dict(vllm_layout_hints())
            except Exception as e:
                captured["lmcache_layout_hints"] = f"ERR {type(e).__name__}"
        return orig(self, kv_caches)

    LMCacheConnectorV1Impl.register_kv_caches = patched

    llm = LLM(model="Qwen/Qwen3-0.6B", enforce_eager=True, disable_log_stats=True,
              max_model_len=2048, gpu_memory_utilization=0.30, seed=0,
              kv_transfer_config=KVTransferConfig(
                  kv_connector="LMCacheConnectorV1", kv_role="kv_both"))
    llm.generate(["hello world " * 40], SamplingParams(temperature=0.0, max_tokens=4),
                 use_tqdm=False)
    dump(stage="done", **captured)
except BaseException as exc:
    tb = traceback.format_exc()
    dump(stage="error", error_type=type(exc).__name__, error=str(exc)[:1200],
         tb_tail=tb[-2500:], **captured if "captured" in dir() else {})
    raise SystemExit(1)
