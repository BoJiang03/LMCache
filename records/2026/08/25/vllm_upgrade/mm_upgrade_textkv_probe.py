"""Text-only LMCache round-trip: does a cache HIT reproduce the MISS output?

No multimodal involved. If this diverges, the defect is in KV transport on
this vLLM version, not in multimodal cache keying.
"""
import os, sys, json, traceback

os.environ.setdefault("HF_HUB_CACHE", "/raid/data/hub")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["LMCACHE_CHUNK_SIZE"] = "16"
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"

out_path = sys.argv[1]
result = {"stage": "start"}

def dump(**kw):
    result.update(kw)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)

try:
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    import vllm
    result["vllm_version"] = vllm.__version__

    prompts = [
        "Question: " + ("The sky is blue and the grass is green. " * 40)
        + f" Repeat exactly the {n}th word of this text. Answer:"
        for n in (3, 7)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=24)

    # Pass A: plain vLLM, no LMCache, no prefix caching -- the reference.
    plain = LLM(model="Qwen/Qwen3-0.6B", enforce_eager=True,
                enable_prefix_caching=False, disable_log_stats=True,
                max_model_len=4096, gpu_memory_utilization=0.35)
    ref = [o.outputs[0].text for o in plain.generate(prompts, sp)]
    del plain
    dump(stage="baseline_done", baseline=ref)

    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    # Pass B: LMCache connector. First generate = miss, second = hit.
    llm = LLM(model="Qwen/Qwen3-0.6B", enforce_eager=True,
              enable_prefix_caching=False, disable_log_stats=True,
              max_model_len=4096, gpu_memory_utilization=0.35,
              kv_transfer_config=KVTransferConfig(
                  kv_connector="LMCacheConnectorV1", kv_role="kv_both"))
    miss = [o.outputs[0].text for o in llm.generate(prompts, sp)]
    hit = [o.outputs[0].text for o in llm.generate(prompts, sp)]

    dump(stage="done", ok=True, baseline=ref, miss=miss, hit=hit,
         baseline_eq_miss=(ref == miss), miss_eq_hit=(miss == hit))

except BaseException as exc:
    tb = traceback.format_exc()
    dump(ok=False, error_type=type(exc).__name__, error=str(exc)[:1500],
         tb_head=tb[:3000], tb_tail=tb[-3000:])
    raise SystemExit(1)
