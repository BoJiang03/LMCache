"""Image probe via LLM.chat(), which lets each model's own template handle
the image content -- Mistral's mistral_common template rejects a bare
{"type": "image"} placeholder and needs the data in the message."""

import os, sys, json, base64, io, traceback

os.environ.setdefault("HF_HUB_CACHE", "/raid/data/hub")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

hf_id, out_path = sys.argv[1], sys.argv[2]
result = {"hf_id": hf_id, "stage": "start"}

def dump(**kw):
    result.update(kw)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)

try:
    from PIL import Image
    from vllm import LLM, SamplingParams
    import vllm
    result["vllm_version"] = vllm.__version__

    buf = io.BytesIO()
    Image.new("RGB", (448, 448), (200, 30, 30)).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    llm = LLM(model=hf_id, trust_remote_code=True, max_model_len=4096,
              gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1},
              enforce_eager=True, disable_log_stats=True)
    dump(stage="engine_built")

    outs = llm.chat(
        [[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "What is the dominant color? Answer in one word."},
        ]}]],
        SamplingParams(temperature=0.0, max_tokens=16),
    )
    dump(stage="generated", ok=True, text=outs[0].outputs[0].text)

except BaseException as exc:
    tb = traceback.format_exc()
    dump(ok=False, error_type=type(exc).__name__, error=str(exc)[:2000],
         tb_head=tb[:4000], tb_tail=tb[-4000:])
    raise SystemExit(1)
