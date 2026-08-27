"""Load-and-generate probe for models blocked on vLLM 0.23.0.

Usage: mm_upgrade_load_probe.py <hf_id> <out.json> [--no-image]

Verdict per model: engine builds, and one image (or text) request decodes.
Anything else is recorded with the head AND tail of the traceback -- the
chained cause prints first and truncating the tail alone hides it.
"""

import os
import sys
import json
import io
import traceback

os.environ.setdefault("HF_HUB_CACHE", "/raid/data/hub")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

hf_id, out_path = sys.argv[1], sys.argv[2]
want_image = "--no-image" not in sys.argv

result = {"hf_id": hf_id, "with_image": want_image, "stage": "start"}


def dump(**kw):
    result.update(kw)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)


try:
    from PIL import Image
    from vllm import LLM, SamplingParams
    import vllm

    result["vllm_version"] = vllm.__version__
    dump(stage="imported")

    llm = LLM(
        model=hf_id,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1} if want_image else {},
        enforce_eager=True,
        disable_log_stats=True,
    )
    dump(stage="engine_built")

    if want_image:
        img = Image.new("RGB", (448, 448), (200, 30, 30))
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What is the dominant color? Answer in one word."},
        ]}]
        tok = llm.get_tokenizer()
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = llm.llm_engine.processor.input_preprocessor.get_tokenizer_group() and None
            raise
        outs = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": img}}],
            SamplingParams(temperature=0.0, max_tokens=16),
        )
    else:
        outs = llm.generate(
            ["The capital of France is"],
            SamplingParams(temperature=0.0, max_tokens=16),
        )

    dump(stage="generated", ok=True, text=outs[0].outputs[0].text)

except BaseException as exc:  # noqa: BLE001 -- probe records everything
    tb = traceback.format_exc()
    dump(
        ok=False,
        error_type=type(exc).__name__,
        error=str(exc)[:2000],
        tb_head=tb[:4000],
        tb_tail=tb[-4000:],
    )
    raise SystemExit(1)
