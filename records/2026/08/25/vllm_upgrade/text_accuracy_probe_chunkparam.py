"""Text-only LMCache accuracy probe: is the hit path version-sensitive
WITHOUT any multimodal code involved?

Mirrors what the mm suite asserts, minus the media:

  phase A (cold)  -- LMCache stores; output is the local-compute reference
  reset vLLM's own prefix cache, by force if the public API refuses
  phase B (hit)   -- the SAME prompt must be served from LMCache and must
                     produce the SAME text

Validity gates, so a red result cannot be an artefact of the probe:

  * every phase-B request must report ``num_external_cached_tokens > 0``.
    Zero means nothing was loaded from LMCache and the comparison is
    vacuous -- the probe reports ``valid: false`` rather than a verdict.
  * every prompt is generated ALONE (batch of one) in both phases, so the
    batch composition -- and therefore the numerics -- are identical
    between the two. Batching a set of prompts changes composition between
    a cold and a warm pass and flips near-tie tokens on its own.
  * phase A output is checked against plain vLLM (no connector) first. If
    those already disagree the engine, not the cache, is the variable.

Imports nothing from tests/e2e_mm: the point is a path with no mm code on
it at all.
"""

import gc
import json
import os
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_CACHE", "/raid/data/hub")
# Same knobs tests/e2e_mm/harness.py::configure_environment sets.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["LMCACHE_CHUNK_SIZE"] = os.environ.get("PROBE_CHUNK", "16")
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "40.0"
os.environ.setdefault("PYTHONHASHSEED", "0")

MODEL = os.environ.get("PROBE_MODEL", "Qwen/Qwen3-0.6B")
N_CASES = int(os.environ.get("PROBE_CASES", "16"))
MAX_TOKENS = 12

out_path = sys.argv[1]
result = {"stage": "start", "model": MODEL, "n_cases": N_CASES}


def dump(**kw):
    result.update(kw)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)


SENTENCES = [
    "The harbour master keeps a ledger of every ship that docks after dusk.",
    "Rain fell on the tin roof for three days without stopping once.",
    "A narrow staircase leads from the kitchen down to the cold cellar.",
    "The clerk stamped each form twice and filed it under the wrong year.",
    "Wind moved through the orchard and shook loose the early fruit.",
    "Every Tuesday the baker sets aside four loaves for the almshouse.",
    "The bridge was rebuilt in stone after the second flood took the timber.",
    "Lanterns hung along the quay burned until the tide turned at dawn.",
    "He counted the sacks twice because the tally never came out even.",
    "The map on the wall showed roads that no longer went anywhere.",
]
COLORS = [
    "violet", "amber", "crimson", "olive", "indigo", "scarlet", "teal",
    "maroon", "azure", "saffron", "emerald", "russet", "cobalt", "ochre",
    "magenta", "turquoise",
]


def build_case(i: int) -> tuple[str, str]:
    """Return (prompt, ground-truth colour) for case ``i``.

    Each document starts with its own index, so no two cases share a
    prefix and every phase-B hit must come from that case's own stored KV.
    """
    color = COLORS[i % len(COLORS)]
    head = [SENTENCES[(i + k) % len(SENTENCES)] for k in range(15)]
    tail = [SENTENCES[(i + k + 3) % len(SENTENCES)] for k in range(15)]
    prompt = (
        f"Document number {i}.\n"
        + " ".join(head)
        + f"\nImportant record: the stone kept in box Q{i} is {color}.\n"
        + " ".join(tail)
        + f"\n\nQuestion: What colour is the stone kept in box Q{i}?\n"
        "Answer with one word.\nAnswer:"
    )
    return prompt, color


def reset_prefix_cache(llm) -> str:
    """Empty vLLM's own prefix cache, by force if the API refuses.

    Same two operations tests/e2e_mm/harness.py performs, inlined so this
    probe imports nothing from the mm suite.
    """
    if llm.reset_prefix_cache():
        return "public_api"
    core = llm.llm_engine.engine_core
    core = getattr(core, "engine_core", core)
    pool = core.scheduler.kv_cache_manager.block_pool
    pool.cached_block_hash_to_block = type(pool.cached_block_hash_to_block)()
    for block in pool.blocks:
        if block.ref_cnt == 0:
            block.reset_hash()
    return "forced"


class PrefillCounters:
    """vLLM's own accounting of who served each prefilled token."""

    def __init__(self) -> None:
        self.local_cached = 0
        self.external_cached = 0

    def install(self) -> None:
        from vllm.v1.metrics.stats import PrefillStats

        counters = self
        original = PrefillStats.set

        def patched(stats, num_prompt_tokens, num_local_cached_tokens,
                    num_external_cached_tokens):
            counters.local_cached += num_local_cached_tokens
            counters.external_cached += num_external_cached_tokens
            return original(stats, num_prompt_tokens, num_local_cached_tokens,
                            num_external_cached_tokens)

        PrefillStats.set = patched


try:
    import vllm
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    result["vllm_version"] = vllm.__version__
    result["torch_version"] = torch.__version__

    cases = [build_case(i) for i in range(N_CASES)]
    prompts = [c[0] for c in cases]
    truths = [c[1] for c in cases]
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=0)

    def gen_one(llm, prompt):
        out = llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0]
        return out.text, list(out.token_ids)

    common = dict(model=MODEL, enforce_eager=True, disable_log_stats=True,
                  max_model_len=4096, gpu_memory_utilization=0.35, seed=0)

    # ---- reference: plain vLLM, no connector -------------------------
    plain = LLM(**common)
    ref = [gen_one(plain, p) for p in prompts]
    result["prompt_tokens"] = len(
        plain.get_tokenizer().encode(prompts[0])
    )
    del plain
    gc.collect()
    torch.cuda.empty_cache()
    dump(stage="baseline_done", baseline=[t for t, _ in ref])

    # ---- LMCache engine ----------------------------------------------
    counters = PrefillCounters()
    counters.install()
    llm = LLM(**common, kv_transfer_config=KVTransferConfig(
        kv_connector="LMCacheConnectorV1", kv_role="kv_both"))

    miss, miss_ext = [], []
    for p in prompts:
        before = counters.external_cached
        miss.append(gen_one(llm, p))
        miss_ext.append(counters.external_cached - before)
    dump(stage="miss_done", miss=[t for t, _ in miss])

    # Let the asynchronous stores land before the cache is dropped.
    time.sleep(15)
    reset_mode = reset_prefix_cache(llm)

    hit, hit_ext, hit_loc = [], [], []
    for p in prompts:
        be, bl = counters.external_cached, counters.local_cached
        hit.append(gen_one(llm, p))
        hit_ext.append(counters.external_cached - be)
        hit_loc.append(counters.local_cached - bl)

    def acc(outs):
        return sum(t.lower().find(g) >= 0 for (t, _), g in zip(outs, truths))

    loaded = [i for i, e in enumerate(hit_ext) if e > 0]
    same_text = [i for i in range(N_CASES) if hit[i][0] == miss[i][0]]
    same_ids = [i for i in range(N_CASES) if hit[i][1] == miss[i][1]]
    base_eq_miss = [i for i in range(N_CASES) if ref[i][0] == miss[i][0]]

    dump(
        stage="done",
        valid=len(loaded) == N_CASES and len(base_eq_miss) == N_CASES,
        reset_mode=reset_mode,
        accuracy={
            "baseline": acc(ref) / N_CASES,
            "miss": acc(miss) / N_CASES,
            "hit": acc(hit) / N_CASES,
        },
        agreement={
            "baseline_eq_miss": len(base_eq_miss) / N_CASES,
            "hit_eq_miss_text": len(same_text) / N_CASES,
            "hit_eq_miss_token_ids": len(same_ids) / N_CASES,
        },
        external_cached_tokens_per_hit=hit_ext,
        local_cached_tokens_per_hit=hit_loc,
        external_cached_tokens_per_miss=miss_ext,
        cases=[
            {
                "i": i,
                "truth": truths[i],
                "baseline": ref[i][0],
                "miss": miss[i][0],
                "hit": hit[i][0],
                "hit_eq_miss": hit[i][0] == miss[i][0],
                "hit_external_tokens": hit_ext[i],
            }
            for i in range(N_CASES)
        ],
    )
except BaseException as exc:
    tb = traceback.format_exc()
    dump(stage="error", ok=False, error_type=type(exc).__name__,
         error=str(exc)[:1500], tb_head=tb[:3000], tb_tail=tb[-3000:])
    raise SystemExit(1)
