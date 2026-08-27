"""Native-only regime test: pass1 = one batched chat over all items;
pass2 = the same items submitted ONE AT A TIME (each in its own chat
call), all served from vLLM's own prefix cache. No LMCache anywhere.
If pass2 flips the same items the MP runs flip, the flip mechanism is
vLLM batch-shape numerics, entered via serialized tail computes."""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, "/home/bo/LMCache-worktrees/multi_modal_verify/tests/e2e_mm")

from benchmark_parity import MMEBenchmark, engine_kwargs  # noqa: E402
from harness import configure_environment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    configure_environment()

    from vllm import LLM, SamplingParams

    items = json.load(open(args.items))
    bench = MMEBenchmark()
    kwargs = engine_kwargs(
        model="Qwen/Qwen2-VL-2B-Instruct", benchmark=bench,
        mm_processor_kwargs={"max_pixels": 602112},
        hybrid_block_tokens=0, hf_overrides={}, hybrid_family="",
        mm_encoder_attn_backend="", trust_remote_code=False,
    )
    kwargs["enable_prefix_caching"] = True
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=8, seed=0, logprobs=5)

    def row(it, out):
        first = out.outputs[0]
        lps = {v.decoded_token: round(v.logprob, 6)
               for v in first.logprobs[0].values()} if first.logprobs else {}
        return {"orig_index": it["orig_index"],
                "flipped_in_full_run": it["flipped"], "text": first.text,
                "num_cached_tokens": out.num_cached_tokens, "lps": lps}

    report = {"backend": "native_seq", "passes": {}}
    outs = llm.chat(bench.conversations(items), sampling_params=sp,
                    use_tqdm=True)
    report["passes"]["pass1"] = [row(i, o) for i, o in zip(items, outs)]
    print("[nativeseq] pass1 (batched) done", flush=True)

    rows = []
    for it in items:
        out = llm.chat(bench.conversations([it]), sampling_params=sp,
                       use_tqdm=False)[0]
        rows.append(row(it, out))
    report["passes"]["pass2"] = rows
    print("[nativeseq] pass2 (one-at-a-time) done", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[nativeseq] report -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
