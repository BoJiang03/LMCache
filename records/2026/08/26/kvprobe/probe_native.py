"""Native prefix-caching analog of the roundtrip probe. No LMCache.

Engine with enable_prefix_caching=True; items = the sibling pair (rows 240,
241, same image). Three passes of the SAME pair:
  pass1: cold. Records per-request num_cached_tokens -- did the second
         sibling reuse the first's blocks mid-batch?
  pass2: warm, served from the radix cache; tail recomputed.
  pass3: warm again (determinism).
Reports answers + first-token logprobs per pass. This is the native
control for the LMCache hit-path logit shift measured by probe_roundtrip.
"""
import argparse
import base64
import io as _io
import json
import pathlib
import sys

sys.path.insert(0, "/home/bo/LMCache-worktrees/multi_modal_verify/tests/e2e_mm")

from benchmark_parity import MMEBenchmark, engine_kwargs  # noqa: E402
from harness import configure_environment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[240, 241])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    configure_environment()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    ds = load_dataset("lmms-lab/MME", split="test")
    items = []
    uri_cache: dict[str, str] = {}
    for r in args.rows:
        row = ds[r]
        qid = row["question_id"]
        if qid not in uri_cache:
            buf = _io.BytesIO()
            row["image"].convert("RGB").save(buf, format="PNG")
            uri_cache[qid] = "data:image/png;base64," + base64.b64encode(
                buf.getvalue()
            ).decode("ascii")
        items.append(
            {"qid": qid, "image_uri": uri_cache[qid], "question": row["question"],
             "answer": row["answer"].strip().lower(), "category": row["category"]}
        )
    bench = MMEBenchmark()

    kwargs = engine_kwargs(
        model="Qwen/Qwen2-VL-2B-Instruct",
        benchmark=bench,
        mm_processor_kwargs={"max_pixels": 602112},
        hybrid_block_tokens=0,
        hf_overrides={},
        hybrid_family="",
        mm_encoder_attn_backend="",
        trust_remote_code=False,
    )
    kwargs["enable_prefix_caching"] = True
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0,
                        logprobs=5)

    report = {"rows": args.rows, "passes": {}}
    for tag in ("pass1_cold", "pass2_warm", "pass3_warm"):
        outs = llm.chat(bench.conversations(items), sampling_params=sp,
                        use_tqdm=False)
        rows = []
        for r, out in zip(args.rows, outs):
            first = out.outputs[0]
            lps = {}
            if first.logprobs:
                lps = {repr(v.decoded_token): round(v.logprob, 6)
                       for v in first.logprobs[0].values()}
            rows.append({
                "row": r,
                "text": first.text,
                "num_cached_tokens": out.num_cached_tokens,
                "prompt_len": len(out.prompt_token_ids),
                "first_token_logprobs": lps,
            })
            print(f"[native] {tag} row={r} cached={out.num_cached_tokens}/"
                  f"{len(out.prompt_token_ids)} text={first.text!r} lps={lps}",
                  flush=True)
        report["passes"][tag] = rows
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[native] report -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
