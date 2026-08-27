"""Two-pass parity probe over the extracted MME subset, with logprobs.

--backend mp    : MP cache server + connector (parity pass1/pass2 shape)
--backend native: vLLM prefix caching, no LMCache (drift control shape)

Per item and pass, records the generated text and the first decoded
token's top-5 logprobs, so decision-gap movement is measurable even where
no answer flips.
"""
import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, "/home/bo/LMCache-worktrees/multi_modal_verify/tests/e2e_mm")

from benchmark_parity import MMEBenchmark, engine_kwargs  # noqa: E402
from harness import (  # noqa: E402
    configure_environment,
    mp_kv_transfer_config,
    start_mp_cache_server,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mp", "native"], required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=8)
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
    server = None
    if args.backend == "mp":
        server = start_mp_cache_server(
            zmq_port=26000 + (os.getpid() % 1000),
            http_port=27000 + (os.getpid() % 1000),
            chunk_size=16,
            log_path=pathlib.Path(args.out).with_suffix(".mp_server.log"),
            l1_size_gb=8.0,
            separate_object_groups=False,
        )
        llm = LLM(kv_transfer_config=mp_kv_transfer_config(server.zmq_port),
                  **kwargs)
    else:
        kwargs["enable_prefix_caching"] = True
        llm = LLM(**kwargs)

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0,
                        logprobs=5)
    try:
        report = {"backend": args.backend, "passes": {}}
        for tag in ("pass1", "pass2"):
            outs = llm.chat(bench.conversations(items), sampling_params=sp,
                            use_tqdm=True)
            rows = []
            for it, out in zip(items, outs):
                first = out.outputs[0]
                lps = {}
                if first.logprobs:
                    lps = {v.decoded_token: round(v.logprob, 6)
                           for v in first.logprobs[0].values()}
                rows.append({"orig_index": it["orig_index"],
                             "flipped_in_full_run": it["flipped"],
                             "text": first.text,
                             "num_cached_tokens": out.num_cached_tokens,
                             "lps": lps})
            report["passes"][tag] = rows
            print(f"[subset:{args.backend}] {tag} done", flush=True)
            time.sleep(5)  # let async stores land before pass2
        pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"[subset:{args.backend}] report -> {args.out}", flush=True)
    finally:
        if server is not None:
            server.process.terminate()


if __name__ == "__main__":
    main()
