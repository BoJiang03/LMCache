# SPDX-License-Identifier: Apache-2.0
"""Does vLLM's OWN prefix cache change MME answers? No LMCache involved.

Motivation (records/2026/08/26/8_ §四点五 and §七.3): on vLLM 0.27.1 the
LMCache hit path stopped being bit-exact -- qwen2-vl-2b moved 19/2374
answers between pass1 (miss) and pass2 (hit), gemma-4-e4b 5/2374, where
0.23 gave 0 for both. But pass2 differs from pass1 in two ways at once:
where the KV comes from, AND the batch composition, since a cached prefix
is not prefilled and the scheduler therefore sees a different shape.

This script isolates the second half. It runs the same 2374 questions
twice through ONE plain-vLLM engine with ``enable_prefix_caching=True``
and no cache reset in between, so the second pass is served largely out of
vLLM's own GPU prefix cache. LMCache is never loaded.

  answers differ  -> the drift is a property of vLLM 0.27.1 that LMCache
                     merely inherits; the parity gate's 0.5% flip budget
                     is what needs recalibrating, not the store path.
  answers match   -> vLLM's own cache preserves answers while LMCache's
                     does not, which puts the fault in the retrieve path.

Running ``benchmark_parity.py --role baseline`` twice does NOT substitute
for this: two plain passes have the same batch shape and are expected to
agree exactly (measured: pass1 vs baseline = 0 flips on both models), so
that comparison cannot see the hypothesis.

Decoding is greedy and seeded (``temperature=0.0, seed=0``, set inside
``run_batch``), so any difference is not sampling.
"""

# Standard
import argparse
import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def main() -> int:
    """Run the two passes and write the comparison.

    Returns:
        0 if the two passes agree on every question, 1 if any answer moved.
        The exit code is a report, not a verdict: either outcome is a valid
        measurement and the interpretation is in the module docstring.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=8)  # matches benchmark_parity.py:849
    parser.add_argument("--mm-processor-kwargs", default="")
    parser.add_argument("--chat-template-kwargs", default="")
    parser.add_argument("--hf-overrides", default="")
    parser.add_argument("--hybrid-family", default="")
    parser.add_argument("--hybrid-block-tokens", type=int, default=0)
    parser.add_argument("--mm-encoder-attn-backend", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    # First Party (test-local): the parity script owns the dataset, the
    # prompt rendering, the engine kwargs and the scoring. Reusing them is
    # the point -- a reimplementation here would not be comparable.
    from benchmark_parity import MMEBenchmark, engine_kwargs, run_batch

    # First Party (test-local)
    from harness import configure_environment

    configure_environment()
    benchmark = MMEBenchmark()
    mm_processor_kwargs = (
        json.loads(args.mm_processor_kwargs) if args.mm_processor_kwargs else {}
    )
    chat_template_kwargs = (
        json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else {}
    )
    hf_overrides = json.loads(args.hf_overrides) if args.hf_overrides else {}

    print(f"[drift] loading MME items (limit={args.limit})", flush=True)
    started = time.monotonic()
    items = benchmark.load_items(args.limit)
    print(
        f"[drift] {len(items)} items in {time.monotonic() - started:.1f}s; "
        f"building the engine with vLLM prefix caching ON",
        flush=True,
    )

    kwargs = engine_kwargs(
        args.model,
        benchmark,
        mm_processor_kwargs,
        args.hybrid_block_tokens,
        hf_overrides,
        args.hybrid_family,
        args.mm_encoder_attn_backend,
        args.trust_remote_code,
    )
    # The only deliberate departure from the parity engine. Everything else
    # is left exactly as parity builds it so the two are comparable.
    kwargs["enable_prefix_caching"] = True

    # Third Party
    from vllm import LLM

    llm = LLM(**kwargs)

    print("[drift] pass A (cold vLLM prefix cache)", flush=True)
    pass_a = run_batch(llm, benchmark, items, chat_template_kwargs, args.max_tokens)
    print("[drift] pass B (warm vLLM prefix cache, NOT reset)", flush=True)
    pass_b = run_batch(llm, benchmark, items, chat_template_kwargs, args.max_tokens)

    verdict_a = [benchmark.parse_answer(t, it) for t, it in zip(pass_a, items)]
    verdict_b = [benchmark.parse_answer(t, it) for t, it in zip(pass_b, items)]
    flips = [i for i in range(len(items)) if verdict_a[i] != verdict_b[i]]
    slots = collections.Counter(i // 2 for i in flips)
    report = {
        "model": args.model,
        "benchmark": "mme",
        "engine": "plain vLLM, enable_prefix_caching=True, no reset between passes",
        "lmcache": "not loaded",
        "num_questions": len(items),
        "scores": {
            "pass_a": benchmark.scores(items, verdict_a),
            "pass_b": benchmark.scores(items, verdict_b),
        },
        "flips_b_vs_a": len(flips),
        "flip_indices": flips,
        "flip_direction": dict(
            collections.Counter(f"{verdict_a[i]}->{verdict_b[i]}" for i in flips)
        ),
        "images_touched": len(slots),
        "images_with_both_questions_flipped": sum(1 for v in slots.values() if v == 2),
        "unparseable": [
            sum(1 for v in verdict_a if not v),
            sum(1 for v in verdict_b if not v),
        ],
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
    answers_path = pathlib.Path(args.out).with_suffix(".answers.json")
    answers_path.write_text(json.dumps({"pass_a": pass_a, "pass_b": pass_b}))
    print(
        f"[drift] flips(B vs A) = {len(flips)}/{len(items)}  "
        f"score {report['scores']['pass_a']['total']} -> "
        f"{report['scores']['pass_b']['total']}  "
        f"images_both_flipped = {report['images_with_both_questions_flipped']}  "
        f"-> {args.out}",
        flush=True,
    )
    return 1 if flips else 0


if __name__ == "__main__":
    sys.exit(main())
