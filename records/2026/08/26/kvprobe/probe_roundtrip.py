"""Bit-level KV roundtrip probe: does LMCache's MP store+retrieve return the
exact bytes the engine computed?

One MME question, one engine, one MP server (chunk 16, same as parity):
  pass1  miss: full prefill, async store   -> snapshot P1 at the store's blocks
  pass2  hit : retrieve + tail recompute   -> snapshot P2 at the retrieve's blocks
  pass3  hit : retrieve again              -> snapshot P3 (hit-path determinism)
Compare P1 vs P2 bitwise per layer/token over the retrieved range.
"""
import argparse
import json
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
    ap.add_argument("--row", type=int, default=241)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    configure_environment()

    # Third Party
    import torch
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    # First Party
    import lmcache.integration.vllm.vllm_multi_process_adapter as mp_adapter

    # --- one MME item, loaded without converting the whole dataset ---
    import base64
    import io as _io

    ds = load_dataset("lmms-lab/MME", split="test")
    row = ds[args.row]
    buf = _io.BytesIO()
    row["image"].convert("RGB").save(buf, format="PNG")
    item = {
        "qid": row["question_id"],
        "image_uri": "data:image/png;base64,"
        + base64.b64encode(buf.getvalue()).decode("ascii"),
        "question": row["question"],
        "answer": row["answer"].strip().lower(),
        "category": row["category"],
    }
    print(f"[probe] row {args.row} qid={item['qid']} cat={item['category']}",
          flush=True)

    # --- record every store/retrieve op the worker adapter submits ---
    rec = {"store": [], "retrieve": [], "adapter": None}
    orig_store = mp_adapter.LMCacheMPWorkerAdapter.submit_store_request
    orig_retr = mp_adapter.LMCacheMPWorkerAdapter.submit_retrieve_request

    def store_wrap(self, request_id, op, event, cache_salt=""):
        rec["adapter"] = self
        rec["store"].append(
            {"request_id": request_id, "start": op.start, "end": op.end,
             "blocks": list(op.flat_block_ids)}
        )
        return orig_store(self, request_id, op, event, cache_salt)

    def retr_wrap(self, request_id, op, event, cache_salt=""):
        rec["adapter"] = self
        rec["retrieve"].append(
            {"request_id": request_id, "start": op.start, "end": op.end,
             "blocks": list(op.flat_block_ids),
             "skip_first_n_tokens": op.skip_first_n_tokens}
        )
        return orig_retr(self, request_id, op, event, cache_salt)

    mp_adapter.LMCacheMPWorkerAdapter.submit_store_request = store_wrap
    mp_adapter.LMCacheMPWorkerAdapter.submit_retrieve_request = retr_wrap

    bench = MMEBenchmark()
    import os
    server = start_mp_cache_server(
        zmq_port=26000 + (os.getpid() % 1000),
        http_port=27000 + (os.getpid() % 1000),
        chunk_size=16,
        log_path=pathlib.Path(args.out).with_suffix(".mp_server.log"),
        l1_size_gb=8.0,
        separate_object_groups=False,
    )
    try:
        llm = LLM(
            kv_transfer_config=mp_kv_transfer_config(server.zmq_port),
            **engine_kwargs(
                model="Qwen/Qwen2-VL-2B-Instruct",
                benchmark=bench,
                mm_processor_kwargs={"max_pixels": 602112},
                hybrid_block_tokens=0,
                hf_overrides={},
                hybrid_family="",
                mm_encoder_attn_backend="",
                trust_remote_code=False,
            ),
        )

        sp = SamplingParams(
            temperature=0.0, max_tokens=args.max_tokens, seed=0, logprobs=5
        )

        def run_once(tag: str):
            out = llm.chat(
                bench.conversations([item]), sampling_params=sp, use_tqdm=False
            )[0]
            first = out.outputs[0]
            lps = {}
            if first.logprobs:
                lps = {
                    repr(v.decoded_token): round(v.logprob, 6)
                    for v in first.logprobs[0].values()
                }
            print(f"[probe] {tag}: text={first.text!r} first-token logprobs={lps}",
                  flush=True)
            torch.cuda.synchronize()
            time.sleep(4)  # let the async store/retrieve fully land
            return first.text, lps

        def snapshot(blocks: list[int]):
            kv = rec["adapter"].kv_caches
            snap = {}
            for name, t in kv.items():
                # vLLM 0.27.1 FA layout: (num_blocks, 2, block_size, ...)
                if t.shape[1] == 2:
                    block_dim = 0
                elif t.shape[0] == 2:
                    block_dim = 1
                else:
                    raise RuntimeError(f"unexpected kv layout {t.shape} for {name}")
                idx = torch.tensor(blocks, device=t.device)
                snap[name] = t.index_select(block_dim, idx).clone().cpu()
            return snap

        report = {"row": args.row, "qid": item["qid"]}

        text1, lp1 = run_once("pass1-miss")
        if not rec["store"]:
            raise RuntimeError("no store op recorded in pass1")
        sop = rec["store"][-1]
        p1 = snapshot(sop["blocks"])
        print(f"[probe] store op: start={sop['start']} end={sop['end']} "
              f"blocks={len(sop['blocks'])}", flush=True)

        text2, lp2 = run_once("pass2-hit")
        if not rec["retrieve"]:
            raise RuntimeError("no retrieve op recorded in pass2 -- lookup missed")
        rop = rec["retrieve"][-1]
        p2 = snapshot(rop["blocks"])
        print(f"[probe] retrieve op: start={rop['start']} end={rop['end']} "
              f"blocks={len(rop['blocks'])} skip={rop['skip_first_n_tokens']}",
              flush=True)

        text3, lp3 = run_once("pass3-hit")
        rop3 = rec["retrieve"][-1]
        p3 = snapshot(rop3["blocks"])

        # --- bitwise comparison over the common stored+retrieved range ---
        block = 16
        n_cmp = min(sop["end"], rop["end"]) - max(sop["start"], rop["start"])
        n_blocks_cmp = n_cmp // block

        def compare(a, b, n_blocks, tag):
            diffs = {}
            total_bad_tokens = 0
            max_abs = 0.0
            for name in a:
                # snapshots are (blocks, 2, block_size, ...)
                ta = a[name][:n_blocks].float()
                tb = b[name][:n_blocks].float()
                neq = (ta != tb)
                if neq.any():
                    # collapse to per-token over (blocks, block_size)
                    per_tok = neq.any(dim=1).flatten(0, 1)
                    per_tok = per_tok.reshape(per_tok.shape[0], -1).any(dim=-1)
                    bad = per_tok.nonzero().flatten().tolist()
                    d = (ta - tb).abs().max().item()
                    max_abs = max(max_abs, d)
                    total_bad_tokens += len(bad)
                    diffs[name] = {"bad_tokens": bad[:32], "n_bad": len(bad),
                                   "max_abs": d}
            print(f"[probe] {tag}: layers_with_diff={len(diffs)} "
                  f"bad_token_positions_total={total_bad_tokens} "
                  f"max_abs={max_abs}", flush=True)
            return {"layers_with_diff": len(diffs), "layers": diffs,
                    "max_abs": max_abs}

        report["answers"] = {"pass1": text1, "pass2": text2, "pass3": text3}
        report["first_token_logprobs"] = {"pass1": lp1, "pass2": lp2, "pass3": lp3}
        report["store_op"] = {k: sop[k] for k in ("start", "end")}
        report["retrieve_op"] = {k: rop[k] for k in ("start", "end",
                                                     "skip_first_n_tokens")}
        report["n_tokens_compared"] = n_blocks_cmp * block
        report["p1_vs_p2"] = compare(p1, p2, n_blocks_cmp, "P1(computed) vs P2(retrieved)")
        report["p2_vs_p3"] = compare(p2, p3, min(len(rop["blocks"]),
                                                 len(rop3["blocks"])),
                                     "P2 vs P3 (hit determinism)")
        pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"[probe] report -> {args.out}", flush=True)
    finally:
        server.process.terminate()


if __name__ == "__main__":
    main()
