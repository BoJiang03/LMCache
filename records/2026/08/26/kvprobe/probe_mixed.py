"""At-concurrency bitwise injection check over the full 110-item subset.

pass1: all items (full compute + store). Snapshot the watch items' computed
KV at their store ops' blocks.
pass2: all items again (retrieve + tail recompute). Snapshot the same
items' injected KV at their retrieve ops' blocks.
Compare bitwise over the retrieved range. Answers + first-token logprobs
recorded per pass. Post-pass snapshots rely on the block pool being far
larger than the run's footprint so freed blocks are not recycled.
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, "/home/bo/LMCache-worktrees/multi_modal_verify/tests/e2e_mm")

from benchmark_parity import MMEBenchmark, engine_kwargs  # noqa: E402
from harness import (  # noqa: E402
    configure_environment,
    mp_kv_transfer_config,
    start_mp_cache_server,
)

WATCH = {24, 299, 329, 352, 392, 396, 449, 465, 531, 741, 781, 823, 949,
         996, 1072, 1482, 1576, 1618,   # flipped in the subset run
         298, 0, 67}                    # sibling of 299 + gap-moved controls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    configure_environment()

    import torch
    from vllm import LLM, SamplingParams
    import lmcache.integration.vllm.vllm_multi_process_adapter as mp_adapter

    items = json.load(open(args.items))
    bench = MMEBenchmark()

    ops = {"store": {}, "retrieve": {}, "adapter": None}
    orig_store = mp_adapter.LMCacheMPWorkerAdapter.submit_store_request
    orig_retr = mp_adapter.LMCacheMPWorkerAdapter.submit_retrieve_request

    def _key(request_id):
        m = re.match(r"(\d+)-", request_id)
        return int(m.group(1)) if m else -1

    def store_wrap(self, request_id, op, event, cache_salt=""):
        ops["adapter"] = self
        ops["store"][_key(request_id)] = {
            "start": op.start, "end": op.end, "blocks": list(op.flat_block_ids)}
        return orig_store(self, request_id, op, event, cache_salt)

    def retr_wrap(self, request_id, op, event, cache_salt=""):
        ops["adapter"] = self
        ops["retrieve"][_key(request_id)] = {
            "start": op.start, "end": op.end, "blocks": list(op.flat_block_ids)}
        return orig_retr(self, request_id, op, event, cache_salt)

    mp_adapter.LMCacheMPWorkerAdapter.submit_store_request = store_wrap
    mp_adapter.LMCacheMPWorkerAdapter.submit_retrieve_request = retr_wrap

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
                model="Qwen/Qwen2-VL-2B-Instruct", benchmark=bench,
                mm_processor_kwargs={"max_pixels": 602112},
                hybrid_block_tokens=0, hf_overrides={}, hybrid_family="",
                mm_encoder_attn_backend="", trust_remote_code=False,
            ),
        )
        sp = SamplingParams(temperature=0.0, max_tokens=8, seed=0, logprobs=5)

        def snapshot(blocks):
            kv = ops["adapter"].kv_caches
            snap = {}
            for name, t in kv.items():
                idx = torch.tensor(blocks, device=t.device)
                snap[name] = t.index_select(0, idx).clone()
            return snap

        watch_pos = {it["orig_index"]: i for i, it in enumerate(items)
                     if it["orig_index"] in WATCH}
        report = {"answers": {}, "compare": {}}
        snaps1 = {}

        for tag, opdict in (("pass1", ops["store"]), ("pass2", ops["retrieve"])):
            outs = llm.chat(bench.conversations(items), sampling_params=sp,
                            use_tqdm=True)
            torch.cuda.synchronize()
            time.sleep(5)
            base = 0 if tag == "pass1" else len(items)
            rows = {}
            for oi, pos in watch_pos.items():
                op = opdict.get(base + pos)
                if op is None:
                    print(f"[mixed] {tag}: no op for item {oi}", flush=True)
                    continue
                first = outs[pos].outputs[0]
                lps = {v.decoded_token: round(v.logprob, 6)
                       for v in first.logprobs[0].values()} if first.logprobs else {}
                rows[oi] = {"text": first.text, "lps": lps, "op_end": op["end"]}
                if tag == "pass1":
                    snaps1[oi] = (snapshot(op["blocks"]), op)
                else:
                    s2 = snapshot(op["blocks"])
                    s1, op1 = snaps1[oi]
                    nb = min(op1["end"], op["end"]) // 16
                    bad_tokens = 0
                    max_abs = 0.0
                    for name in s1:
                        ta = s1[name][:nb].float()
                        tb = s2[name][:nb].float()
                        neq = (ta != tb)
                        if neq.any():
                            per_tok = neq.any(dim=1).flatten(0, 1)
                            per_tok = per_tok.reshape(per_tok.shape[0], -1).any(-1)
                            bad_tokens += int(per_tok.sum())
                            max_abs = max(max_abs,
                                          (ta - tb).abs().max().item())
                    report["compare"][str(oi)] = {
                        "tokens": nb * 16, "bad_token_positions": bad_tokens,
                        "max_abs": max_abs}
                    print(f"[mixed] item {oi}: injected-vs-computed "
                          f"bad_tokens={bad_tokens} max_abs={max_abs:.6g} "
                          f"text {snaps1[oi] is not None and rows[oi]['text']!r}",
                          flush=True)
            report["answers"][tag] = rows
            print(f"[mixed] {tag} done", flush=True)
        flips = [oi for oi in report["answers"]["pass1"]
                 if oi in report["answers"]["pass2"]
                 and report["answers"]["pass1"][oi]["text"]
                 != report["answers"]["pass2"][oi]["text"]]
        report["flipped_now"] = sorted(flips)
        print(f"[mixed] flips this run: {sorted(flips)}", flush=True)
        pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"[mixed] report -> {args.out}", flush=True)
    finally:
        server.process.terminate()


if __name__ == "__main__":
    main()
