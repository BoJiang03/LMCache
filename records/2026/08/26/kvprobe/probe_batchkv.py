"""Is prefill KV batch-composition-dependent, bitwise?

Two modes, run as two separate processes on the same GPU:
  solo : items=[row 241] alone. Snapshot its computed KV at the store op's
         blocks, save to disk.
  batch: items=[11 filler questions on distinct images] + [241, 240], all in
         ONE chat call so prefills batch into shared waves. Both siblings
         miss (nothing stored yet; stores land after decode). Snapshot 241's
         and 240's KV, save.
Offline compare: solo-241 vs batch-241 (same tokens, different batch), and
batch-240 vs batch-241 over their shared token prefix (the two copies the
parity pass1 dedup races between).
"""
import argparse
import base64
import io as _io
import json
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


def load_rows(rows):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/MME", split="test")
    items, uri_cache = [], {}
    for r in rows:
        row = ds[r]
        qid = row["question_id"]
        if qid not in uri_cache:
            buf = _io.BytesIO()
            row["image"].convert("RGB").save(buf, format="PNG")
            uri_cache[qid] = "data:image/png;base64," + base64.b64encode(
                buf.getvalue()).decode("ascii")
        items.append({"qid": qid, "image_uri": uri_cache[qid],
                      "question": row["question"],
                      "answer": row["answer"].strip().lower(),
                      "category": row["category"]})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["solo", "batch"], required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    configure_environment()

    import os
    import torch
    from vllm import LLM, SamplingParams
    import lmcache.integration.vllm.vllm_multi_process_adapter as mp_adapter

    # fillers: first question of 11 distinct images, none sharing 240/241's
    rows = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 241, 240] \
        if args.mode == "batch" else [241]
    watch_rows = [241, 240] if args.mode == "batch" else [241]
    items = load_rows(rows)
    bench = MMEBenchmark()

    rec = {"store": {}, "adapter": None}
    orig_store = mp_adapter.LMCacheMPWorkerAdapter.submit_store_request
    orig_retr = mp_adapter.LMCacheMPWorkerAdapter.submit_retrieve_request

    def store_wrap(self, request_id, op, event, cache_salt=""):
        rec["adapter"] = self
        rec["store"][request_id] = {
            "start": op.start, "end": op.end,
            "blocks": list(op.flat_block_ids),
            "token_ids": [int(t) for t in op.token_ids]
            if op.token_ids is not None else [],
        }
        return orig_store(self, request_id, op, event, cache_salt)

    def retr_wrap(self, request_id, op, event, cache_salt=""):
        print(f"[batchkv] UNEXPECTED retrieve for {request_id} "
              f"start={op.start} end={op.end}", flush=True)
        return orig_retr(self, request_id, op, event, cache_salt)

    mp_adapter.LMCacheMPWorkerAdapter.submit_store_request = store_wrap
    mp_adapter.LMCacheMPWorkerAdapter.submit_retrieve_request = retr_wrap

    server = start_mp_cache_server(
        zmq_port=26000 + (os.getpid() % 1000),
        http_port=27000 + (os.getpid() % 1000),
        chunk_size=16,
        log_path=pathlib.Path(args.outdir) / f"{args.mode}.mp_server.log",
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
        outs = llm.chat(bench.conversations(items), sampling_params=sp,
                        use_tqdm=False)
        torch.cuda.synchronize()
        time.sleep(4)

        # store ops are keyed by internal request ids; map by token length +
        # arrival order: vLLM ids are "chatcmpl-..." in submission order?
        # Robust mapping: match op token_ids against each item's prompt ids.
        prompt_ids = [list(o.prompt_token_ids) for o in outs]
        meta = {"mode": args.mode, "rows": rows, "answers": {}}
        outdir = pathlib.Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        # LMCache rewrites mm placeholder tokens into hash-derived ids, so
        # op.token_ids does not literally match the prompt. Requests are
        # submitted in `rows` order and vLLM's LLM entrypoint numbers them
        # sequentially, so map by the trailing integer of the request id.
        def op_for(idx: int):
            for rid, op in rec["store"].items():
                m = re.match(r"(\d+)-", rid)
                if m and int(m.group(1)) == idx:
                    return rid, op
            raise RuntimeError(
                f"no store op for submission index {idx}; "
                f"ids={list(rec['store'])}")
        for w in watch_rows:
            idx = rows.index(w)
            pid = prompt_ids[idx]
            rid, op = op_for(idx)
            if abs(len(op["token_ids"]) - len(pid)) > 32:
                raise RuntimeError(
                    f"row {w}: op token count {len(op['token_ids'])} "
                    f"far from prompt {len(pid)} -- id mapping wrong")
            kv = rec["adapter"].kv_caches
            snap = {}
            for name, t in kv.items():
                if t.shape[1] != 2:
                    raise RuntimeError(f"layout {t.shape} for {name}")
                idxs = torch.tensor(op["blocks"], device=t.device)
                snap[name] = t.index_select(0, idxs).clone().cpu()
            torch.save(
                {"snap": snap, "op": {k: op[k] for k in ("start", "end")},
                 "token_ids": op["token_ids"], "row": w},
                outdir / f"{args.mode}_{w}.pt",
            )
            first = outs[idx].outputs[0]
            lps = {repr(v.decoded_token): round(v.logprob, 6)
                   for v in first.logprobs[0].values()} if first.logprobs else {}
            meta["answers"][str(w)] = {"text": first.text, "lps": lps,
                                       "prompt_len": len(pid),
                                       "store_end": op["end"]}
            print(f"[batchkv] {args.mode} row={w} prompt={len(pid)} "
                  f"store_end={op['end']} text={first.text!r} lps={lps}",
                  flush=True)
        (outdir / f"{args.mode}_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[batchkv] {args.mode} done", flush=True)
    finally:
        server.process.terminate()


if __name__ == "__main__":
    main()
