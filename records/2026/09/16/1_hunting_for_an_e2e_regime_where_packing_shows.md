# Hunting for an e2e regime where packing shows, and what the search cost

**Date:** 2026-09-16
**Branches:** `delta_token_ids_pr` @ `5d96a232` (PR1), `delta_token_ids` @ `56887e9e` (work)
**Follows:** [2026-09-15/3_splitting_the_packing_pr_and_running_it_e2e.md](../15/3_splitting_the_packing_pr_and_running_it_e2e.md)

## What this session was

One question, asked five ways: is there a configuration where packing
`token_ids` shows up as something other than a CPU number? The answer so far
is "server CPU, reliably; wall clock, not yet", and most of the value is in
the failures along the way.

## PR1 is test-clean now, and the wider sweep earned its keep

`tests/v1/multiprocess` alone had been green. Running `tests/v1 + tests/cli`
against the same sweep on the base commit gave 176 failures here against 172
there, so four were mine:

| failure | verdict |
| --- | --- |
| `test_ping::test_kvcache_default` | **not mine.** Reproduces on the base checkout whenever anything listens on :8080, which an LMCache server does by default. My own benchmark server was the thing listening. |
| `test_server_bench::TestMakeKey` | stale `key.token_ids` assertion |
| `test_mp_nonblocking_lookup` | passed `list(range(256))` where the parameter is now `packed_token_ids: bytes` |
| `test_fs_l2_adapter_keys::test_wire_compat_old_payload_decodes` | the wire break, see below |

Two are worth keeping.

**The nonblocking-lookup failure is a warning about the type change.**
`num_packed_tokens` divides by the stride, so a list of 256 ints reports 64
tokens rather than raising. The symptom was a free-lock range of `(128, 64)`
where `(128, 256)` was expected: a plausible-looking wrong number, not a type
error. Only a test hit it, but any caller that passes the old type gets the
same silence.

**The wire-compat test was measuring something narrower than it looked.** It
was written when `cache_salt` was added, and it pins that a *defaulted* field
does not break old payloads. That property still holds, so it keeps its
assertion with a current-shape payload, and a second test now pins the
opposite: a payload carrying the old `token_ids` tuple must be rejected,
because silently decoding one would leave `token_bytes` empty and mint chunk
hashes for an empty sequence.

While fixing it I found I had written a false precedent into the class
docstring. It claimed the break works "as `num_kv_readers` already requires".
`num_kv_readers` is defaulted: an old payload decodes fine and is refused
later by `require_num_kv_readers` with a specific message. This refusal
happens in msgspec as a missing-required-field error. The operational
consequence is the same, the failure surface is not, and the docstring said
so only after I checked.

Final: 887 passed, 1 skipped, 0 failed across `tests/v1/multiprocess` plus the
three touched files. An earlier run of the same set reported 1 failure and 6
`LMCacheTimeoutError` errors in `test_cache_server.py`, all of which were CPU
contention from a benchmark running at the same time and none of which
reproduce on an idle box.

## Two measurements had to be thrown away, for two different reasons

**A concurrent pytest.** The 200k / concurrency-4 A/B ran its baseline arm
between 23:54 and 00:04 while a pytest suite ran 23:59 to 00:04. The pytest
covered the baseline arm's cold and warm passes and none of the packed arm's,
an asymmetric disturbance pointing the favourable way. Re-run on an idle box,
the two headline differences vanished: warm p90/p99 had looked 30% better and
became 0.2%, vLLM CPU had looked 53.5s better and became 0.1s. The one number
that survived was LMCache server CPU, 18.5s against 10.6s.

**Undersized L1.** At 200k on DeepSeek-V4 a request is about 27 GB of KV
(2.85M tokens of GPU KV pool across 8 GPUs works out near 135 KB/token, since
the sparse index carries state beyond MLA's latent). `L1_SIZE_GB=200` holds
seven of them. The concurrency-4 runs evicted from mid-cold-pass onward, so
their warm pass was re-prefilling: warm p50 42.8s against cold p50 43.7s, an
"85x cache miss" that I read for a while as a cache hit. Sized properly the
same shape of workload gives warm 1121ms against cold 61.6s with zero
evictions. **Every warm number from before that fix is void.**

## The e2e search, in order

| # | configuration | result |
| --- | --- | ---: |
| 1 | Qwen2.5-7B, 31k x 64, c=16, TP=4 | nothing above noise |
| 2 | DSV4, 200k x 16, c=4, TP=4 | lmcache CPU **-42.6%**, latency flat within 0.3% |
| 3 | DSV4, 200k x 8, c=1, TP=4 | lmcache CPU -38.5%, warm TTFT noise (p50 +3.5%, mean -5.6%, n=8) |
| 4 | DSV4, 200k, TP=8, `--max-num-batched-tokens 2048` | busiest server thread **3.8%** of one core |
| 5 | DSV4, 200k x 32, c=32, TP=8, `--long-prefill-token-threshold 256` | busiest thread **32.9%**, packed arm crashed |

Run 5's lever is the one worth remembering. A STORE op fires once per
scheduler step per request and carries the **whole** sequence regardless of
how few tokens it covers. `--long-prefill-token-threshold` caps how many
tokens one request may take per step, so an 8192-token budget spreads across
32 requests and each emits its own op. The GPU attention work per step is
unchanged, since 32 x 256 is the same 8192 tokens against the same contexts,
but the ops per step go from 1 to 32. It is a real vLLM fairness knob, not a
rigged setting, and it moved the busiest thread 11x.

Shrinking `--max-num-batched-tokens` (run 4) looks similar and is not: it
multiplies ops but also stretches GPU time, because every prefill step
re-reads the whole context for attention. Cold p50 went 43.6s to 61.5s while
TP doubled.

## The ceiling is not arbitrary

Per-step dispatch load is `N_requests x ranks x c x avg_context`, and GPU KV
capacity forces `N x L <= capacity`, so per-step dispatch is bounded by
`ranks x c x capacity/2`, a constant independent of how N and L are traded.
On this box: `8 x 20.9ns x 1.43M / 0.68s per step` is about 35%. Measured
32.9%. **Tuning N or L cannot saturate that thread**, which is why the search
stopped being about finding a bigger workload.

## I was 5x optimistic three times, for one reason

Predicted then measured: 16-30% then 3.8%; 217% then 41%; and on TTFT a 3x
warm improvement then nothing. The common cause: **during prefill the op
payload grows from 0 to L**, so the average op costs about half the peak, and
I kept pricing every op at the full-context figure. Correcting for it, the
model now predicts the server stages at 15.2s per request against 16s
measured, which is close enough to trust.

The same correction resolves an apparent contradiction between "2105 ms per
request" (microbenchmark, 25 steps, TP=8, full context) and "20 ms per step"
(one worker, one stage). Under run 5's configuration the per-request total is
not 2105 ms but about 33 s, because 781 steps at half the average payload is
15.6x the microbenchmark's work.

## Where the cost sits relative to vLLM's critical path

- LMCache server decode and hash: **off** the path, the handler's executor
  runs after `unwrap_request_payloads`, and vLLM does not wait.
- vLLM scheduler build and `pickle.dumps`: **overlapped**, `step_with_batch_queue`
  schedules the next batch without waiting for the current one.
- vLLM worker `pickle.loads`: **serial** before `execute_model` within a step
  (`multiproc_executor.py` dequeues at :1001 and calls at :1010). It can only
  be hidden by cross-step pipelining, and at run 5's shape it is roughly 20 ms
  against a ~200 ms step.

So the mechanism points at prefill wall time, not at TTFT and not at decode.
That prediction is **still unmeasured**; every run so far has been flat on
wall clock, and the one run that appeared not to be was an artifact.

## DeepSeek-V4 in this vLLM build is not stable

- `--kv-cache-dtype fp8_ds_mla` is mandatory. With `auto` the assert fires
  inside every TP worker at load and reads as "WorkerProc failed to start".
- `--enforce-eager` is mandatory. Otherwise FlashMLA's sparse_fp8 decode
  aborts on `Failed to initialize the TMA descriptor 700`. Note 700 is
  `CUDA_ERROR_ILLEGAL_ADDRESS` surfacing a sticky error from an earlier
  kernel, so the TMA parameters in the message are not the thing to debug.
- Under preemption it dies in a Triton kernel: `deepseek_v4/attention.py:372`
  into `fused_qk_rmsnorm.py:86`, illegal memory access. Run 5 asked for 32
  concurrent 200k requests when GPU KV holds about 13, and the packed arm
  died there. Not a packing bug; the baseline arm survived the same load by
  luck, and the fix is to keep `N x L` inside the KV pool.

`--enforce-eager` inflates GPU time, which pushes the connector's share
**down**. Every ratio in this record is therefore conservative, and the PR
should say so rather than wait to be asked.

## State

- `delta_token_ids_pr`: `e4986c82` (the change) + `5d96a232` (test fixes).
  Not squashed, not pushed. Whether to fold them is Bo's call.
- `delta_token_ids`: harness at `56887e9e`, with every run's raw JSON under
  `benchmarks/e2e/packed_token_ids_results/`, including the void ones, named
  so they stay identifiable.
- Running: a corrected A/B at 200k x 12, concurrency 12, TP=8, threshold 256,
  which is the first configuration that is both inside the KV pool and at the
  dispatch ceiling.
- Running: a fable subagent asked for experiment designs that would convince a
  skeptical reviewer, explicitly including the option "no such demonstration
  exists, claim this instead".

## Still open

- PR2, delta, which is what actually attacks the amplification: an op covering
  256 new tokens carries 200k. Packing divides that cost by 162, delta removes
  the factor of 781.
- The `resolve_prefetched_obj_keys` chunk off-by-one, its own small PR.
- The `qringbuffer.py:700` missing `op.token_offset`, which belongs to PR2.
