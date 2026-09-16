# The prefill-throughput result, and what it cost to believe it

2026-09-16, continuing from `1_hunting_for_an_e2e_regime_where_packing_shows.md`.

That record ended with "the only reproducible e2e signal is LMCache server CPU,
and no latency effect is measurable in any configuration tried". This one
retracts that. There is a large, reproducible prefill-throughput effect, it was
invisible earlier because the runs were in the wrong regime, and finding it
required throwing out two of my own explanations.

## The result

200k-token prompts x 12, concurrency 12, TP=8, DeepSeek-V4-Flash,
`--enforce-eager --long-prefill-token-threshold=256 --max-num-seqs=64`,
`fp8_ds_mla`, L1 450 GB. Both arms are the same worktree pair as before:
baseline is the exact merge-base `c40b7807`, the branch is `5d96a232`, both
trees clean, and the only production-code difference is the packing change.

    run    arm       slot  cold total   warm p50   LMCache CPU   vLLM CPU
    ab_v2  baseline     1      256.6s     2380ms        204.0s    2213.8s
    ab_v2  packed       2      160.8s     2313ms         98.0s    1462.2s
    ab_v3  packed       1      140.0s     2280ms         86.3s    1354.5s
    ab_v3  baseline     2      209.5s     2346ms        207.4s    1985.7s

The cold pass is pure prefill (2.4M tokens, one output token each), so it
restates as aggregate prefill throughput:

    baseline   9,355 .. 11,454 tok/s
    packed    14,923 .. 17,147 tok/s

The ranges do not overlap: the worst packed run still beats the best baseline
run by 1.30x. Mean over mean is 1.54x. Server CPU halves and is the most stable
number of the set (204.0/207.4 against 98.0/86.3). Warm p50 moves 2-3%, which
is inside the noise and is not claimed.

The absolute throughputs are depressed by `--enforce-eager` and are not a
statement about the model. Only the ratio is a claim, and eager mode inflates
GPU time, so the ratio is biased against the change.

## Why it was invisible before

`--long-prefill-token-threshold=256` caps how many tokens one request may
advance per scheduler step. At 256, every one of the 12 in-flight requests
accumulates exactly one LMCache chunk per step, so every request emits an op
every step. At the default 8192 only one request advances per step, so there is
one op instead of twelve, and the per-step cost falls under the GPU step time
and disappears. The threshold is the amplifier; the PR has to say so, because a
reviewer who runs the default will see nothing.

## The mechanism, traced to lines

The cost is inside vLLM and never reaches the LMCache server:

1. `lmcache_mp_metadata.py` `GetStoreMetadata` emits one op per request per step
   once a chunk has accumulated, and the op carries the **whole** prefix, not a
   delta. Averaged over a pass that grows 0 to 200k, that is 100k tokens per op.
2. The metadata hangs off `SchedulerOutput.kv_connector_metadata`
   (`vllm/v1/core/sched/output.py:247`).
3. `MessageQueue.enqueue` (`shm_broadcast.py:824`) pickles it at protocol 5. The
   out-of-band callback only diverts torch tensors, via
   `dispatch_table[torch.Tensor] = _reduce_tensor`, so plain Python objects go
   in-band. A `list[int]` is walked element by element; a `bytes` is a memcpy.
4. Every TP worker's `worker_busy_loop` (`multiproc_executor.py:999-1010`) is a
   plain sequential loop: `rpc_broadcast_mq.dequeue()` and then
   `func(*args, **kwargs)`. The `pickle.loads` at `shm_broadcast.py:901` is
   strictly serial with the forward pass, on every rank, with no overlap
   available.

Measured per step at the real payload shape (12 ops of 100k tokens):

    payload build (scheduler)     3.65 -> 0.16 ms
    pickle.dumps (scheduler)     16.56 -> 0.36 ms
    pickle.loads (each rank)     34.56 -> 0.40 ms
    msgspec encode (rank 0)      10.03 -> 0.19 ms
    msgspec decode (server)      30.53 -> 0.19 ms

The serial chain is scheduler 19.7 ms plus worker unpickle 34.2 ms plus rank-0
encode 9.8 ms, about 64 ms, against an observed 89-123 ms per step. The
remainder is unexplained. `submit_store` returns a future, so the server decode
is not directly on the path; ZMQ back-pressure is a candidate and is **not**
claimed, because it was not measured.

## Three things I had wrong

**"The effect needs a GQA model."** A subagent brainstorm pointed out, and
`vllm_multi_process_adapter.py:384-395` confirms, that under MLA only the first
rank per node is a writer (`is_kv_writer` returns
`vllm_worker_id % (tp_size // n_servers) == 0`), gating
`lmcache_mp_connector_0201.py:665`, `vllm_multi_process_adapter.py:1652` and
`qringbuffer.py:328`. So every DSV4 run had one STORE reaching the server, not
eight, and the microbenchmark's "x200" describes a GQA model. That correction is
right about the server, but it does not touch the headline: `SchedulerOutput` is
broadcast to all ranks whatever the attention shape, so the dominant term is x8
regardless. The planned Llama-70B run existed only to get eight writers, and its
reason is gone. `start_load_kv` (`lmcache_mp_connector_0201.py:578`) has no such
gate, so RETRIEVE is x8 even under MLA; the warm pass, not the cold one, is
where the server-side amplification lives.

**"Three independent routes agree on 120 ms/step."** One of them was not a
measurement. Dividing the vLLM CPU delta by 8 ranks treats a total that includes
spin-wait as if it were work, and a shorter run spins less, so the quantity
moves with wall time by construction. The direct `pickle.loads` measurement,
34.2 ms per rank per step, is about a third of the observed wall saving, not the
whole of it. The agreement I reported was partly an artifact of the metric.

**"Running first is worth 15%."** Inferred from packed alone (140.0s first,
160.8s second), and it does not survive the other arm: baseline was *faster*
second (256.6s then 209.5s). The spread is run-to-run noise of 15-20% whose sign
differs between arms, not a slot effect. It is smaller than the gap either way,
but the reasoning was wrong before the second run landed.

## Harness

`ARM_ORDER` was added to the driver so the pair can be run both ways; arm order
had been an uncontrolled variable in every previous A/B.

Two report columns are wrong when the arms differ in wall time, and were nearly
read as evidence: the busiest-thread utilisation divides absolute CPU by
`measured_wall_s`, which differed by 1.5x between arms, and the `dispatch`
column picks the busiest thread by tid without a name, so the two arms may not
be reporting the same thread at all. Only the absolute CPU seconds are
comparable. This is the same weakness the subagent flagged independently.

## Where the PR stands

Claimable: prefill throughput 1.30x-1.54x in this regime, surviving an order
flip; server CPU halved; the per-component costs above; the mechanism traced to
lines. Not claimable: the full 89-123 ms decomposition, anything about warm TTFT,
and the vLLM-CPU-over-8 arithmetic. Must be stated unprompted: eager mode biases
the ratio against the change, the 256 threshold is the amplifier, and the
microbenchmark caption needs the MLA-versus-GQA distinction.

Still open: whether to squash `5d96a232` into `e4986c82` (Bo's call), the PR1
description itself, PR2 (delta) with the `qringbuffer.py:700` `token_offset` bug,
and the separate `resolve_prefetched_obj_keys` off-by-one.
