# The window is one notch wide, and the decode roof is what closes it

Record 5 reported c48 and c72 and left c60 and c84 running. Both are now in.
They bound the window on each side, and the reason the window is narrow turns
out to have nothing to do with the tier policy.

## 1. The ladder, complete

All arms: fp8, native 262,144 context, 256k corpus, L1 320 GiB, DEFER 30,
FLOOR 8192, HORIZON 2.5, TP=2, 1800 s window, no `--unsafe-override`.

| arm | x slots | compute | local | ext | TTFT p50 | TTFT p90 | waiting | preempt | kv p90 |
|---|---|---|---|---|---|---|---|---|---|
| f8k256c48 | 1.19 | 6.3% | 90.7% | 3.0% | 1.02 s | 2.73 s | 0.09 | 0 | -- |
| f8k256c60 | 1.49 | 7.7% | 86.7% | 5.6% | 1.04 s | 3.17 s | 0.09 | 0 | 84.2% |
| f8k256c72 | 1.78 | 11.8% | 40.6% | 47.6% | 9.23 s | 36.86 s | 3.32 | 7 | 99.3% |
| f8k256c84 | 2.08 | 17.3% | 29.6% | 53.1% | 25.96 s | 79.98 s | 8.29 | 2 | 99.5% |

Against the target `local 40-60%, ext >25%, TTFT p50 <10 s`, only c72 passes.
c60 fails ext, c84 fails TTFT. The window is one notch of the ladder wide.

## 2. What the two new arms scored

P6 predicted c60 would land in the middle at local 50-65% and ext 25-40%.
Falsified. c60 sits on the uncongested branch next to c48, at ext 5.6%, with
`waiting_mean` 0.09 and zero preemptions.

P7 predicted that if the transition were genuinely soft, c84 would degrade
continuously (local 20-32%, ext 55-70%, TTFT p50 15-30 s) rather than snap to
the parent corpus's collapsed branch (TTFT >60 s, compute >25%). Confirmed:
local 29.6%, TTFT p50 25.96 s, compute 17.3%. ext 53.1% is just under the
predicted band. P8's trigger, c84 collapsing, did not fire.

So the shape is asymmetric, and neither half of record 5's framing survives
untouched:

- Entry into the congested branch is still a jump. Between 1.49x and 1.78x,
  ext goes from 5.6% to 47.6%. Bistability was not removed.
- The congested branch itself is now graded. c72 to c84 degrades smoothly,
  and its near-knee end is livable: same branch as the bf16 parent's
  n16L320, but TTFT 9.2 s instead of 61.6 s.

Record 5 is right that the middle exists and right about why. What it did not
establish, and what was overstated in conversation, is that the transition
into that middle became continuous. It did not. c72 is the tamed congested
branch, not a new stable point between the two.

## 3. TTFT is queue wait, to the digit

Little's law on the engine's own occupancy samples, `waiting_mean` divided by
completed throughput:

    c60   0.09 / 0.3641 =  0.25 s     measured p50  1.04 s (work only)
    c72   3.32 / 0.2933 = 11.32 s     measured p50  9.23 s
    c84   8.29 / 0.3115 = 26.61 s     measured p50 25.96 s

There is no prefill term worth naming. Cutting TTFT at a fixed working point
means draining the waiting queue, and the queue is for GPU blocks.

## 4. The decode roof

Blocks are released when a request finishes, and a request spends most of its
life decoding: at c72, `lat_p50` 46.9 s of which about 32 s is 391 output
tokens at 81.6 ms each. So block turnover is set by decode speed. Decode
speed does not move:

| arm | inflight | tpot p50 | inflight/tpot | bytes/GPU/step | effective BW |
|---|---|---|---|---|---|
| c60 | 21.93 | 58.4 ms | 376 tok/s | 83.9 GB | 1.44 TB/s, 30% |
| c72 | 30.22 | 81.6 ms | 370 tok/s | 105.9 GB | 1.30 TB/s, 27% |
| c84 | 39.38 | 94.7 ms | 416 tok/s | 117.4 GB | 1.24 TB/s, 26% |

tpot is proportional to in-flight count with a near-zero intercept. The
machine decodes about 390 tok/s in aggregate no matter how the scheduler
arranges it. Per-step bytes are attention KV, `inflight x isl x 24,576` at two
of four KV heads per GPU at fp8, plus the 30 GB MoE weight read. That lands at
26-30% of H200 peak bandwidth, and prefill compute in the same steps is about
11 TFLOP/s/GPU against roughly 990 peak. Neither roof is being hit, so about
3x of the decode path is lost to kernel efficiency.

This kills two levers before they were run. A larger `--max-num-batched-tokens`
cannot redistribute a budget that is not the constraint, and a 6% deeper pool
from `--gpu-memory-utilization 0.93` moves loop gain by 6%.

## 5. Where the 3x might be

The engine log names one candidate. Attention is FLASH_ATTN v3, which is fine.
MoE is not:

    Using TRITON Unquantized MoE backend out of potential backends:
    ['TRITON', 'BATCHED_TRITON', 'FlashInfer TRTLLM', 'FlashInfer CUTLASS']

`auto` chose Triton for a 128-expert N=384 MoE, and there is a
`fused_moe_kernel` JIT compilation warning during inference. flashinfer 0.6.12
is installed.

The second candidate is independent of the first. At `--block-size 16` a
95,000-token sequence carries 5,940 block-table entries per layer and FA3's
paged decode indexes all of them every step. Block size 64 cuts that by four.
`FLOOR` is counted in blocks, so 8192 to 2048 holds the danger floor at the
same 131,072 tokens and leaves the policy unchanged; LMCache's 256-token chunk
divides evenly by 64.

Arms in flight at c72, one variable each, predictions P15-P22 pre-stated:

- `f8k256c72fic`  MOE_BACKEND=flashinfer_cutlass
- `f8k256c72b64`  BLOCK=64 FLOOR=2048

`f8k256c72fit` (flashinfer_trtllm) failed bringup and scores nothing:
`Unquantized MoE backend FlashInfer TRTLLM does not support the deployment
configuration since kernel does not support current device cuda`.

If both leave tpot flat, the roof is structural for TP=2 at 4 KV heads, which
is TP=4 territory and out of scope. Then c72 is the operating point and there
is no third knob to reach for.

## 6. L1 over L0 is already in range

Asked for `[1,3]`. It is 1.71 and needs no change.

L0 is fixed by the GPU budget at 4,077,968 tokens x 49,152 B = 186.67 GiB of
fp8 KV. L1 at 320 GiB gives 1.71. Record 4's 3.1 was bf16 with L1 at 576 GiB;
fp8 halves bytes per token on both sides and leaves the ratio alone, so what
brought it into range was shrinking L1, not the dtype.

Going lower is not free. c72 ended with 233.37 GiB of live L1 objects and 9
watermark events, so the tier is sitting on its working set. `L1_GB=256` would
give a ratio of 1.37 but puts the 0.80 watermark at 204.8 GiB, below the
measured working set, and would cost ext hit. Worth measuring only if the
lower ratio is wanted for its own sake.

## 7. Open

1. Done, see section 8. Both arms scored; the search stopped as pre-committed.
2. `max_deferral_seconds` still deserves its own PR. Unchanged from record 4.
3. Not attempted, and named so it is not silently dropped: nothing here
   measures the exclusive move-not-copy retrieve path, which is the second
   half of the method. Every arm to date still copies L1 to L0.

## 8. Both arms are in. The roof holds, and the search stops

All three rows are the same working point, CONC=72 on fp8 + 256k at native
context, one variable each against `f8k256c72`.

| arm | pool tok | inflight | tpot p50 | agg decode | lat p50 | TTFT p50 | TTFT p90 | waiting | preempt | compute | local | ext |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c72 | 4,077,968 | 30.22 | 81.6 ms | 370 tok/s | 46.90 s | 9.23 s | 36.86 s | 3.32 | 7 | 11.8% | 40.6% | 47.6% |
| c72fic | 4,090,176 | 30.16 | 83.4 ms | 362 tok/s | 49.34 s | 6.43 s | 40.65 s | 3.03 | 10 | 12.5% | 41.1% | 46.3% |
| c72b64 | 4,072,256 | 30.11 | 74.6 ms | 404 tok/s | 45.12 s | 8.50 s | 35.58 s | 3.13 | 7 | 12.1% | 40.9% | 47.1% |

`flashinfer_cutlass` does nothing. The backend switch is confirmed in the log
(`Using FlashInfer CUTLASS Unquantized MoE backend`), and tpot went from
81.6 ms to 83.4 ms, i.e. nowhere. P15 and P16 falsified. The Triton MoE
kernel was not the missing 3x.

`--block-size 64` is real but small. tpot 81.6 to 74.6 ms, aggregate decode
370 to 404 tok/s, +9%. P20 asked for under 70 ms and did not get it, so it is
falsified as stated while moving in the predicted direction. P21 confirmed:
pool moved -0.14% and local 40.6% to 40.9%, both inside a point, so the
coarser allocation granularity costs nothing at these lengths.

The useful by-product is a noise floor. c72 and c72fic have the same decode
path, since the MoE backend turned out to be a no-op, so their 2.2% tpot gap
is run-to-run variance. b64's 8.6% is about four times that and is a real
effect.

TTFT p50 is not. Three runs at the same working point gave 9.23, 6.43 and
8.50 s while `waiting_mean` moved only 3.32, 3.03, 3.13. A 30% spread on the
statistic against a 9% spread on the queue that produces it means TTFT p50
here has a wide confidence interval, and c72fic's 6.43 s must not be read as
a lever working. Nothing in this pair improved TTFT.

P22's trigger was "if both leave tpot flat". Only one did. Reporting it
straight: the 3x gap in the decode path is not the MoE kernel at all, and the
block table accounts for about 9% of it. The remainder is long-context paged
attention at 4 KV heads with TP=2, which is TP=4 territory and out of scope.
Per the pre-commitment, the search for a TTFT knob stops here and c72 is the
operating point.

Two things worth keeping:

- `BLOCK=64 FLOOR=2048` is free and strictly better on every axis that moved
  (tpot, aggregate decode, `lat_p50`, TTFT p90). It should be the default for
  subsequent arms even though it does not change the verdict.
- The working point reproduces. Three independent 1800 s runs gave local
  40.6 / 41.1 / 40.9% and ext 47.6 / 46.3 / 47.1%. Against the target of
  local 40-60% and ext >25%, that is a stable configuration, not a lucky run.

The honest summary of the target `local 40-60%, ext >25%, TTFT p50 <10 s`:
it is met, reproducibly, at CONC=72 on fp8 + 256k + native 262,144 + L1
320 GiB, and TTFT sits at 8.5-9.2 s with no margin to spare and no available
knob to buy more.

## Appendix. Reproducing sections 5 and 8

Harness lives in the session scratchpad, not the repo, so the two knobs the
new arms needed are recorded here rather than committed. Both are additive
and default to the previous behaviour, so every arm before `f8k256c72fic`
reproduces unchanged.

`up.sh`, two optional passthroughs (backups `up.sh.bak7`, `up.sh.bak8`):

```sh
BATCH_ARGS=()
if [ -n "${MAX_BATCHED:-}" ]; then
  BATCH_ARGS=(--max-num-batched-tokens "$MAX_BATCHED")
fi
...
  --block-size ${BLOCK:-16} \
  ...
  "${BATCH_ARGS[@]}" \
```

`MAX_BATCHED` was added for the step-budget lever and then never used, since
section 4 killed that lever before it ran. It is left in place unset.
`BLOCK` replaces the hardcoded `--block-size 16`. The banner and the snapshot
header carry `batched=` and `block=` so an arm's dtype, context, dataset,
MoE backend and block size are all recoverable from its own `snapshot.txt`.

`arm.sh` needed no change: it already forwards trailing assignments via
`env "$@"` and records them verbatim as `env='...'` in the snapshot header.

The three arms, each `setsid`-detached (see record 5's appendix for why):

```sh
SLOT=1 CONC=72 DATASET=semianalysis-cc-traces-weka-062126-256k \
  arm.sh lazy f8k256c72fic KV_DTYPE=fp8 MAX_MODEL_LEN=262144 ROPE_OVERRIDE='{}' \
  L1_GB=320 DEFER_SECS=30 FLOOR=8192 MOE_BACKEND=flashinfer_cutlass

SLOT=2 CONC=72 ... f8k256c72fit ... MOE_BACKEND=flashinfer_trtllm     # failed bringup

SLOT=2 CONC=72 DATASET=semianalysis-cc-traces-weka-062126-256k \
  arm.sh lazy f8k256c72b64 KV_DTYPE=fp8 MAX_MODEL_LEN=262144 ROPE_OVERRIDE='{}' \
  L1_GB=320 DEFER_SECS=30 FLOOR=2048 BLOCK=64
```

`FLOOR=2048` is not an independent change. The danger floor is counted in
blocks, so 2048 at block size 64 is the same 131,072 tokens as 8192 at 16.
Pairing them is what keeps `BLOCK` a single variable.

Mid-run sampling that produced the early read, before either snapshot landed:
two scrapes of `/metrics` 60 s apart on `vllm:inter_token_latency_seconds_{sum,count}`
at ports 27212 and 27222. It pointed the right way for `fic` and the wrong way
for `b64` (114.2 ms interval ITL against a 74.6 ms full-window `tpot_p50`),
because the instantaneous in-flight count during the sample was 36 against a
window mean of 30.1. Interval ITL is not comparable to `tpot_p50` and should
not be used to call an arm early.

The MoE backend is verifiable from the arm's own log rather than from the
flag, which matters because `auto` silently picks Triton:

```
Using FlashInfer CUTLASS Unquantized MoE backend out of potential backends:
['TRITON', 'BATCHED_TRITON', 'FlashInfer TRTLLM', 'FlashInfer CUTLASS']
```

miss.py crashed on both new arms and it was not the block size: it reads the
arm banner from `<tag>.log`, and these two were launched writing
`<tag>.launch.log`. Symlinking `<tag>.log` to it made the reuse-clock section
print normally. The token split above `presented ... = compute + local +
external` prints before that point, so those numbers were never affected.
