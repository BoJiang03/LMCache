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

Arms run at c72, one variable each, predictions P15-P22 pre-stated. Neither
lever moved the roof.

| arm | MoE / block | inflight | tpot | agg decode | TTFT p50 | p90 | local | ext |
|---|---|---|---|---|---|---|---|---|
| c72 | triton / 16 | 30.22 | 81.6 ms | 370 tok/s | 9.23 s | 36.86 s | 40.6% | 47.6% |
| c72b64 | triton / 64 | 30.11 | 74.6 ms | 404 tok/s | 8.50 s | 35.58 s | 40.9% | 47.1% |
| c72fic | fi-cutlass / 16 | 30.16 | 83.4 ms | 362 tok/s | 6.43 s | 40.65 s | 41.1% | 46.3% |

FlashInfer CUTLASS moved aggregate decode from 370 to 362 tok/s, which is
noise and on the wrong side of it. Block size 64 bought 8.6% of tpot, real
but two orders short of the 3x on the table. P22 fires: the roof is neither
the MoE kernel nor the block table, and what is left is long-context paged
attention at 4 KV heads with TP=2. That is TP=4 territory and out of scope,
so c72 is the operating point and there is no third knob.

`f8k256c72fit` (flashinfer_trtllm) failed bringup and scores nothing:
`Unquantized MoE backend FlashInfer TRTLLM does not support the deployment
configuration since kernel does not support current device cuda`.

## 6. Three replicates, and which numbers actually reproduce

The failed lever hunt left something more useful: three runs of the same
working point, differing only in a knob that did not work.

    local     40.6%   40.9%   41.1%      spread 0.5 points
    ext       47.6%   47.1%   46.3%      spread 1.3 points
    TTFT p50   9.23 s  8.50 s  6.43 s    spread 2.8 s
    TTFT p90  36.86 s 35.58 s 40.65 s    spread 5.1 s

The tier numbers are tight. TTFT p50 is not, and its run-to-run spread is
larger than either lever's claimed effect, which is why neither p50 gain
above is attributable. The defensible statement about this working point is
that it delivers local ~41%, ext ~47%, and TTFT p50 in the 6-9 s band, under
the 10 s target in 3 of 3 runs, with p90 in the 36-41 s band.

That also revises record 5 section 6, which called c72's p50 target "met
without margin" off a single run. Three runs say the margin is real but thin,
and that p90 is where this working point is genuinely uncomfortable.

## 7. L1 over L0 is already in range

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

## 8. Open

1. `max_deferral_seconds` still deserves its own PR. Unchanged from record 4.
2. Not attempted, and named so it is not silently dropped: nothing here
   measures the exclusive move-not-copy retrieve path, which is the second
   half of the method. Every arm to date still copies L1 to L0.
3. A harness note, in case it misleads a later reading: miss.py crashed on
   both round-3 arms. It was not the block size. It reads the arm banner from
   `<tag>.log` and round 3 wrote `<tag>.launch.log`.
