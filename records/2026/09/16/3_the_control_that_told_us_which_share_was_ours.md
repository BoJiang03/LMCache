# The control that told us which share of the win was ours

2026-09-16, continuing from `2_the_prefill_throughput_result_and_what_it_cost_to_believe_it.md`.

That record established a reproducible 1.30x-1.54x prefill-throughput result at
`--long-prefill-token-threshold=256` and called that flag an "amplifier". This
one keeps the measurement and replaces the claim built on it. Two runs did the
damage: a threshold sweep, which falsified the scaling law I had just written
into the PR description, and a no-connector control, which showed that most of
the throughput the A/B appeared to recover was never LMCache's to lose.

Bo asked both questions. Neither was on my list.

## What was already written down, and wrong

After the previous record I drafted the PR1 description. It contained this:

> the benefit of this change scales with context length squared and inversely
> with that threshold

The reasoning: every store op carries the whole token prefix, so a request of
length `L` advancing `T` tokens per step broadcasts about `L^2 / 2T` token ids
over its lifetime. At 200k x 12 that is 938M token ids at `T=256`, 117M at
`T=2048`, 29M at the default, a 32x span. The arithmetic is right. I applied it
to wall-clock time, which is where it fails.

Predicted from it, and stated before running: 69-95 s of wall saving at `T=256`
becomes 2-3 s at the default setting and 9-12 s at 2048.

Measured: 0.1 s at both. The 2048 prediction was off by two orders of magnitude.

## The sweep

Same configuration otherwise: 200k x 12, c12, TP=8, `--enforce-eager
--max-num-seqs=64`, `fp8_ds_mla`, L1 450 GB, prompt seed 20260915.

    threshold   arm        cold total   warm p99   server CPU
    default     baseline       108.6s     9803ms        72.6s
    default     packed         108.4s     9059ms        69.4s
    2048        baseline       109.8s     7170ms        78.3s
    2048        packed         109.7s     6778ms        66.5s
    256 (v2)    baseline       256.6s     3213ms       204.0s
    256 (v2)    packed         160.8s     3102ms        98.0s
    256 (v3)    baseline       209.5s     3385ms       207.4s
    256 (v3)    packed         140.0s     3366ms        86.3s

The model holds for server CPU and only for server CPU. Predicted 32x / 4x / 1x;
measured savings 33-38x / 3.7x / 1.0x, that is 106-121 s, 11.8 s, 3.2 s. Across
a 32x span that is as good a fit as this setup can produce.

Wall clock is a cliff, not a slope: nothing at the default setting, nothing at
2048, 33-37% at 256. Why remains unexplained. The reading I find plausible is
that the connector's per-step cost is normally hidden behind GPU execution and
only becomes visible once it approaches it, since at `T=256` each step does a
third of the GPU work while emitting twelve times the ops. It is a hypothesis.
It is labelled as one in the PR description, and nothing claimed depends on it.

A side effect worth keeping: within a pair the two arms reproduce to 0.1-0.2%
at the default and 2048 settings. So the earlier "run-to-run noise is 15-20%"
was wrong as stated. That 15-18% is drift *between* pairs run about twenty
minutes apart; *within* a pair it is 0.2%. The nulls at default and 2048 are
therefore real negatives, not a resolution limit, and the 33-37% at `T=256` is
enormous against within-pair precision.

## The control, which is the actual finding

Bo's question: why does `T=256` hurt throughput so much, and has anyone measured
what it costs without LMCache in the picture? The answer to the second was no.
Every number above compares two arms that both carry a connector, so vLLM's own
cost of the setting is common-mode and cancels, and is therefore invisible.

`packed_token_ids_e2e.py` gained `--connector {lmcache,none}`. The `none` arm
starts vLLM with no `--kv-transfer-config` and no LMCache server. An enum rather
than a boolean because it is a third arm, not the absence of one, and the arm is
recorded in the result JSON.

Stated before the run: vLLM alone at `T=256` should land at 140-155 s, because
781 scheduler steps instead of 293 under `--enforce-eager` means about 488 extra
steps of fixed per-step Python overhead. Ruled out at the same time: KV re-reads,
which do scale as `L^2/2c` exactly like the token-id volume, but at MLA's
`fp8_ds_mla` footprint amount to about 540 GB across twelve requests, under 0.2 s
of HBM time. Attention FLOPs are independent of chunk size.

Measured 134.4 s. Close enough to the prediction, and it decomposes everything:

    threshold  arm                    cold total   over control
    default    no connector             104.2 s
    default    LMCache baseline         108.6 s      +4.2%
    default    LMCache packed           108.4 s      +4.0%
    256        no connector             134.4 s
    256        LMCache baseline         209.5 s      +55.9%   (v2: 256.6, +90.9%)
    256        LMCache packed           140.0 s      +4.2%    (v2: 160.8, +19.6%)

Lowering the threshold to 256 costs vLLM **1.29x by itself**. No connector change
touches that.

What survives, and it is better than what it replaces: with packing the
connector costs about 4% of prefill wall time at both threshold settings, +4.2 s
on 104.2 and +5.6 s on 134.4. Without it, 4% at the default setting and 56% to
91% at 256. **The connector's cost stopped depending on how finely prefill is
chunked**, and the 4.2% reproduces independently at two settings, which is the
strongest internal check in this whole line of work.

## What I had to take out of the PR description

- "the benefit scales as `L^2/2T`" -- true for server CPU, false for wall clock.
- "`--long-prefill-token-threshold=256` is an amplifier, not a trick" -- it is
  also, separately, an expensive setting for vLLM, and calling it an amplifier
  hid that.
- "lowering it costs 58% of throughput, and with this change 27%" -- this was the
  worst of the three. The 27% residual is almost entirely vLLM's 1.29x, not ours.
  Quoting it credited this change with work it does not do, in the direction that
  flatters it.
- "run-to-run noise is 15-20%" -- between pairs, not within.

Also corrected earlier in the session, before the sweep: the `L^2/2T` model was
first stated as scaling with concurrency. It does not. Concurrency decides how
many steps the volume is spread over, not the volume.

## What the microbenchmark says, restated correctly

The per-request rollup in `packed_token_ids_results.txt` multiplies the
server-side stages by 200, which is 25 steps x 8 ranks. That is the GQA shape.
Under MLA only the first rank per node is a KV writer
(`vllm_multi_process_adapter.py:384-395`), so a store reaches the server once,
not once per rank. Rather than fix the caption, the PR description drops the
rollup and states the multiplier rules instead:

- build payload, `pickle.dumps`: once per step, scheduler process.
- `pickle.loads`: once per step **per TP rank**, because `SchedulerOutput` is
  broadcast to all of them. Independent of the attention implementation, and the
  dominant term.
- key + encode and everything server-side: once per step per **KV writer**.

The one real regression: `pickle.dumps` netted against payload construction on
the scheduler thread crosses over around 2-3k of context, +1.9 us/op at 1k. The
existing commit message says "crossover around 16k-32k", which is stage A alone
and more pessimistic than the net. That message needs fixing.

## State

Seven A/B pairs plus two control runs, no regression in any of them. Commits on
`delta_token_ids`, none pushed:

- `61e25e39` the control arm, the sweep, six result JSONs
- `b4700dc9`, `2eda2577` from the previous record

PR1 description drafted at 198 lines in the job scratch directory, not in the
repo. Still open: PR1 needs the benchmark files moved onto it (`e4986c82`'s body
references a microbenchmark that is not on that branch) and the crossover figure
corrected; then PR2 (delta) stacked on it, and the separate
`resolve_prefetched_obj_keys` off-by-one.

Task list cleared to empty. Everything in it was either done or recorded here.
