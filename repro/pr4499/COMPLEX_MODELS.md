# Additional architecture matrix

This is a one-off validation supplement for PR #4499. It is intentionally kept
on the reproduction branch rather than in the production PR diff.

## Environment

- Production commit: `8e4e851f91316bb7994be3d096966f0d1ef0b52b`
- GPU: NVIDIA H200
- Lazy configuration: production default `EVICTION_AWARE`, horizon `2.5`
- Gemma model: `google/gemma-3-12b-it` (hybrid sliding/full attention,
  multimodal model family)
- DeepSeek model: `deepseek-ai/DeepSeek-V2-Lite-Chat` (MLA + MoE)

Each performance row below is a same-machine A/B with the same request stream,
GPU block budget, and L1 budget. Coverage is
`(vLLM APC hit tokens + external hit tokens) / queried tokens`; the two hit
ranges overlap by construction only at their boundary, as reported by the
harness counters.

## GSM8K retrieval correctness

Sixty fixed questions were run cold and then cached. The goal is not to compare
model quality between architectures; it is to verify that external KV retrieval
preserves answer behavior and has non-vacuous cache coverage.

| model | mode | cold strict | cached strict | cached coverage | cached time |
|---|---:|---:|---:|---:|---:|
| Gemma 3 12B | eager | 0.900 | 0.900 | 0.961 | 56.18s |
| Gemma 3 12B | lazy | 0.900 | 0.900 | 0.961 | 56.94s |
| DeepSeek V2 Lite | eager | 0.650 | 0.633 | 0.961 | 19.87s |
| DeepSeek V2 Lite | lazy | 0.633 | 0.617 | 0.944 | 19.43s |

Gemma's cached final answers are identical between eager and lazy. DeepSeek's
independent engine runs differ on three of 60 cached final answers; its cold
runs also differ on three, and both eager and lazy lose one strict answer from
cold to cached. This is consistent with the model's run-to-run MoE numerical
variation rather than a lazy-only cache regression. Neither run reports a
traceback, preemption, failed store, or request-drop loss.

Two byte-level probes add a stronger content check:

- Gemma 3 hybrid-attention probe: all 22 lazy chunks under comparison are
  byte-identical to eager, with identical greedy outputs.
- DeepSeek V2 MLA probe: all 15 lazy chunks under comparison are byte-identical
  to eager, with identical greedy outputs. The original Qwen-calibrated probe
  reports a failed quantity floor because eight operations remain pending
  without GPU pressure; its key inclusion, byte equality, and output equality
  assertions pass.

## Hot/cold performance

The query phase is 120 long-document requests with a 75% hot / 25% cold mix.
The GPU block budget holds the hot working set but not the complete rotation;
L1 is sized so eager writes create eviction pressure.

### Gemma 3 12B

Configuration: aggregate hybrid block budget `27,500`, L1 `112 GiB`.

| mode | wall-time runs | median wall | coverage runs | L1 eviction cycles | hot TTFT p50 median | cold TTFT p50 median |
|---|---:|---:|---:|---:|---:|---:|
| eager | 51.29s, 51.45s | 51.37s | 0.725, 0.725 | 14, 14 | 155ms | 943ms |
| lazy | 36.21s, 32.48s | 34.35s | 0.936, 0.971 | 2, 1 | 203ms | 393ms |

Median wall time decreases by **33.1%** (`1.50x` throughput-equivalent
speedup). Lazy makes hot TTFT about 48ms slower in this regime but cuts cold
TTFT by about 550ms and avoids most L1 eviction cycles.

Gemma's sliding-window null blocks are visible in the policy ledger:
`rejected_unhashed=14/37` and `rejected_prefix_broken=1/0` across the two lazy
runs. These are safety rejections, not failed stores: vLLM leaves hash-less null
blocks after positions leave the sliding window, and the lazy path refuses to
store those stale/nonexistent KV ranges. `dropped_failed_store=0` and
`dropped_on_request_drop=0` in both runs.

### DeepSeek V2 Lite

Configuration: GPU block budget `9,102`, L1 `9 GiB`.

| mode | wall-time runs | median wall | coverage runs | L1 eviction cycles | hot TTFT p50 median | cold TTFT p50 median |
|---|---:|---:|---:|---:|---:|---:|
| eager | 32.42s, 30.65s | 31.54s | 0.731, 0.729 | 13, 13 | 219ms | 548ms |
| lazy | 27.26s, 27.04s | 27.15s | 0.848, 0.870 | 7, 7 | 227ms | 486ms |

Median wall time decreases by **13.9%** (`1.16x` throughput-equivalent
speedup), with higher coverage and six fewer L1 eviction cycles. Both lazy runs
have zero unhashed/prefix-broken rejection, failed store, request-drop loss, and
preemption.

## Artifacts

Raw JSON and byte-probe logs are in [`results/complex_models/`](results/complex_models/).
The only retained warnings outside expected policy safety messages are model
startup warnings present in both eager and lazy modes (`disable_chunked_mm_input`
for Gemma and missing tuned MoE config for DeepSeek). Structured benchmark runs
contain no tracebacks.
