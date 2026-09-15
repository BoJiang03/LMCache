# Splitting packing out as its own PR, and getting the e2e to actually run

**Date:** 2026-09-15
**Branches:** `delta_token_ids` (working, `cf801298`), `delta_token_ids_pr` (PR, `e4986c82`, based on `dev` @ `c40b7807`)
**Follows:** [2_delta_and_packed_token_ids_implementation.md](2_delta_and_packed_token_ids_implementation.md)

## What this session decided

The previous session landed delta + packing as one 41-file, +3552/−212 change.
This session split it. **Packing goes first, on its own branch; delta stacks
on top later.**

The split is not about line count — the two halves are almost the same size —
it is about **risk asymmetry**:

- Packing changes no cache key at all (blake3 was already being fed exactly
  these bytes). Review is "is the codec symmetric", revert is clean.
- Delta carries every semantic risk in the change: gap degradation, the
  scheduler-side high-water mark, resync after an unhealthy server, TTL moving
  from creation to last use. Its tests outnumber its code.

## Sizing the two halves

Asked how much each axis costs. Keyword-matching the diff was useless (two
thirds of added lines carry neither side's vocabulary), so the diff was read
file by file and each half sized as a **counterfactual**: how big would the
PR be if only that half shipped.

| | alone |
| --- | ---: |
| packing | ~+490 / −110 |
| delta | ~+540 / −80 |

They sum to more than the combined +878/−149 in `lmcache/`: roughly **150
lines do double duty**, in `session.py`, `custom_types.py` and
`vllm_multi_process_adapter.py`.

### The rename is not the churn

Pushback worth recording: is `token_ids` → `token_bytes` worth "all that code
changing"? Measured it — rewrite every added line's `token_bytes` back to
`token_ids` and count how many then match a deleted line exactly:

```
added lines that become identical once the name is kept:   15
added lines mentioning the field:                         279
```

**15 lines.** The churn is the *type*, not the name: `token_ids=tuple(x)` has
to become `pack_token_ids(x)` whatever the field is called. And the type
cannot stay `list[int]`, because 74% of the win (1637 of 2197 ms) is in
`pickle.loads` and the msgspec decode — costs that only vanish if bytes are
what crosses the boundary.

What the pushback *was* right about: blast radius. Five peripheral modules
(sdk, cli bench, atom, sglang, tensorrt_llm) were each packing by hand. They
now go through `IPCCacheServerKey.from_token_ids`, which owns the packing, so
those modules never mention the codec.

## PR1 as it stands

`delta_token_ids_pr` @ `e4986c82`: **35 files, +706 / −185.**

Stripping delta out shrank the pieces unevenly, which is itself informative
about what belonged to which half:

| file | combined | packing only |
| --- | ---: | ---: |
| `session.py` | +173 | +37 |
| `engine_context.py` | +78 | +1 |
| `modules/lookup.py` | +73 | +7 |
| `lmcache_mp_metadata.py` | +168 | +67 |

`engine_context.py` collapsing from 78 lines to 1 is the clearest signal: the
`resolve_obj_keys_from_session` extraction existed solely so delta's central
invariant could be tested, and packing has no use for it.

Tests: `tests/v1/multiprocess` **781 passed, 1 skipped, 0 failed**.

## A bug in the delta half, found via an abandoned branch

Creating the branch surfaced `perf/mp-store-token-delta` (pushed 2026-09-04,
never opened as a PR, a local experiment). Its commit message names a call
site that must forward `token_offset` — and the current delta implementation
misses exactly that one:

```python
# lmcache/sdk/qringbuffer.py:700
key = self._adapter._create_key(
    op.token_bytes, op.start, op.end,   # op.token_offset not forwarded
    request_id=request_id, cache_salt=cache_salt,
)
```

Consequence is milder than in that older design but real: `absorb_tokens(0,
delta)` returns early as a no-op whenever the delta is shorter than what the
session holds, so the hashes stay correct **by accident**. Once the session
has been swept or recreated, `held` shrinks and the delta overwrites the whole
sequence at position 0. Q-ring configurations only. Belongs to PR2.

That branch also did something deliberately that the current implementation
undoes: it **left RETRIEVE on the full sequence**, because
`prepare_failed_retrieve_release` used whole-sequence equality to prove range
ownership. The current code deltas RETRIEVE too and weakens that check to
`_matches_tokens_locked(offset, bytes)` — slice agreement, not prefix
identity. Not obviously wrong, but it was a considered decision being
reversed silently, and PR2 has to argue it.

## The e2e harness had never been run

The previous record listed it as "written but not yet run". Running it turned
up four problems — three in the harness, one a misconfiguration:

1. `--disable-log-requests` was removed in vLLM 0.26. vLLM exited in under a
   second; only the LMCache server's 1.3 GB CUDA context showed on the GPU.
2. The readiness waiters polled `/health` without checking process liveness,
   so a dead server looked like a slow one for the full five-minute timeout.
   Fixed to take the Popen and log, and fail immediately with the log tail —
   **problems 3 and 4 were both diagnosed in seconds because of this.**
3. `terminate()` returned early when the direct child had exited. vLLM's TP
   workers are in the process group but are not this script's children, so an
   engine-core crash left four of them alive holding **125 GiB each**, and the
   next arm failed with "free memory 14.67/139.8 GiB". The group is now
   signalled regardless of the child's state.
4. Mine: `--max-model-len 131072` against Qwen2.5-7B, whose
   `max_position_embeddings` is **32768**. vLLM only warns; then a 120k prompt
   runs past the RoPE table and trips a device-side assert in an
   inductor-compiled kernel, killing the engine core mid-pass.

Plus an environment requirement now baked into the driver:
`CUDA_HOME=/usr/local/cuda-13.0`, or FlashInfer's JIT calls the system
`/usr/bin/nvcc` (CUDA 11.5), which rejects `compute_90a`.

## Numbers

### Microbenchmark, packing only (200k prompt, TP=8, per request)

| stage | x | list | packed |
| --- | ---: | ---: | ---: |
| A sched build payload | 25 | 16.5 | 3.9 |
| B1 sched `pickle.dumps` | 25 | 53.8 | 0.7 |
| B2 worker `pickle.loads` | 200 | 672.8 | 8.2 |
| C1 worker key+encode | 200 | 396.0 | 4.4 |
| C2 server msgspec decode | 200 | 836.9 | 5.2 |
| D server session hash | 200 | 124.3 | 6.6 |
| E server LOOKUP hash | 1 | 4.7 | 1.5 |
| **TOTAL CPU (ms)** | | **2105.0** | **30.5** |

Stage D is **worse than in the combined branch** (6.6 ms vs 0.2 ms): packing
alone calls `set_tokens` and copies the whole buffer per store, where delta's
`absorb_tokens` is a no-op once the range is covered. Measuring PR1 with the
combined branch's benchmark would have hidden this, which is why the
packing-only benchmark is a separate file rather than a column selection.

### End to end (30k x 16, concurrency 4, TP=4) — first scale

| | cold TTFT p50 | warm TTFT p50 | warm p99 | lmcache CPU | vLLM CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1095 ms | 156 ms | 202 ms | 2.52 s | 40.01 s |
| packed | 994 ms | 157 ms | 184 ms | 1.88 s | 38.85 s |

**No latency benefit at this scale, and that should not be dressed up.** Warm
p50 is 156 vs 157 ms. The cold p50 and warm p99 deltas are single-run n=16 and
are not claimed. The one clean signal is LMCache server CPU, **−0.64 s
(−25%)**, which matches the microbenchmark's prediction to within a factor of
two (~340 ms predicted from store ops alone; retrieves account for the rest).

The reason is scale, not the change: the connector's CPU is simply not on the
critical path at 30k / TP=4 / concurrency 4. The microbenchmark's 2105 ms is a
200k, TP=8 figure, and both factors are multiplicative.

A larger run (31k x 64, concurrency 16, TP=4) was in flight when this record
was written; 128k prompts would need YaRN rope scaling enabled explicitly on
this model, which was left alone as an extra variable.

## Where things are

- `delta_token_ids_pr` @ `e4986c82` — PR1, packing only, tests green.
- `delta_token_ids` @ `cf801298` — working branch; holds both benchmark suites
  (`token_ids_transport_*` for the combined change, `packed_token_ids_*` for
  PR1). Benchmarks deliberately live here and not on the PR branch.
- Still open: PR2 (delta, stacked, must fix the qringbuffer offset and
  re-argue RETRIEVE); a separate small PR for the pre-existing
  `resolve_prefetched_obj_keys` chunk off-by-one, which this work only
  noticed because delta forced a rewrite of that function.
