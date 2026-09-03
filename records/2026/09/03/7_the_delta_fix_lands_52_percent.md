# The delta fix lands 52% of loss #1 — and the first attempt voided itself

Branch: `fix_mp_store_key_prefix_resend_pr` (pushed to fork, commit `88ccf635`;
cherry-picked here as `50c6cf7f`). PR not opened.

Follows record 6, which named the defect and sized it with the `tinykey`
diagnostic. This one implements the real fix and measures it.

---

## Result

Step probe, differenced, TP=8, 1000 × 60k-token prompts. **All arms 7200
steps**, so `ms/step` is comparable (see "What voided the first run").

| arm | probe ms/step | end-to-end | Δ vs `none` |
|---|---|---|---|
| `tp8_none` (no connector) | 83.94 | 625.2 s | — |
| `tp8_nostore` | 85.71 | 639.0 s | +1.77 |
| **`tp8_mpfix`** | **87.72** | **654.3 s** | **+3.78** |
| `tp8_tinykey` (diagnostic) | 88.53 | 658.6 s | +4.59 |
| `tp8_mp` (before) | 91.90 | 686.0 s | +7.97 |

**Recovered 4.18 of 7.97 ms/step = 52%**, and 31.7 s of 60.8 s end to end.

The probe and the client clock agree on every arm:

- `Δ(mp − mpfix) = 4.18 × 7200 = 30.1 s` vs measured `31.7 s`
- `Δ(mpfix − none) = 3.78 × 7200 = 27.2 s` vs measured `29.1 s`

It beats the `tinykey` upper-bound diagnostic (88.53). That was pre-registered
in chain25's header as the reading meaning "it also took back part of the
scheduler-side +1.77, because the connector metadata vLLM broadcasts every step
got smaller too" — which is what the code does and `tinykey` did not.

## What the fix is

`lmcache_mp_metadata.py:GetStoreMetadata` put `tracker.get_token_ids()` — the
request's whole grown prefix — into every `LoadStoreOp`, with `start`/`end`
indexing into it. Stores now carry only `[start, end)` plus `token_offset`.

The prefix was pure retransmission. `Session._compute_hash` is already
incremental (`num_chunks_processed`, `last_prefix_hash`, `chunk_hashes`); it
only ever hashes chunks it has not seen. It needs the new tokens, nothing else.

Measured on the real dataclass:

| | wire | `tuple()` | msgspec encode |
|---|---|---|---|
| whole prefix, N=60000 | 179,747 B | 0.241 ms | 0.333 ms |
| delta, N=8192 | 24,327 B | 0.029 ms | 0.046 ms |

A payload written without `token_offset` decodes as 0 — the whole-prefix
meaning — so the wire stays forward compatible.

LOOKUP and RETRIEVE are unchanged: they still send the whole prefix with
`token_offset == 0`, which `Session.prepare_failed_retrieve_release` relies on
(it proves range ownership by comparing `key.token_ids` against the lookup
key's). Scoping the change to STORE keeps that comparison valid — and STORE is
the only per-step caller anyway, at 0.96 calls/step against retrieve's
once-per-request.

## Correction to records 5 and 6

I wrote that the msgpack encode happens **on the worker's model-execution
thread**. It does not. `process_outbound_task` has exactly one callsite in the
tree, `mq.py:247`, inside `ClientPollingLoop._main_loop` — the
`mq-client-shared-loop` daemon thread.

The cProfile caller table appeared to say otherwise, but that table is garbage
for this frame: it lists 0.96 calls/step against callers that each show 0.00,
the same cross-thread corruption signature as `poll.py:80(poll)` in record 4.

This does not weaken the finding, it sharpens it. The encode still costs,
through the **GIL**, not through direct serialization on the hot thread —
`msgspec._core.msgpack_encode` does not release it. That is why "move it to a
background thread" was never the fix: it was already on one. It also explains
the ~3× amplification as GIL contention × 8-way TP lockstep max, rather than
additive latency.

## What voided the first run

The first `tp8_mpfix` read **81.71 ms/step — faster than the no-connector
baseline**. That is impossible, and chasing it is what found the bug.

Two tells:

1. It ran **7800 steps**; every other arm ran 7200. Steps are capped at 8192
   tokens but not equal to it, so `ms/step` across arms with different step
   counts is not a comparison. chain25 now prints step counts before the table.
2. `lmcache_server.log` held **6599 `Skipping STORE`** lines. The arm had
   degenerated into `nostore`.

The gap distribution named the cause outright — `held` was always exactly one
chunk behind `offset`:

```
offset  held    count
16384   8192      664
24576   16384    1713
32768   24576    1917
40960   32768    1760
49152   40960     544
49152   0           1
```

`extend_tokens` replaced the token list wholesale when `token_offset == 0`. All
eight TP ranks store the same ranges against **one shared session**,
asynchronously. A straggler rank's `[0, 8192)` delta truncated the list back to
one chunk; every rank already further along then found its next offset past the
end and failed as a gap — and stayed failed, because each retry truncated
again.

The fix: splice, never shorten. `end >= held` extends, otherwise overwrite in
place and keep the tail.

**My unit test did not catch this because it modelled a single rank.** The
repo test added with the fix
(`tests/v1/multiprocess/test_session_token_delta.py`) now interleaves 8 ranks
with per-rank order preserved. It discriminates: the truncating version
produces **242 gaps** across 5 seeds, the spliced version **0**.

## Two things worked as designed

- **The graceful gap path.** 6599 skipped stores, zero hangs, the run completed
  normally. `return b"", False` rather than raising matters: `mq.py:636` logs
  handler exceptions and sends no response, so a raise would have left every
  client future pending forever. Second run: **1 gap in ~7000 stores**.
- **The correctness arm as an instrument.** It reported warm ≈ cold on the
  broken build, independently of the perf arm.

## Correctness

vLLM logs LMCache hits as `External prefix cache hit rate`. Warm pass, 16
prompts, `APC=0` so vLLM's own prefix cache cannot serve it:

| build | external hit rate |
|---|---|
| `tp8_mpfix` | **2.8%** |
| voided build (stores skipped) | **0.0%** |

Non-zero hits with the fix, zero without stores — the delta keys resolve to the
same objects end to end. The rate is low only because L1 evicted 18 times: 16 ×
60000 tokens does not fit the arm's 32 GB. That is the arm's capacity, not a
key problem. A high-hit-rate demonstration would need a smaller working set;
not run.

The hash-chain equivalence itself is proved offline, not inferred: delta stores
and whole-prefix stores produce identical chunk hashes, with and without a
preceding lookup, at hit_chunks ∈ {0, 1, 5}, under 8-rank interleaving, and the
spliced session's chain reproduces a fresh whole-prompt session's exactly.
`ipc_key_to_object_keys` reads only `model_name`, `world_size`, `worker_id`,
`cache_salt` and the chunk hashes — never `token_ids` — so identical hashes
mean identical storage keys.

## What is left

`+3.78 ms/step` still stands between `mpfix` and no connector, and `nostore`
puts a floor of `+1.77` under it (scheduler-side LOOKUP and connector
metadata). So roughly:

```
+1.77  scheduler-side, floor -- still not decomposed at function level
+2.01  store submission beyond the key size -- event IPC export, MQ, futures
```

The `+1.77` is the honest open item. I have only one datum on it: `_pickle.loads`
is 1.25 ms/step in both `mp` and `nostore`, i.e. it is not the mp-vs-nostore
delta. Whether it constitutes the `none`→`nostore` delta needs a `none` profile,
which was never taken.

Lever (c) from the analysis is untouched: the store key is byte-identical
across all 8 ranks bar `worker_id`, and the scheduler already owns an MQ
channel to the server (`vllm_multi_process_adapter.py:794`). Sending the delta
once from the scheduler and having workers ship only a reference would take the
worker's per-step Python to tens of microseconds.

## Tests

- `tests/v1/multiprocess/` + `tests/v1/test_vllm_mp_adapter.py`: **770 passed,
  1 skipped** on the PR branch (755 before, +15 new).
- `ruff check` and `ruff format --check` clean.
- One test fixture needed updating: `test_blend_observability.py` fakes a key
  with `SimpleNamespace`, so it had to gain `token_offset=0`. The blend store
  path reads it.
