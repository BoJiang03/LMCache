# Lazy offload: deferred stores were invisible to the next request

Commit `5ea3cc6e`, on top of `924e2c1c`. Continues record 5 (the trace-driven
A/B that found the defect) and record 6 (session index).

## What was wrong

`LMCacheMPRequestTracker.num_stored_tokens` -- the "already safe, do not
re-store" watermark -- has exactly one cross-request source: an LMCache
lookup hit (`lmcache_mp_connector.py:814`). A lookup reports what the
**server** holds. In lazy mode the covering store is still buffered in the
scheduler process and the server has never heard of it, so:

1. Turn N buffers its store ops; they sit pending for the eviction horizon.
2. Turn N+1 arrives over the same prefix. vLLM's APC serves it, so
   `computed_tokens` jumps to the full prefix, but its LMCache lookup
   misses (`ret == 0`, early return at connector:811).
3. `num_stored_tokens` is therefore 0, so `GetStoreMetadata` computes
   `start_token_idx = 0` and stages `[0, full prefix)` as **one op**.

The APC hit is one-directional: it raises the willing-to-store upper bound
without raising the already-safe watermark.

The dedup that should have caught this is an exact-match hash --
`_content_key = (cache_salt, prefix_end_tokens, tuple(block_hashes))` --
so a longer follower range can never equal a shorter pending one. Measured:
`deduplicated=4` out of 979 admissions.

### The cost is not bandwidth

This is the correction that mattered. The server already filters the
redundant range **before** any copy: `prepare_store` reserves with
`reserve_write(mode="new")` (`server_transfer.py:325`), an existing key
returns `KEY_NOT_WRITABLE` and gets no shm slot (`l1_manager.py:478`), and
the worker copies only the chunks that got slots (`shm.py:182` even has an
explicit all-cached zero-copy path). Two overlapping *pending* ops are
filtered the same way whichever drains first.

The `2.5x` figure in record 5 came from summing the server's
`"Stored N tokens"` log line, which counts the **requested range**
(`len(obj_keys) * chunk_size`, `engine_driven_transfer.py:468`) regardless
of how many chunks were actually written. Actual copy volume equals the
final L1 object count (zero watermark events, `mode="new"` never
overwrites). Lazy actually copied *less* than eager, and total store time
was lower (4.8 s vs 7.7 s).

What the redundancy really cost was the work paid to *reach* that filter:
hashing the full range, one reservation round-trip per chunk, a large mq
payload, drain budget spent on ops that write ~nothing, and an oversized
atomic store on the per-instance affinity thread -- where stores and
retrieves are the same FIFO lane (`affinity_pool.py:5-9`, handlers at
`engine_driven_transfer.py:118-135`), so retrieves on the TTFT critical
path queue behind it.

## The fix

`_PendingOperations` now indexes the block content live pending ops cover,
refcounted by `(cache_salt, block_id, block_hash_snapshot)`. The hash is
part of the **key**, not a stored value, so a stale snapshot can never be
mistaken for current content.

`EvictionAwareStoreQueue.covered_prefix_tokens()` walks a request's blocks
in prefix order, takes the leading covered run, mins across engine groups,
and floors to a chunk boundary. The connector calls it via
`_skip_pending_covered_prefix()` immediately before `GetStoreMetadata`,
under `if self.lazy_offload:` -- the eager path is untouched, and a test
asserts it never consults.

Two properties worth keeping in mind when reading it:

- **Probing starts at the caller's watermark, not at zero.** Walking from
  zero every step would be O(prefix) per scheduler step on the critical
  path -- thousands of probes for the 90k-token prefixes this exists for.
  From the watermark, each block is paid for once per request lifetime.
- **Drop releases the cover.** When an op dies its range leaves the index,
  so the *next* request stages it again. A request that already skipped
  does not, which can leave prefix-orphaned chunks in L1 -- wasted space,
  never wrong data. Predecessor and follower share the same GPU blocks, and
  a live follower pins them, so the correlated case is unlikely; the run
  showed no retrieve regression (retrieves went *up*).

New counters, outside the admission ledger on purpose (a skipped range
never becomes an op, so it belongs to neither `admitted` nor any drop
counter): `covered_prefix_advances`, `covered_prefix_tokens_skipped`
(effect) and `covered_blocks_probed` (cost).

18 new tests: policy-level (shared prefix, salt isolation, stale snapshot,
hole truncation, chunk alignment, least-covered group, watermark-start
probing, counters, argument validation, drop and emission both releasing
the cover) and connector-level (follower range starts at the covered end
on both the new- and cached-request paths; eager never queries).

## Measured, same 900 s scenario-native A/B

Eager reproduced the earlier run closely (stores 968 vs 949, tokens_stored
1,606,144 vs 1,601,792, store p99 0.042 vs 0.033 s), so the baseline is
stable and `1,606,144 / 256 = 6274 = l1_objects` exactly -- eager's write
amplification is 1.000 for the second run running.

| | eager | lazy before | lazy after |
|---|---|---|---|
| tokens_stored | 1,606,144 | 3,823,616 | **1,608,704** |
| vs eager | 1.000x | 2.381x | **1.0016x** |
| staged chunks / L1 resident | 6274 / 6274 | 14936 / 5946 | 6284 / 5868 |
| write amp | 1.000 | 2.512 | **1.071** |
| retrieves / tokens_retrieved | 41 / 1.33M | 62 / 1.93M | 56 / 1.46M |

TTFT, within-run comparison:

| | before | after |
|---|---|---|
| avg | +12.1% | **+3.2%** |
| p50 | +4.0% | **+0.0%** |
| p90 | -- | **-2.3%** (lazy faster) |
| p99 | +19.2% | +17.4% |

`covered_prefix_advances=133`, `covered_prefix_tokens_skipped=4,267,520` --
**32,087 tokens per advance**, i.e. it is catching whole accumulated
prefixes, which is the signature the mechanism predicts.

The query is net *cheaper* than what it removes: `covered_blocks_probed`
is 5.6 per drain step, while `blocks_validated` fell from 32.6 to 9.0 per
step (-72%) because the ops it validates are smaller.

Cache quality, same write volume:

| | written | retrieved/req | L1 | yield |
|---|---|---|---|---|
| eager | 1,606,144 | 5,361 | 147.1 GiB | 0.831 |
| lazy | 1,608,704 | **5,891** | **137.5 GiB** | **0.905** |

Lazy retrieves 9.9% more per request from a 6.5% smaller L1 at equal write
volume. Record 5's "yield 0.506, lazy worse" used the inflated log-line
denominator and had the sign backwards.

## Retracted this session

- **"The shared prefix is transferred twice; bandwidth is wasted."** No --
  the server dedups before the copy (see above). The cost is request-side
  and round-trip work, not PCIe. Record 5's `2.5x` was a measurement-unit
  error on my side.
- **"The fat op head-of-line-blocks retrieves and causes the p99 tail."**
  Not supported. In the post-fix run the largest store (92,928 tokens) took
  0.069 s while a 57,600-token one took 1.002 s and its near-twin (57,344)
  took 0.019 s; size/duration Pearson r = 0.32 over 201 stores. The 1.0 s
  event is a stall of some other kind. **Do not implement the op-splitting
  fix on this rationale.**
- **"lazy 全面超越 eager" (the 3.3x first A/B).** Already retracted in
  record 6; restating because it keeps resurfacing. It was an artifact of
  eager hitting an LRU sequential-scan pathology under synthetic timing,
  not a lazy win.

## Still open

- **`max_drain_per_step` counts operations, not work**
  (`eviction_aware.py:342` documents it as bounding the D2H burst; op size
  is unbounded). Still a real defect, but the evidence that motivated
  fixing it now is retracted, so it needs its own justification first.
- **The +17.4% p99 is not established.** Eager's own p99 moved 6,598.7 ->
  9,378.2 ms (+42%) between two identical configurations. One sample
  cannot resolve a 17% gap through that noise. Repeat, or go straight to
  the regime that matters.
- **Residual write amp 1.071** (6284 staged vs 5868 resident) mixes
  leftover duplicates with `dropped_evicted=139` content that never landed;
  not separated yet.
- **The regime question.** On this configuration lazy structurally cannot
  win: L1 200 GB never fills (watermark events 0 both arms), PCIe duty
  cycle <1%, arrival-limited at ~2.4 effective concurrency. The "fewer
  copies" ceiling is GPU pool / working set = 24/147 = 16%; measured 6.5%.
  The quality advantage (yield 0.905 vs 0.831) is real but has nothing to
  convert it into latency. Proposed next: keep this pair as the regression
  arm and add **eager / lazy @ L1=60 GB** (L1:GPU = 2.5:1, below the
  ~147 GiB working set) as the benefit arm, scenario-native timing, 1800 s.

## Test status

- lazy suite: 214 passed
- `tests/v1` full: 1206 passed, 48 skipped; 1 pre-existing `caplog` error in
  `tests/v1/distributed/test_valkey_l2_adapter.py`, reproduced on the
  unmodified tree
- `ruff check` and `ruff format --check` clean on every touched file
  (`tests/v1/multiprocess/test_blend_v3_load_store_opts.py` was reformatted
  by an over-broad `ruff format tests/v1/` and reverted)

## Artifacts

`scratchpad/smoke3/` (session `84352f47`): `ab_trace.sh` (adds the per-store
size/duration distribution to the snapshot), `abt_eager/`, `abt_lazy/`.
Server logs remain under session `7445f449`'s `smoke/logs/` because `up.sh`
truncates per arm.
