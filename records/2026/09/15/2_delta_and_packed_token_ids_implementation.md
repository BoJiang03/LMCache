# Delta + packed `token_ids`: implementation, microbenchmark, e2e harness

**Date:** 2026-09-15
**Branch:** `delta_token_ids` (worktree `/raid/bo/delta_token_ids`, based on `dev` @ `c40b7807`)
**Commit produced:** `70c31e5a` — `mp: send only each op's tokens, packed, instead of the whole sequence`
**Follows:** [1_connector_token_ids_transmission_overhead_at_200k.md](1_connector_token_ids_transmission_overhead_at_200k.md), which measured the problem

## Goal

Implement the delta change the previous record left as "next step", then
measure it, then run it end to end. Mid-session the scope grew to a second,
independent optimization (packing) after measuring that it was the larger
win.

## What shipped

Two orthogonal changes, both on by default.

### 1. Delta ranges

A store or retrieve op used to carry the request's **whole** token sequence
even though it only covers `[start, end)`. Now it carries only its own
range, at a new `token_offset`, and the server chains it onto the prefix its
session already holds from that request's LOOKUP.

The enabling idea: **`Session` becomes the server's single source of truth
for a request's token sequence.** Everything that needs tokens or chunk
hashes — `resolve_obj_keys`, `resolve_prefetched_obj_keys`,
`_publish_token_bindings`, blend's store leg, the `same_lookup` ownership
checks — reads them back from the session instead of from the key. That is
what makes a delta key and a full key resolve to *identical* ObjectKeys.

Safety, in the order it matters:

- **Gap ⇒ no keys at all.** `Session.absorb_tokens(offset, packed)` returns
  `False` when `offset` is past what the session holds. Chunk hashes are
  prefix-chained, so hashing across a gap would mint valid-looking keys for
  content that was never stored. Callers return an empty key list instead:
  a store writes nothing, a retrieve reports failure and vLLM recomputes.
  Both are existing degradation paths.
- **Never raise.** Server-side handler exceptions are logged and the frame
  dropped, which would leave the client's future unresolved — a hang, far
  worse than a cache miss. So the gap path degrades, it does not throw.
- **Scheduler-side high-water mark.** `LMCacheMPRequestTracker` tracks how
  many tokens the server is known to hold (seeded by the LOOKUP's return
  value) and sends the full sequence whenever it cannot prove the prefix is
  there. `build_op_tokens` makes that decision.
- **Health recovery.** A server that went away may come back without its
  sessions, so once the connector has seen it unhealthy,
  `_resync_server_token_state` clears every tracker's mark on the recovery
  edge; the next op per request re-seeds it.
- **TTL now runs from last use, not creation.** Previously a session expired
  600 s after `created_at`; a request generating for longer would lose the
  prefix its own deltas chain onto. Now `last_used_at` is refreshed on every
  absorb and hash.

Switch: `lmcache.mp.delta_token_ids` (default true).

### 2. Packed tokens

`IPCCacheServerKey.token_ids: tuple[int, ...]` → `token_bytes: bytes`, and
`LoadStoreOp.token_ids` → `token_bytes`: big-endian `uint32`, one word per
token (`lmcache/v1/multiprocess/token_codec.py`).

**This does not change a single cache key.** `_make_blake3_hash_func` already
hashes `struct.pack(f">{n}I", *tokens)` — the packed buffer *is* the byte
string blake3 was already being fed. Verified two ways: a unit test asserting
`compute_packed_chunk_hashes == compute_chunk_hashes`, and a standalone check
that the digests match bit for bit.

- `TokenHasher` gains `hash_packed_chunk` / `compute_packed_chunk_hashes`.
  blake3 takes the buffer as it arrives; any other algorithm falls back to
  unpacking, so it still works, just without the speedup.
- `Session` holds a `bytearray`, so splicing and hashing are memcpy-cheap.
  Only callers that genuinely need Python ints (`tokens_in_range`, blend's
  fingerprint matcher) unpack.
- The scheduler keeps a **packed prefix per request**
  (`LMCacheMPRequestTracker.packed_tokens`), so packing costs only the
  step's new tokens (57 µs for an 8192-token step) rather than the whole
  context (1.4 ms at 200k) on every step.

## Numbers (microbenchmark, 200k prompt, TP=8, per request)

`benchmarks/microbenchmark/token_ids_transport_benchmark.py`, now a full 2x2
so each axis can be read on its own:

| | full | delta |
| --- | ---: | ---: |
| **list** | 2218.0 ms / 170.79 MB | 79.9 ms / 7.06 MB |
| **packed** | 21.4 ms / 171.74 MB | **6.8 ms / 7.11 MB** |

Per stage (ms per request, multiplied by 25 scheduler steps and, for
worker/server stages, by 8 TP ranks):

| stage | x | list+full | list+delta | packed+full | packed+delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| A sched build payload | 25 | 16.7 | 0.7 | 3.9 | 2.0 |
| B1 sched `pickle.dumps` | 25 | 54.4 | 2.2 | 0.7 | 0.2 |
| B2 worker `pickle.loads` | 200 | 738.7 | 24.9 | 6.2 | 2.0 |
| C1 worker key+encode | 200 | 398.2 | 16.5 | 4.6 | 0.7 |
| C2 server msgspec decode | 200 | 898.5 | 28.2 | 4.4 | 0.4 |
| D server session hash | 200 | 106.7 | 2.7 | 0.2 | 0.2 |
| E server LOOKUP hash | 1 | 4.8 | 4.8 | 1.3 | 1.3 |

Saved output: `benchmarks/microbenchmark/token_ids_transport_results.txt`

## Things worth remembering

- **The two effects are not additive, and quoting either as a marginal
  number is misleading.** Reported alone against the same baseline: delta
  saves 2138 ms, packing saves 2197 ms, both save 2211 ms. Whichever lands
  first takes the bulk of each stage's cost with it. An earlier version of
  this analysis reported "17.6 ms from delta" — delta's *leftover* value
  given packing already applied — which understated it by ~120x. The 2x2 is
  the honest framing; the missing `list+delta` cell was added at the user's
  suggestion and is what exposed the error.
- **Only delta reduces bytes.** Packing leaves the wire volume essentially
  unchanged (170.79 → 171.74 MB) because msgpack already encodes a
  128k-vocab int in ~4 bytes. The 24x byte reduction is delta's alone, and
  it is what relieves vLLM's shm ring and the ZMQ path, which the CPU table
  does not show.
- **Packing is a pessimization below ~16k context.** Stage A at 1k: 3.1 µs
  (list) vs 11.1 µs (packed), 0.3x. `array.array` has fixed overhead that a
  short list copy beats. Crossover is around 16k-32k.
- **LOOKUP cannot be deltaed.** It is the call that seeds the session every
  delta chains onto, so it always carries the whole sequence. Packing still
  helps it (3.7x). Its hashes are also still thrown away — the session's
  memoization starts empty afterwards, so the first store re-hashes the
  prefix. Deliberately left alone as out of scope; remaining headroom.
- **Transport was never the bottleneck.** A real cross-process ZMQ round
  trip is 5464 µs against 4493 µs for the decode alone. Compression or a
  faster wire would have bought almost nothing; the cost was always
  materializing Python ints.

## Two bugs found in my own change, by reading rather than by a test

- `Session._compute_hash` held a live `memoryview` over the token
  `bytearray`. An exception whose traceback kept that frame alive would pin
  the buffer and make the next `absorb_tokens` raise
  `BufferError: cannot be re-sized`. Now scoped with `with memoryview(...)`.
- `resolve_prefetched_obj_keys` fed `key.start` straight into
  `Session.get_hashes`, which asserts chunk alignment — but
  `free_lookup_locks` documents that unaligned bounds are tolerated. Both
  ends now snap inward to whole chunks, which also fixes a latent
  off-by-one the old code had (it computed `start_chunk` with floor while
  selecting chunks with ceil).

## Testing

- New: `tests/v1/multiprocess/test_delta_token_ids.py` (43),
  `tests/v1/test_mp_connector_delta_token_ids.py`.
- `MPCacheServerContext.resolve_obj_keys`'s body was extracted to a module
  function `resolve_obj_keys_from_session(session, key, group_ids)` — it
  only ever touched the session, and the method needed a storage manager,
  GDS context and event bus to construct, so the change's central invariant
  was untestable. Now directly tested: a delta key and a full key name
  identical ObjectKeys; four successive delta stores chain to the same keys
  as one full store; a missing prefix yields none.
- **Methodology that mattered:** ran the same suite in a worktree at the
  base commit (`/raid/bo/delta_baseline`, extensions copied in since `csrc`
  is unchanged) and diffed the failure sets. Baseline has exactly one
  pre-existing, order-dependent failure
  (`test_http_api_registry::test_exclude_none_is_default`), so everything
  else was mine. Without that comparison I would have chased it.
- Result: `tests/v1/multiprocess` **808 passed, 1 skipped, 0 failed**
  (baseline: 764 passed, 1 failed).
- The 28 regressions fell into three shapes, all test-side: constructors
  passing `token_ids=`; mocks stubbing `compute_chunk_hashes` while
  production now calls `compute_packed_chunk_hashes` (this one surfaced as
  `TypeError: fold(): incompatible function arguments` from a native call
  and looked like an ABI problem); and `SimpleNamespace` fake keys whose
  missing attribute was swallowed by a bare `except`.

## Environment notes

- Built the native extensions in this worktree:
  `CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
  python setup.py build_ext --inplace`. System `nvcc` is 11.5 and torch is
  cu130, so `CUDA_HOME` must be set or the build refuses.
- Run anything against this tree with `PYTHONPATH=$PWD`: the venv's editable
  install points at a different worktree, and a script's `sys.path[0]` is
  its own directory, not the CWD.
- `benchmarks/microbenchmark/_native_stubs.py` now defers to real extensions
  when they exist, so it no longer shadows a built checkout.

## Not done

The **e2e A/B is written but not yet run**: `benchmarks/e2e/` plus
`run_token_ids_ab.sh`, three configurations (baseline checkout / packed /
packed+delta), long prompts sent as raw token id lists so the second pass is
an exact cache hit, recording TTFT alongside LMCache-server and vLLM CPU
time. Everything it needs is in place — baseline worktree built, GPUs free,
Qwen2.5-7B-Instruct readable at `/raid/jiayi-data/hf/hub/`. So there is
still **no TTFT number**; every figure above is CPU and bytes.
