# L0/L1 duplication meter

`lmcache/integration/vllm/l1_duplication.py` measures how much of L1 (the MP
server's CPU cache) L0 (vLLM's GPU prefix cache) also holds. A duplicated
object takes L1 capacity without adding a hit: a request that reaches it
through its prefix hits it on the GPU first. At ratio 0, L0 and L1 hold
disjoint data and their capacities add up. Lazy offload
([lazy_offload.md](lazy_offload.md)) lowers duplication by storing a block
only when vLLM is about to reuse it; exclusive retrieve would lower it
further by dropping L1's copy of a chunk once it is loaded into the GPU. The
meter gives both a number to compare.

## Definition

At one sample, for the scheduler's model:

```
byte_ratio   = matched_bytes   / model_bytes
object_ratio = matched_objects / model_objects
```

- **L1 object**, **committed**, **model_\***: as
  [l1_residency.md](../../v1/multiprocess/l1_residency.md#definitions)
  defines them. **matched_\***: the committed objects of the chunks L0
  holds.
- **Held in L0**: chunk `i` of a prefix is held in an object group iff
  vLLM would hit at least `(i + 1) * chunk_size` tokens of that prefix from
  the GPU now, and L0 still caches the chunk's own data in that object
  group: for each Mamba KV cache group the state block at the chunk's end,
  for each sliding-window group the window ending there (full attention
  is cached by the hit itself). The meter asks vLLM's own
  `KVCacheManager.coordinator.find_longest_cache_hit`, so every attention
  type (full, sliding window, Mamba, hybrid) uses vLLM's rule for the hit.
  Past the hit, LMCache loads from L1, so L1's copy is used, not wasted.
  With `--separate-object-groups` each object group is measured on its own
  (see below).

**In use vs free-cached.** `matched_inuse_*` is the part of `matched_*`
whose chunk a running request holds: some block of the chunk (per vLLM KV
cache group: the blocks overlapping the chunk for full attention, the window
ending at the chunk's end for sliding window, the state block at the chunk's
end for Mamba), looked up with
`BlockPool.get_cached_block`, has `ref_cnt > 0`. Without separated object
groups every vLLM KV cache group's blocks are read; with them, only the
groups stored in that object group. `matched_* - matched_inuse_*` is held
only by blocks in vLLM's free queue (`ref_cnt == 0`, still hashed), which
vLLM may evict at any allocation. `matched_inuse_demoted_bytes` and
`matched_inuse_mru_<cause>_bytes` split `matched_inuse_bytes` like
`matched_demoted_*` and `matched_mru_<cause>_*` split `matched_*`: an
in-use chunk whose L1 copy is not demoted was made MRU by that cause.

The denominator is L1 because the question is how much L1 capacity is
wasted; a ratio over L0 would grow when L1 grows even if L1 wasted nothing.

Example: a hybrid model, `chunk_size = 4`, one 12-token prefix `a0 a1 a2`
(4 KiB per object). Committed L1 of model `m` holds `a0`, `a1`, `a2` and
`b0`. Attention holds all three chunks on the GPU, but Mamba keeps a state
only at the end of `a0` (vLLM freed the later ones). vLLM hits 4 tokens, so
only `a0` is held: `matched = 1 object / 4 KiB`, `model = 4 objects /
16 KiB`, both ratios 0.25. Had Mamba kept its state at the end of `a2` only,
vLLM would hit all 12 tokens, but only `a2` would count: the objects of
`a0` and `a1` carry Mamba states L0 no longer has, so L1's copies of them
are the only full ones.

### When a chunk of one group counts as in L0

With `--separate-object-groups`, LMCache stores chunk `i` as one object per
object group, and each object group holds the layers of one or more vLLM KV
cache groups. Chunk `i` counts as in L0 in an object group iff vLLM's whole
hit covers it and that object group's KV cache groups cache its own data,
as above: an attention object group needs only the hit, a recurrent one
also the chunk's own state.

An earlier rule decided each object group on its own KV cache groups (a
full-attention group's own prefix hit, a sliding window cached at the
chunk's end, a Mamba state cached at the chunk's end). That overcounts
hybrid models: vLLM's hit is the minimum over all groups and ends at the
last cached Mamba state, so attention blocks past it serve nothing and the
next hit loads that chunk's attention from L1, which is then not a
duplicate. Counting every object of a chunk inside the hit overcounts the
other way: a Mamba object whose own state vLLM freed is not in L0, though a
later state serves the hit.

Example: the hybrid prefix above (`chunk_size = 4`, 4 KiB objects), with L1
holding `a0..a2` in both object groups (attention id 0, Mamba id 1).
Attention holds all three chunks; Mamba keeps states at the ends of `a0` and
`a2`. vLLM hits 12 tokens, so `matched = 5 objects / 20 KiB` (`a0..a2`
attention, `a0` and `a2` Mamba) of `model = 6 objects / 24 KiB`. With
Mamba's only state at the end of `a0`, vLLM hits 4 tokens and
`matched = 2 objects` (`a0` in both groups), where the per-group rule
counted 4 (`a0..a2` attention, `a0` Mamba).

**Object groups.** The scheduler names an object group by the key the
server buckets KV cache groups by: kind (`attention` or `recurrent`) and
window in chunks, from `layer_window_tokens` and `window_tokens_to_chunks`
(full attention `-1`, sliding window `ceil(sliding_window / chunk_size)`,
Mamba `ceil(block_size / chunk_size)`). vLLM groups with the same key share
one object group. The server maps the key to its object group id through
the registered `AttnWindowDesc` (see
[l1_residency.md](../../v1/multiprocess/l1_residency.md#object-groups)).

## How it works

```
scheduler                                      MP server
  request tracked   -> follow its live Request.block_hashes
  request finished  -> keep only its chunks L0 holds now
  every 10 s, when no query is pending:
    per prefix: chunks < find_longest_cache_hit // chunk_size,
      per measured object group those whose own state/window is cached
      (chunk hashes via TokenHasher)
      finished prefix: chunks cut to the held ones, dropped at 0 in all
    if servers are healthy, send model,
    kv world size, chunk hashes by salt
    (per object group key if separated) ------>  l1_residency.query:
                                                 committed L1 usage of those
                                                 chunks and of the model,
                                                 probe hash, object groups
  a later step reads the reply   <-----------
    object groups differ from what was measured -> switch rules, drop sample
    sum over servers -> DuplicationSample
    -> LMCacheMPConnectorStats
```

The scheduler cannot see the server's `--separate-object-groups`, so the
meter starts with the whole hit and the reply tells it the server's object
groups. The first reply that lists several switches the meter to
per-object-group queries and drops that sample (the server counted nothing in it); a
key of the scheduler's that the server lacks, servers that list different
object groups, or vLLM prefix caching disabled stop the meter with an error.

- `find_longest_cache_hit` is read-only: it looks hashes up in the block
  pool's dict and builds fresh lists; it touches no ref count, LRU, prefix
  cache stats or KV event (those live in `get_computed_blocks`).
- A finished prefix keeps its block hashes and the chunk hashes L0 held,
  not its tokens. Its block hashes stay whole: vLLM can hit less from a cut
  list than from the prompt (EAGLE drops the last hit block; Mamba hits only
  up to a cached state), so a cut list would shrink the prefix each sample.
  Its count is capped at the chunks it kept; chunks past them belong to a
  longer prefix that counts them. A chunk hash chains every chunk before it,
  so finished prefixes are keyed by their last chunk, which bounds them by
  what L0 holds; one whose last chunk a longer prefix already holds is not
  looked up again (turn `t` of a conversation while turn `t + 1` runs).
  A finished prefix keeps its held chunks per measured object group, and
  skips the lookup only when every object group already holds all of them
  (for an object group of full attention only, its last). Of
  two that end at one chunk the later-finished one stays: vLLM evicts freed
  blocks oldest first, so its blocks past that chunk outlive the other's.
- vLLM's block hash and LMCache's chunk hash differ, so the scheduler hashes
  chunks with `TokenHasher` and `lmcache.mp.hash_algorithm`. The reply
  carries the server's hash of a fixed probe chunk; a mismatch (another
  algorithm, or another `PYTHONHASHSEED` for a non-blake3 one) logs an error
  and stops the meter, since no chunk could match.
- The step never waits on a server. A failed answer drops that sample; a
  query unanswered when the next is due is dropped, and after 3 in a row the
  meter stops (over ZMQ, a server without `--server-module` never answers).
  A server that does not serve the method (`unknown server module method`,
  or gRPC `UNIMPLEMENTED`), or whose reply this version cannot decode, stops
  the meter with an error. A handler failure (`success=False` with any
  other error) drops only that sample, so a server that fails every query,
  for example one too old to decode this version's query, logs a warning
  every 10 s and never yields a sample.

## Cost

One sample costs the scheduler thread one `find_longest_cache_hit` per
running prefix and per finished prefix not covered by a longer one: about
70 ns per vLLM hash block with full attention. Measured with a Python
full-attention coordinator (`chunk_size=256`, 16 hash blocks per chunk):

| Tracked prefixes (60 chunks each)                    | One sample |
|------------------------------------------------------|------------|
| 100 running + 2,000 finished, all distinct           | ~200 ms    |
| 100 running + 2,000 finished, each a running prefix  | ~20 ms     |

The distinct case holds 120,000 chunks (31 M tokens) in L0, far beyond a
GPU's capacity; held finished prefixes are bounded by what L0 holds. A step
with no sample due costs ~0.1 µs. Finishing a request costs one lookup and
hashes the chunks L0 holds (~0.7 ms for 60 chunks).

A prefix lookup costs one `find_longest_cache_hit`, plus, per chunk inside
the hit and object group, one `get_cached_block` per Mamba state or
sliding-window block that holds the chunk's own data.

The in-use split adds, per distinct held chunk and object group, one
`get_cached_block` per block it reads, stopping at the first in-use block.

## Output

Each sample replaces the previous one in `LMCacheMPConnectorStats` (latest
value, not a mean); vLLM's `KV Transfer metrics` log line shows it:

```
'LMCache MP L1 dup byte_ratio': 0.412, 'LMCache MP L1 dup object_ratio': 0.412, 'LMCache MP L1 dup matched_objects': 1195, ..., 'LMCache MP L1 dup matched_inuse_objects': 402, 'LMCache MP L1 dup matched_inuse_bytes': ...
```

## Several schedulers on one server

When several schedulers share a server under one model name (data
parallelism, or several vLLM instances), each one's denominator is the
model's whole committed L1, written by all of them, while its numerator
covers only its own L0. Each engine's ratio is therefore its own view: the
share of the shared L1 that this engine's GPU also holds. Do not sum
`matched_*` across engines; a chunk two engines both hold counts in both.

## Limits

- The server module's
  [limits](../../v1/multiprocess/l1_residency.md#limits) apply; its scan of
  the L1 object table runs once per sample per measuring scheduler.
- With EAGLE or Mamba, a finished prefix dropped for sharing its last chunk
  with a later one is undercounted if its blocks past that chunk outlive
  the later one's, which vLLM's eviction order makes rare.
- With separated object groups, the first sample (10 s after start) is
  dropped.
- A server with `--separate-object-groups` and full sliding-window KV
  (`full_sw_kv`, set by the blend engine) refuses to load the module:
  every attention group then reports a full window, so the key no longer
  names one object group.
  Measuring separated object groups also needs vLLM prefix caching.

## Usage

Both sides are opt-in. Start the MP server with the module
([l1_residency.md](../../v1/multiprocess/l1_residency.md#usage)) and enable
the meter in vLLM's `kv_transfer_config.kv_connector_extra_config` (a JSON
boolean; a string such as `"false"` is rejected):

```json
{"lmcache.mp.measure_l1_duplication": true}
```
