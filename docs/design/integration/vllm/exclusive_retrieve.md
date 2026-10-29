# Exclusive Retrieve in the LMCache MP Connector

Exclusive retrieve keeps the GPU prefix cache (L0) and LMCache L1 mostly
disjoint. Whenever L0 holds a chunk for a request, whether loaded from L1 or
hit locally in vLLM's prefix cache, the L1 copy becomes the **next eviction
victim** instead of the most recently used entry. Lazy offload then stores
the chunk again before vLLM evicts its GPU copy, so the chunk returns to L1
when (and only when) the GPU is about to lose it.

Why: without it, every hit lives twice, once in GPU blocks and once at the
MRU end of L1, and L1 capacity is spent on KV the GPU already holds. With it,
L1 capacity goes to KV the GPU does not hold.

Enable it in `kv_connector_extra_config` together with lazy offload:

```json
{"lmcache.mp.lazy_offload": true, "lmcache.mp.exclusive_retrieve": true}
```

It applies to the LMCache-driven transfer path (the default). The
engine-driven path does not demote, so the flag only causes a redundant,
no-op re-store there.

## Rules

| Event | L1 copy |
|-------|---------|
| Chunk loaded from L1 into L0 by a retrieve | demoted (rule 1) |
| vLLM local prefix hit, including another request's hit on a prefix L0 already holds | demoted after allocation (rule 8); the lookup does not touch it |
| Lazy re-store at L0 eviction, L1 copy still resident | touched to most recently used (rule 6) |
| Lazy re-store at L0 eviction, L1 copy gone | written again |
| Lazy store of a chunk L0 still holds (another block of the op made it due) | committed or touched demoted, and buffered again for its own eviction (rule 9) |
| `end_session` | no L1 touch (rule 4) |

1. **Demote after a fully successful retrieve.** The server's retrieve
   submits a `demote_l1_keys` host callback on the transfer stream right after
   `finish_read_by_owner`, with the same `(l1_owner, keys)` groups. On success
   those groups are exactly the loaded keys: full-attention chunks plus the
   sliding-window / recurrent chunks inside each group's window. A failed or
   partial retrieve demotes nothing.
2. **Demote, never delete.** Demotion only reorders the eviction policy
   (`EvictionPolicy.on_keys_demoted`): keys stay readable and usage is
   unchanged. A later lookup still hits them until L1 runs out of room.
3. **Demote in request order.** LRU moves each key to the front in order, so
   the request's last chunk ends at the very front and suffixes are evicted
   before the prefixes they extend. IsolatedLRU does the same per cache salt;
   ARC moves the key to the LRU end of T1; noop ignores it. Each L1 manager
   (multi-L1) demotes only the keys it owns.
4. **`end_session` does not undo it.** For an exclusive request
   `end_session` touches no L1 key, only L2: every L1 copy that is not
   demoted was made MRU by its own store or lookup, so a session-end touch
   would only promote chunks L0 still holds. A plain request's `end_session`
   still touches its keys in L1 and L2, skipping the ones the session
   recorded as demoted.
5. **Re-store from token 0.** When vLLM allocates the request's blocks (the
   `PREFETCHING` transition of `update_state_after_alloc`) the scheduler sets
   `num_stored_tokens` to 0, so the loaded chunks and the local-hit prefix
   below them become ordinary lazy store candidates again (with their block
   hashes snapshotted on admission; Mamba null blocks are filtered as usual).
6. **A re-store of a resident key promotes it.** If the demoted copy is still
   in L1 when the store arrives, `reserve_write` skips it (already resident),
   and the store touches it in L1 and L2. The L1 copy is the primary again.
7. **The connector alone decides.** The flag reaches the server per request as
   `request_configs["lmcache.mp.exclusive_retrieve"] = True` on the LOOKUP,
   FREE_LOOKUP_LOCKS, RETRIEVE and STORE keys. The connector strips a
   user-supplied value (and a user-supplied held-prefix hint), so a request
   cannot turn it on without lazy offload behind it.
8. **Demote a local hit after allocation, per object group.** The lookup of an
   exclusive request skips its touch of the APC-covered prefix. Once vLLM has
   allocated the request's blocks, the FREE_LOOKUP_LOCKS the connector sends
   carries `request_configs["lmcache.hint.engine_held_tokens"] = D`, vLLM's
   aligned hit (`num_vllm_hit_tokens`). The server's `free_lookup_locks`
   demotes, through `StorageManager.demote_cached_keys`, the keys of
   `[0, D)` that L0 holds in every layer of their object group, and records
   them so `end_session` skips them. `AttnWindowDesc.engine_held_chunks`
   gives the chunks per object group:

   | Object group | Chunks of `[0, D)` demoted |
   |--------------|----------------------------|
   | full attention | every chunk |
   | sliding window of `w` chunks | the last `w` |
   | recurrent (Mamba / linear) | only the chunk ending exactly at `D`; none when `D` is mid-chunk |
   | mixed (no `--separate-object-groups`) | the narrowest rule of its layers: with Mamba layers, only the chunk ending at `D` |

   The other keys (earlier Mamba states, which vLLM no longer holds) are the
   only copy and keep their L1 position. The demotion and the lock release of
   one FREE_LOOKUP_LOCKS share one rolling hash of the prompt.
9. **A store of a chunk L0 still holds is demoted.** Lazy offload makes an
   op due when any of its blocks nears vLLM's eviction head, and an op spans
   every cache group, so a due op can carry chunks whose blocks vLLM still
   references (an align-mode Mamba group frees older state blocks while the
   attention blocks stay held). At emission the connector counts chunk `k`
   as engine-held iff vLLM could serve it and keeps its attention: vLLM's
   own `KVCacheCoordinator.find_longest_cache_hit` over the request's
   `Request.block_hashes` (all KV cache groups, read-only), restricted to
   blocks vLLM references, covers the chunk's end, and every non-null block
   of the chunk's full-attention groups (of every group in a model without
   attention) has `ref_cnt > LMCache's own pins`. The chunk's own Mamba or
   sliding-window blocks need not be referenced: the hit already requires
   the state or window that anchors it past the chunk. The restriction is
   applied by cutting: when a block the hit returns (an attention block
   below the hit end, the Mamba state or window that makes the end a hit
   point) has `ref_cnt <= LMCache's pins`, the lookup is repeated with the
   hit capped at that block's start, until every returned block is
   referenced. It sends the held chunks' referenced block ids as
   `request_configs["lmcache.hint.engine_held_blocks"]` on the STORE. The
   server's store demotes, chunk by chunk and per object group through
   `StorageManager.demote_cached_keys`, the keys whose every non-null block,
   over the kernel groups of that object group, is listed, so a chunk's
   freed Mamba object is written MRU while its attention object is demoted;
   a mixed object group needs its Mamba block listed too. A resident key is
   demoted instead of touched; a new key is admitted demoted when its write
   finishes (`L1Manager.demote_keys` marks a staged key, and `finish_write`
   demotes it after marking the write MRU). Any other chunk is stored as
   before.

   Each cache group's part of a chunk is not held on its own blocks: on a
   hybrid model vLLM's local hit is the minimum over groups and ends at the
   last cached Mamba state, so attention blocks past it serve nothing, and
   the next hit loads that attention from L1. Demoting those copies would
   put the copy the next hit needs first in L1's eviction order, turning
   the hit into a recompute. A free-queue block still serves a hit until
   vLLM recycles it, so a chunk resting on one is not counted either when
   it is the chunk's attention or the Mamba state that anchors the hit past
   it: its demoted copy could be evicted before that hit.

   The connector keeps the held part of each emitted op outside the policy
   ("parked"), with every chunk that was not held, and every block of a
   held chunk vLLM no longer references, replaced by the null block, and
   hands it back to the policy once vLLM drops a reference to one of its
   blocks (`ref_cnt <= LMCache's pins`). From then on it is an ordinary
   candidate: stored at its own eviction, touched or written to MRU. A
   parked op does not ride along with later ops of its request, so each
   release re-stores it at most once; it keeps the request's session open.

```text
op due (Mamba state block at the free-queue head)
   │  chunks within vLLM's hit, every block still referenced by vLLM
   ├─> STORE [s, e), hint engine_held_blocks = those chunks' blocks
   │      server: held keys -> demote_cached_keys (new: demoted at commit)
   └─> park the held part (non-held chunks nulled)
          vLLM frees one of its blocks -> back to the policy -> store at eviction
```

```text
LOOKUP (start = APC-covered)    no touch of [0, start) for an exclusive key
   │
vLLM allocates blocks -> update_state_after_alloc
   │   tracker.num_stored_tokens = 0
   └─> FREE_LOOKUP_LOCKS [0, free_end), hint engine_held_tokens = D
          -> demote_cached_keys(held keys of [0, D)), record on Session
          -> release the locks of [lookup start, free_end) as before
```

**Why after allocation.** Until vLLM allocates the request's blocks, the
local hit is only a prediction: a waiting request's APC blocks can be evicted
while it waits, and lazy offload tracks blocks only once they belong to a
request. Demoting at lookup would leave a demoted L1 copy with no store
candidate behind it. After allocation the blocks hold a reference and are
lazy candidates (rule 5), the same state a loaded chunk is in after its
retrieve.

**Local-hit-only requests.** A request LMCache serves nothing past vLLM's hit
still sends FREE_LOOKUP_LOCKS: the lookup counts the covered prefix as hit, so
the hit is at least `lookup_start > 0` and `free_end > 0`. That FREE releases
nothing (the lookup locked nothing below its start) but carries `D`. A lookup
that returns 0 (no covered prefix, a declined load, or a full re-lookup after
an APC shrink that found nothing) sends no FREE, so nothing is demoted; its
held chunks are touched by `end_session` as before.

```text
lookup hit ──> RETRIEVE [s, e) ──> finish_read_by_owner ──> demote_l1_keys
                                                            (keys -> LRU front)
     GPU blocks hold [0, e); tracker.num_stored_tokens = 0
                                      │
      lazy offload: store [0, e) when its blocks near vLLM's eviction head
                                      │
              ┌───────────────────────┴────────────────────────┐
    L1 copy still resident                          L1 copy already evicted
    reserve_write skips it -> touch (MRU)           write it again (MRU)
```

## Why nothing is dropped

The L1 copy of a loaded chunk is not removed by the retrieve; it only moves to
the front of the eviction order. Until something actually evicts it, both
copies exist. The GPU copy is a lazy store candidate from the moment the load
is admitted, and lazy offload changes only **when** a candidate is stored, not
**whether**: the store is emitted before vLLM reuses the blocks. So in the
normal path, each chunk has at least one copy at every point: in L1 (demoted),
in GPU blocks (candidate), or in both.

The same holds for a local-hit chunk: it is demoted only after vLLM allocated
the request's blocks and only in the object groups whose blocks L0 holds, and
the request's lazy candidates start at token 0, so each demoted key's GPU copy
is a candidate until it is stored (a touch when resident, else a write).

This is why the mode requires lazy offload, enforced with a `ValueError` at
connector init. The eager path stores only new tokens, so a demoted chunk
would have no copy re-entering L1.

## Known limits

- **Another request's touch promotes a key L0 still holds.** A touch by a
  different request moves the key back to most recently used while L0 still
  holds it: for example the lazy re-store of the same chunk from another
  request's GPU blocks that vLLM is evicting, or the `end_session` of a plain
  request. This costs L1 capacity, never data.
- **A release is not an eviction.** A parked chunk is handed back to the
  policy when any holder drops a reference, which can be one of several
  requests sharing the prefix; the policy then stores it at its eviction,
  which may still be later than the release.
- **Shared prefixes are re-stored once per request.** Every exclusive
  request holding a local-hit prefix buffers its own lazy store for it; the
  first one touches or writes the key, the later ones touch it again.

## Known loss windows

These cases lose the chunk. In each one, the cost is a recompute on a later
request, never wrong KV:

- **Load fails or is aborted after demotion.** Demotion runs only on a fully
  successful server-side retrieve, but the request can still be aborted, or
  its load reported failed, before vLLM caches the loaded blocks. No store
  candidate exists then, so the demoted L1 copy is the next victim and
  nothing re-stores it.
- **Preemption.** A preempted request's lazy candidates and parked ops stay:
  its freed blocks keep their hashes, so its demoted chunks are stored
  again when they come due, unless the restarted request offers the same
  tokens first, which supersedes them.
- **Stale block hash.** A candidate whose block was reused before the drain is
  skipped by the lazy hash check, as for any lazy store; so is a parked op
  whose block was reused before its release was seen. When the next step
  may allocate every free block, the drain pins parked blocks, so a release
  in that step cannot recycle them before the next drain.
- **Anchor released before the chunk.** A parked op is readmitted when one
  of its own blocks is released. The Mamba state that anchored its hit was
  referenced at emission, and a request holds both until it ends, so they
  are normally released together. If the anchor's last holder is a request
  that does not hold the chunk's own blocks, vLLM can free and recycle the
  anchor while the chunk stays parked, and its L1 copy stays demoted.
- **Mixed hybrid chunk before the tail.** Without `--separate-object-groups`,
  one object holds attention and align-mode Mamba slices, and vLLM leaves a
  loaded chunk's Mamba blocks null except at the tail. The re-store skips a
  chunk that is null in any kernel group, rather than commit a null recurrent
  state under a valid key, so its demoted L1 copy may be evicted. Separate
  object groups avoid this: only the all-null Mamba object is skipped.
  A chunk parked without its freed Mamba state costs a mixed object group a
  park, its pins and a re-store that is skipped: the chunk was not demoted,
  since its state was not listed, so its copy was already stored MRU.

## Flag plumbing

```text
kv_connector_extra_config["lmcache.mp.exclusive_retrieve"]
  -> LMCacheMPConnector.exclusive_retrieve      (ValueError without lazy offload)
  -> kept only if every server's GET_EXPERIMENTAL lists "exclusive_retrieve"
  -> tracker.request_configs[...] = True        (user-supplied value stripped)
  -> LMCacheMPRequestMetadata.request_configs   (RETRIEVE and STORE)
  -> worker adapter key -> IPCCacheServerKey.request_configs
  -> IPCCacheServerKey.requests_exclusive_retrieve() on the server
```

A server advertises `exclusive_retrieve` when it builds the LMCache-driven
transfer module. Against one that does not (older, or `engine_driven`), the
scheduler connector logs one WARNING and runs lazy offload alone: an older server
ignores the request config, so XR would only re-store hits, and refusing to start
would turn a connector-first rolling upgrade into an outage. The scheduler
adapter asks every server for its experimental features once, at init,
whether or not the flag is set.

The held prefix travels only on the post-allocation FREE_LOOKUP_LOCKS:

```text
LMCacheMPRequestTracker.num_vllm_hit_tokens
  -> request_configs["lmcache.hint.engine_held_tokens"]   (FREE_LOOKUP_LOCKS)
  -> IPCCacheServerKey.engine_held_tokens() on the server
```

The held blocks of a lazy store travel only on that STORE:

```text
LazyOffloadManager drain (xr ops; ref_cnt > own pins)
  -> request_configs["lmcache.hint.engine_held_blocks"]   (STORE)
  -> IPCCacheServerKey.engine_held_block_ids() on the server
```

`END_SESSION` carries only the request id, which is why the demoted keys are
recorded on the server `Session` rather than looked up from the key.
