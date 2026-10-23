# L1 residency server module

`lmcache/v1/multiprocess/l1_residency.py` answers one question over
`server_module_call`: how much committed L1 do these chunks hold, and how
much does their whole model hold? A client that knows which chunks another
cache holds (for example, a vLLM scheduler that knows which chunks its GPU
prefix cache holds) divides the two to get the share of L1 that the other
cache duplicates.

## Definitions

- **L1 object**: one exact `ObjectKey` (model, cache salt, kv rank, object
  group, chunk hash). Each kv rank's object counts on its own.
- **Committed**: resident in an L1 manager's object table and not
  `is_temporary`, so its write finished; staging and temporary objects never
  count. This is `L1CommittedUsage` in `lmcache/v1/distributed/internal_api.py`,
  returned by `L1Manager.get_committed_usage`.
- **model_\***: committed objects and bytes of the model, over every kv
  rank, object group and cache salt. **matched_\***: those among the queried keys.

## Request and reply

```
L1ResidencyQuery(model_name, world_size, chunk_hashes_by_salt,
                 chunk_hashes_by_group=[GroupHeldChunks(window,
                                                        chunk_hashes_by_salt)])
  -> object groups of (model_name, world_size):
       merged: [GroupWindow("attention", -1)]
       separated: one GroupWindow(kind, window_chunks) per entry of the
                  registered AttnWindowDesc, by object group id
  -> keys: one object group: chunk_hashes_by_salt in group 0
           several: each chunk_hashes_by_group entry in the group whose
                    GroupWindow equals its window (none: skipped)
     per salt, per chunk hash, one ObjectKey per kv rank of world_size
     (as a lookup of that model would build)
  -> StorageManager.get_l1_committed_usage(keys, model_name)
       = sum of L1Manager.get_committed_usage over every L1 manager
L1ResidencyReply(probe_hash, matched_objects, matched_bytes,
                 model_objects, model_bytes, object_groups,
                 matched_inuse_objects, matched_inuse_bytes)
```

`inuse_chunk_hashes_by_salt` / `inuse_chunk_hashes_by_group` (default
empty) name a subset of the queried chunks; the server expands them the same
way and answers their committed count as `matched_inuse_*`, in a second
`get_l1_committed_usage` call (so a second lock pass and model scan).
`firm_chunk_hashes_by_*` and `inuse_firm_chunk_hashes_by_*` name two
more subsets (dev diagnostic), answered as `matched_firm_*` and
`matched_inuse_firm_*`, each with its demoted part, in one more pass each.

The reply also splits `matched_*` by recency state (dev measurement):
`matched_demoted_*` counts keys an exclusive retrieve demoted
(`demote_keys`) and nothing touched or rewrote since; `matched_mru_*`
maps every other matched key to the `L1RecencyCause` that last made it
MRU: `write` (store commit), `lazy_touch` (a store found it resident),
`lookup_touch`, `end_session_touch`, or `other` (any other touch, or a
commit by an L2 prefetch). Demoted plus all causes equals `matched_*`.
The in-use subset is split the same way, in bytes only:
`matched_inuse_demoted_bytes` and `matched_inuse_mru_bytes` (by cause).
Each `L1ObjectState` carries the state, so it is dropped with the object;
`touch_keys`, `demote_keys` and admission update it together with the
listener call under the L1 manager's recency lock, so it agrees with the
eviction order.

`probe_hash` is the server's `TokenHasher` hash of a fixed probe chunk
(`compute_probe_hash`). A client that hashes the same probe differently
(another `hash_algorithm`, or another `PYTHONHASHSEED` for a non-blake3 one)
would match nothing, so it can detect that instead of reporting 0.

## Object groups

A client cannot see whether the server runs with `--separate-object-groups`,
so it names object groups by the key the server buckets KV cache groups by
(`KVLayerGroupsManager._detect_object_groups`): kind and window in chunks.
`AttnWindowDesc` reports both per object group id (`group_kinds`,
`num_chunks_in_sw`), and the reply's `object_groups` lists them, so the
client learns whether to send `chunk_hashes_by_group` and can check that
each of its keys exists. Under separation without `full_sw_kv`, regular
buckets are exactly `(recurrent, window chunks)`, so a key names at most one
object group; connector-private `aux` groups never match a client key.

Example: an attention + Mamba model with `chunk_size = 256` and a Mamba
block size of 256 registers `AttnWindowDesc([-1, 1], ("attention",
"recurrent"))`, so the reply lists `[GroupWindow("attention", -1),
GroupWindow("recurrent", 1)]` and `GroupHeldChunks(GroupWindow("recurrent",
1), ...)` counts in object group 1.

## When a chunk of one group counts as in L0

The module counts whatever chunks the client names; deciding which chunks
another cache holds is the client's job. For vLLM's GPU prefix cache, the
scheduler-side meter counts chunk `i` as in L0 in an object group iff vLLM
could serve it, its hit over all KV cache groups covering
`(i + 1) * chunk_size` tokens, and L0 caches the object's own data (the
chunk's Mamba state or sliding window, where the object group has one).
Deciding each object group on its own KV cache groups overcounted hybrid
models: attention blocks cached past the last Mamba state serve no hit, so
their L1 copies are the ones a later hit loads. See
[l1_duplication.md](../../integration/vllm/l1_duplication.md#when-a-chunk-of-one-group-counts-as-in-l0)
for the rule and an example.

## Cost and side effects

`get_committed_usage` takes the L1 manager lock once, looks the keys up and
scans the object table for the model count. It takes no read lock, updates
no recency, and emits no event or listener call, so measuring does not
change eviction.

The module reads L1 directly instead of following the observability event
bus: the bus tail-drops events once its 10,000-entry queue fills, so an
index built from it could drift.

## Limits

- `separate_object_groups` together with `full_sw_kv` (set by the blend
  engine) is not supported: every attention group then reports a full
  window, so a key no longer names one object group. The module raises
  `ValueError` at startup in that case.
- Under separation, a query for a model and world size that no worker has
  registered yet fails (`ValueError` from `find_attn_desc`); the client sees
  a failed call.
- With several L1 managers, each is read under its own lock in turn, so the
  sum is consistent per manager, not across managers; a key held by two of
  them counts twice, in both numerator and denominator.
- Each query scans each L1 manager's whole object table under its lock,
  like `report_status`: the cost grows with every resident object, of every
  model on the server, not only the queried model's committed ones.

## Usage

The module is opt-in and lives outside the built-in `modules` package, so a
server that does not load it pays nothing:

```bash
--server-module '{"module_path": "lmcache.v1.multiprocess.l1_residency"}'
```
