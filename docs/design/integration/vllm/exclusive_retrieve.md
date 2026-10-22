# Exclusive Retrieve in the LMCache MP Connector

Exclusive retrieve keeps the GPU prefix cache (L0) and LMCache L1 mostly
disjoint. When a chunk is loaded from L1 into GPU blocks, the L1 copy becomes
the **next eviction victim** instead of the most recently used entry. Lazy
offload then stores the loaded chunk again before vLLM evicts its GPU copy, so
the chunk returns to L1 when (and only when) the GPU is about to lose it.

Why: without it, every hit lives twice — once in GPU blocks, once at the MRU
end of L1 — and L1 capacity is spent on KV the GPU already holds. With it, L1
capacity goes to KV the GPU does not hold.

Enable it in `kv_connector_extra_config` together with lazy offload:

```json
{"lmcache.mp.lazy_offload": true, "lmcache.mp.exclusive_retrieve": true}
```

It applies to the LMCache-driven transfer path (the default). The
engine-driven path does not demote, so the flag only causes a redundant,
no-op re-store there.

## Rules

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
4. **`end_session` does not undo it.** `end_session` touches every key of the
   request; the session records the demoted keys and skips them.
5. **Re-store from the retrieve start.** At the `WAITING_FOR_LOAD` transition
   the scheduler sets `num_stored_tokens` to the retrieve's start token, so the
   loaded chunks become ordinary lazy store candidates again (with their block
   hashes snapshotted on admission; Mamba null blocks are filtered as usual).
6. **A re-store of a resident key promotes it.** If the demoted copy is still
   in L1 when the store arrives, `reserve_write` skips it (already resident),
   and the store touches it in L1 and L2. The L1 copy is the primary again.
7. **The connector alone decides.** The flag reaches the server per request as
   `request_configs["lmcache.mp.exclusive_retrieve"] = True` on the RETRIEVE
   and STORE keys. The connector strips a user-supplied value, so a request
   cannot turn it on without lazy offload behind it.

```text
lookup hit ──> RETRIEVE [s, e) ──> finish_read_by_owner ──> demote_l1_keys
                                                            (keys -> LRU front)
     GPU blocks hold [s, e); tracker.num_stored_tokens = s
                                      │
      lazy offload: store [s, e) when its blocks near vLLM's eviction head
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

This is why the mode requires lazy offload, enforced with a `ValueError` at
connector init. The eager path stores only new tokens, so a demoted chunk
would have no copy re-entering L1.

## Known loss windows

These cases lose the chunk. In each one, the cost is a recompute on a later
request, never wrong KV:

- **Load fails or is aborted after demotion.** Demotion runs only on a fully
  successful server-side retrieve, but the request can still be aborted, or
  its load reported failed, before vLLM caches the loaded blocks. No store
  candidate exists then, so the demoted L1 copy is the next victim and
  nothing re-stores it.
- **Preemption.** A preempted request's lazy candidates are dropped with its
  blocks, so its demoted chunks are not re-stored.
- **Stale block hash.** A candidate whose block was reused before the drain is
  skipped by the lazy hash check, as for any lazy store.

## Flag plumbing

```text
kv_connector_extra_config["lmcache.mp.exclusive_retrieve"]
  -> LMCacheMPConnector.exclusive_retrieve      (ValueError without lazy offload)
  -> tracker.request_configs[...] = True        (user-supplied value stripped)
  -> LMCacheMPRequestMetadata.request_configs   (RETRIEVE and STORE)
  -> worker adapter key -> IPCCacheServerKey.request_configs
  -> IPCCacheServerKey.requests_exclusive_retrieve() on the server
```

`END_SESSION` carries only the request id, which is why the demoted keys are
recorded on the server `Session` rather than looked up from the key.
