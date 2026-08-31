# Why the FIFO arm stored nothing: the trigger fires, the blocks are gone

Bo did not believe the "the trigger is never reached" explanation recorded
earlier ("fifo 没触发其实我还是觉得挺奇怪的"). He was right; the outcome is
the same (0 stores) but the cause is not.

## 1. The trigger is reached

`FIFOOffloadPolicy.pop_items_for_offload` (fifo.py:80) drains only when the
number of requests that are finished AND still hold buffered stores reaches
`lmcache.mp.lazy_offload_threshold` (100). That set does not shrink on its
own: `LazyOffloadRequestRegistry` keeps a FINISHED slot alive while the
request has pending items (`on_request_finished` returns early at
lazy_offload_manager.py:472), and nothing else prunes it. So the level climbs
monotonically with completions and crosses 100.

f40 engine log: aiperf starts 02:09:09, first drain 02:20:45, i.e. ~11 min
and ~100 completions in. Then a sawtooth: each drain pops
`lazy_offload_select_count` = 10, the level falls to ~90, ten more
completions bring it back. 33 bursts of exactly 10 in the log, last at
02:49:18.

## 2. Every drained store is stale

Under lazy offload `LMCacheMPConnector.request_finished` returns False
(lmcache_mp_connector.py:1063-1068): the finished request's GPU blocks go
straight back to the free queue, unpinned. A buffered store keeps only block
ids plus the block hashes captured at admission. The manager revalidates at
drain time (`_drain`, lazy_offload_manager.py:561-580): `pool.touch`, then
compare current hashes against the snapshot; on mismatch or None it logs
"Block hashes missing or mismatched ... dropping its remaining chunks",
frees the blocks and drops the request.

Ten minutes of free-queue exposure is fatal. f40: 330 warnings, 330 distinct
request ids, i.e. every request the drain ever released. Nothing was
submitted: `lmcache_mp_l1_memory_usage_bytes` 0.0 at the end,
`lookup_hit_tokens_total` 0 against 4.51e7 requested. f48 310 warnings, f72
340. The eviction-aware arms: l40 has zero such warnings, because the policy
drains at the free queue's eviction head, before reuse.

This is not a bug in the validation; storing a recycled block would write
another request's KV under this request's key. The bug is that a blind
count trigger guarantees the race is lost.

## 3. Why the threshold is not a one-line fix

- High threshold: a backlog large enough to fire is a backlog old enough to
  be stale. Nothing is stored.
- `threshold=1`: drains on the step after each finish, wins the race (this is
  what the prior line's harness used, driver.py:1171), but it is storing at
  compute time with one step of lag. The deferral's two benefits, ~1/3 write
  volume and skipping content that is never reused, are both gone.

Same conclusion as before, better argument: it is the eviction awareness,
not the trigger constant, that makes deferral work. The colleague's own
`lazy_offload.md` already documents the hazard ("Blocks may be reallocated
to other requests during the buffer phase"); what is new here is that at the
default threshold it hits 100% of the time.

## 4. Text corrected

- `pr_info.md`: motivation paragraph rewritten (trigger reached at ~11 min,
  330/330 dropped, 0 bytes stored, threshold=1 tradeoff), results footnote
  ("no store survived drain-time revalidation"), reviewer note points at the
  documented buffer-phase hazard.
- record 1 section 3 and `artifacts/ab_analysis_snapshot.md` FIFO section
  carry the correction inline.
- No PR-branch change: the design docs never claimed the trigger was
  unreachable, so `lazy_offloading_policy_pr` @ d45edab6 is untouched.

Evidence: `f40_server.log`, `f48_server.log`, `f72_server.log`,
`l40_server.log` (scratchpad of session aa12d55f, sweep/), and
`artifacts/sweep/f40/f40_mp_final.prom`.
