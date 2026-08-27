# Escalation 2 — Should the retrieve path renew its L1 read lock at transfer time?

**Severity:** performance / availability, **not** correctness (see the
mitigation note below). **Component:** `lmcache.v1.distributed` L1 read
locks. **Status:** deliberately not implemented — the fix has a real
trade-off that is not a certification task's call to make.

---

## The mechanism

A read lock is taken at **lookup** time and consumed at **transfer** time:

```
submit_prefetch_task    -> L1Manager.reserve_read     (lock taken)
read_prefetched_results -> unsafe_read                (lock consumed)
```

`TTLLock.is_locked()` is `counter > 0 AND now < expiration`, and only
`lock()` refreshes `expiration`. Nothing between those two points refreshes
it. But vLLM looks a request's prefix up when it *enters the waiting queue*
and transfers only once blocks free, so the gap is a queue wait — unbounded
in a batch workload. Past the 300 s default, every lock reserved before the
crossing has expired and the load quietly returns nothing.

## The evidence it is real

Gemma 4-E4B corrupted **1288 of 2374** MME answers while reporting a hit
coverage of 1.0076. Diagnosis details that made the timeout hard to see:

- The first failed read lands **332 s** into pass 2, not at question 1, and
  failures then run ragged for 6.6 minutes — each lock expires on its own
  reserve time, so the onset never looked like a threshold.
- All 7699 failures reported `object_group_id=0`, which looked like a
  property of that group and was not: the retrieve loop breaks on the first
  failing group (`lmcache_driven_transfer.py:1366`) and all six groups,
  reserved in the same instant, expire together.
- Capacity was never involved (37 GB used of 280 GB).
- Gemma 3-4B, whose entire two-pass run is 643 s, never held a lock long
  enough and logged zero failures. Every previously-green certificate was
  a short run.

Confirmed by single-variable rerun: raising only the TTL took the same
configuration from 1288 flips to **0**, with zero failed reads where it had
logged 7699.

## The question

**Should the retrieve path renew the lock at transfer time rather than
inherit the lookup-time expiration?**

The trade-off is why this is a maintainer decision:

- The TTL exists for a reason — it is the safety valve against a client
  that reserves and then dies, pinning L1 memory forever.
- Renewal weakens exactly that valve: a wedged-but-still-polling consumer
  could renew indefinitely.
- Lengthening or removing the TTL globally trades one failure mode for a
  memory leak.

## What was done instead (both deliberately non-invasive)

1. **Test-suite only:** `MP_SERVER_L1_READ_TTL_S = 86400`, on the same
   reasoning as the reap timeout beside it — a batch benchmark is not a
   live server, so make the timeout irrelevant rather than tune it. This
   hides no leak; `finish_read_prefetched` still releases every lock.
2. **Diagnostics:** an expired read lock is now reported as
   `read_lock_expired` instead of as a write collision, so the next person
   sees the real cause.

**Production semantics were flagged, not changed.**

## Important mitigation

`d43e817a` (report failed MP retrieves as load errors instead of successes)
already downgraded this from *silent wrong answers* to *safe failure*: on a
non-hybrid it is a load error and vLLM recomputes; on a hybrid it fails
loudly. That is why this is filed as a performance/availability issue and
not a correctness emergency.

Worth noting for whoever picks it up: on the hybrid path the loud failure
is fatal to the engine rather than recoverable, because vLLM's
`_update_requests_with_invalid_blocks` unpacks a single KV cache group and
raises on a multi-group model. So "safe failure" means different things on
the two paths, and the hybrid one is harsher.
