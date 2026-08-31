# Session log: the sweep grows to three policies, and the current lazy default turns out dead

Conversation record for 2026-08-30 late night into 08-31 morning,
continuing from record 2026/08/30/13. Bo extended the A/B into a
three-policy comparison and set the endgame (gsm8k gate, then PR prep).
Along the way: a two-driver incident with an orphaned session, and the
discovery that upstream's current lazy default policy never offloads at
all on this workload.

## 1. Two more eager/EA pairs: 32 and 40

Bo asked for a point with TTFT near 5 s. CONC=32 overshot down (eager
TTFT avg 3.38 s), CONC=40 landed it (5.37 s). Both pairs favor lazy
(EVICTION_AWARE), margins narrowing as load falls:

- 32: TTFT avg -14.9%, thpt +2.5%, requests +6.1%; L0 dominates
  (54-59%), external only ~15-16%. rejected_unhashed rises to 7.5% at
  low load (blocks live longer, more out-of-window SWA nulls at store
  time).
- 40: TTFT avg -4.5% (5.13 vs 5.37 s), thpt +3.1%, ITL p99 -24.7%,
  external 34.3% vs 30.4%.
- Eager TTFT curve over CONC: 32 -> 3.4 s, 40 -> 5.4 s, 48 -> 10.1 s,
  72 -> 48 s. Queueing takes off between 40 and 48.

## 2. The two-driver incident

The first e40 died strangely (engine "ready" in 37 s, no aiperf export,
external split nonsense). Cause: TWO ab_chain.sh instances — this
session launched e40/l40 at 23:16, and an orphaned session launched the
same pair at 23:10. The orphan (`lazy-offloading-b8`) was a resumed copy
of this session's own pre-compaction transcript: it shared the chain-1-3
history, believed it was driving the sweep, and its knowledge of Bo's
instructions stopped before 23:00 (no FIFO plan, no PR endgame). It had
competently quarantined the ruined arm (bad_run1/), added a flock guard
to ab_chain.sh, and relaunched e40/l40 clean at 00:10.

Resolution: cross-session messages established the facts; Bo confirmed
he only sees this conversation; the orphan was killed by pid at Bo's
hand (the auto-mode classifier refused to let this session kill a peer
Claude process, correctly). Clean e40/l40 came out of the orphan's own
relaunch. Lessons: the harness can re-execute a nohup launch (hence the
flock); watcher loops must not pgrep a pattern their own cmdline
contains (rewritten to probe the flock instead); the lock fd must be
closed in child processes (9>&- on every nohup) or a surviving MP server
blocks the next chain.

## 3. FIFO: the current upstream lazy default never offloads

Bo added a third strategy: "默认 lazy" = policy FIFO, all defaults. f40
result: **zero external tokens, zero L1 writes for the whole arm**. Not
a config error and not a crash. First reading (corrected on 08-31 from
the engine log, see record 3): the trigger IS reached, and every drain
it produces is dead on arrival. `fifo.py:80` drains only when 100
finished-but-undrained requests coexist; that set never shrinks on its
own, so it climbs to 100 about 11 minutes in (02:20:45) and then
sawtooths, 10 requests per drain. But lazy offload's
`request_finished` returns False, so those requests' blocks are back in
the free queue and get reused long before the drain wakes; the
manager's hash revalidation (`lazy_offload_manager.py:571`) drops the
whole request. 330 drained requests in f40, 330 dropped, 0 stores.
Design doc calls FIFO the "explicit legacy
fallback"; the prior line's own harness only ever ran it with
threshold=1 (driver.py:1171), i.e. with the threshold mechanism
disabled. And on upstream origin/dev, FIFO IS the current default lazy
policy (pending_store: policy default "FIFO"; no eviction_aware.py
exists there).

Bo's ruling: the policy is not ours to fix. f48 confirms (0% external,
TTFT avg 15.2 s vs eager 10.1 s, thpt 154 vs 210). The FIFO arms double
as the no-offload reference curve. Presentation decision: the headline
table shows FIFO at its best point only, other cells "-", one neutral
footnote (留点体面); full numbers stay in records.

## 4. Naming for the comparison

Verified against origin/dev, then fixed: "lmcache default" (eager,
lazy off — the term "eager" stays in records since lazy_offload.md
defines it), "lazy current default (FIFO)" (what upstream gives you
today if you turn lazy on), "lazy eviction-aware (this PR)". The naming
carries the PR narrative: the current lazy default silently does
nothing; the PR's policy is the working one.

## 5. State at close and the standing plan

- f72 running (started 03:57, ~04:55 done). FIFO predictions were
  registered in ab_analysis.md before f40 landed; the zero-offload
  branch was called as the tail risk but at 72, not 40 — reality was
  worse than the registered central case.
- ab_analysis.md snapshot copied to artifacts/ab_analysis_snapshot.md
  (the live copy is in the session scratchpad sweep/ dir with all raw
  arm artifacts; full copy into records happens at PR-prep time).
- Endgame approved by Bo: finish f72 -> three-policy table -> gsm8k
  (lazy EA on, verify the "lazy offload enabled" log line) -> branch
  lazy_offloading_policy_pr off latest upstream dev with only the
  necessary complete pieces + gsm8k again -> branch
  lazy_offloading_policy_dev with records and experiment material ->
  PR title/body doc in records for Bo to copy -> push both to the fork,
  Bo opens the PR.
