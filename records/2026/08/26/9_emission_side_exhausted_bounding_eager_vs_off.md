# 9. Emission side exhausted; bounding eager against off

Session segment following record 8's plan. Code shipped, two acceptance
rounds judged, one bounding round in flight at close. Tree clean at
`230d15bc`; live verdicts were appended to record 8 as they landed, this
file is the segment log.

## Shipped (branch lazy-offload-publish, not pushed)

1. `d03106da` -- default `StoreReleasePlacement` flipped to `LRU_TAIL`.
   One-line default + enum/`__init__` docstrings + design docs; the two
   tests that asserted head behavior off the default harness now name
   `eviction_head` explicitly; default test flipped. 53/53.
2. `230d15bc` -- idle drain + block volume cap, both off by default:
   - `lazy_offload_idle_drain_max_ops` / `lazy_offload_idle_threshold_blocks`:
     on a step whose allocation rate (max of per-step EMA, next-step
     estimate) is at or below the threshold, emit up to N oldest ops,
     admission-FIFO, with all backlog-drain constraints (prefix closure,
     dedup-hole cut, snapshot validation, economy backstop, one in-flight
     batch per request -- the backlog drain now records emissions in the
     shared skip set so the idle pass cannot double-emit).
   - `lazy_offload_max_drain_blocks_per_step`: one `_DrainBudget` (ops +
     blocks) shared by pressure/backlog/idle paths; soft bound -- the op
     crossing it still emits, overshoot charged.
   - Counters `idle_emitted`, `idle_drain_steps`; ledger equation
     unchanged. 236 tests green, ruff clean.

## Verdicts (details and tables in record 8 addenda)

- **u60** (eager x2 vs tail+IDLE_OPS=4 x2, 60G): idle drain harmful.
  medD +28/+29 ms, sumD +40/+41 s, retM 1.19/1.28. Mechanism worked as
  specified (idle_emitted 956/1000, pending=0, drops 180 -> 2) -- the
  hypothesis was wrong: agentx has no truly idle steps, "idle" = another
  request's decode; 4x emissions each paying pin + D2H on the serving
  path, and the wait's filtering given up.
- **v60** (eager x2 vs tail+BLOCK_CAP=64 x2, 60G): worse still.
  medD +49/+51 ms, sumD +102/+104 s, 46% drops. Attribution corrected
  in-session: drain-step counts are in family with tail-only (~81K/round;
  u60's 5.4K was the outlier) and t60 read 8.2M queue blocks while
  staying mixed, so per-step machinery cost does not correlate with the
  losses. Leading cause: receipt serialization -- 64-block fragments x
  one in-flight batch per request stretch a due chain over ~9
  emit-receipt round trips, its tail dying at the eviction edge, every
  fragment still in phase with the burst.
- **Emission-side scoreboard at 60G** (vs in-round eager): head -17s /
  tail -7..-10s / tail+hz10 mixed / tail+idle4 -40s / tail+cap64 -102s.
  Everything that changes when or how finely lazy copies loses to plain
  tail waiting. The residual 7-10s has no validated mechanism yet.

## w60 verdict (landed 15:43)

**w60**: off x2 vs eager x2 at 60G -- first-ever off arm on agentx.
The decision rule (off ~ eager => parity ceiling; off < eager => store
less) assumed off <= eager. Both branches falsified: off is far worse.
sumD +248.5/+218.5 s, p90 5722-6389 vs eager 2160-2577, 49/53 pairs
lost >1 s, pair count 226/227 (43-44 requests out of pairing window).

Meaning: L1 at 60G carries ~230-250 s of TTFT value on agentx even
though reuse is below pool turnover -- the tail L1 serves is large.
Eager banks all of it; lazy tail banks ~97% and is 7-10 s short.
"Store less" is dead as a direction. The residual is value-capture or
machinery: lazy stores later and less (sto 330-840 vs eager ~1050,
dropped ops never reach L1) so hits can be smaller/missed at equal
retrieval counts, and/or pins+copies on the serving path. Cheapest next
probe: retrieved token volume per arm from existing r60/s60/w60 logs,
no new round needed. Full table in record 8.

## Harness notes (old-session scratchpad par/)

- Slot->GPU remap while another session holds GPUs 1 and 6:
  SLOT 1->GPU2, SLOT 3->GPU3 (env.sh). Revert when they free up.
- env.sh gained IDLE_OPS/IDLE_THRESH/BLOCK_CAP (default off) wired into
  up.sh KV_ARGS; STORE_RELEASE env default changed eviction_head ->
  lru_tail to match the shipped default.
- Config-echo check worth keeping: grep the arm's vllm log for
  "lazy offload enabled with EVICTION_AWARE policy:" to confirm knobs
  reached the policy before trusting a round.

## Standing

User's bar unchanged: lazy >= eager on unfavorable workloads, stably >
eager on favorable. Favorable (hot/cold) and 90G already met with tail;
60G open pending w60's bound. Branch not pushed; push to fork
BoJiang03/LMCache only, user opens PRs.
