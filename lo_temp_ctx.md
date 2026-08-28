# lo_temp_ctx.md -- lazy-offload handoff context

**Refreshed 2026-08-27 ~11:20** (corrects an 11:07 refresh written from a stale
resume that predated the e60A/e180A scoring). If you are resuming a context
older than this timestamp, read "Currency check" first.

## Currency check (do this before reporting anything)

1. `git log -1 --format='%h %s'` -- if it is not the HEAD below, later sessions
   have moved on; find their records before trusting your own conclusions.
2. `ls records/2026/08/2*/` -- read anything your context does not know about.
   The record trail is the authority, in file order.
3. `nvidia-smi` + `pgrep -af aiperf` -- rounds your context thinks are in flight
   may have finished hours ago, and GPUs it thinks are yours may not be.

## Code state

- Branch `lazy-offload-publish` in worktree `/home/bo/LMCache-worktrees/lazy_offloading`.
- HEAD `22be2125 Announce hit admissions to the danger window` (08-27 09:12),
  39 commits ahead of `origin/dev`, **not pushed, no PRs**.
- Tree clean; this file is the only untracked one, and stays untracked.
- Lazy-offload suites: **344 passed / 13 skipped** at HEAD
  (`/home/bo/venvs/vllm-lazy/bin/python -m pytest tests/v1 -k lazy -q`,
  verified 11:05); ruff + isort clean. Lint via `~/.local/bin/ruff`.

## The feature

Lazy offloading defers KV store emission until GPU blocks approach recycling
(danger window over the free queue) instead of storing at request end (eager).
Policy: `lmcache/integration/vllm/lazy_offload_policy/eviction_aware.py`
(regime machine NORMAL/TRIAL/DEGRADED/PROBE in `observe_l1_pressure`; knob
`lmcache.mp.lazy_offload_degrade_l1_residence_secs`), plus announce-then-admit
on top (22be2125, see verdicts). Docs:
`docs/design/integration/vllm/lazy_offload{,_policy/eviction_aware}.md`.
Controller invariant: degrading may change store *timing*, never *volume*;
every regime change is verified by a bounded trial or probe, never by an
estimate alone.

## Goal and the scenario map (record 2 section 11)

User's bar: significant win in favorable scenarios, parity with eager in
unfavorable ones. The clocks: lazy's stores must outlive the inter-turn gap
(p50 2 s / p75 16 s / p90 144 s), eager's must outlive turn+gap (p50 90 s /
p75 117 s). Favorable band = L1 residence between the two, roughly **L1
20-130 GB at this load** (60 G = 46 s residence covers ~80% of reuse tokens
vs eager's ~7%; 180 G = 450+ s, both ~98%, lazy's only edge is op economy).
Parity already holds at 30 G (-0.05%) and on no-reuse traffic (-57 ms).
Known headroom: 60 G's measured win is far below ceiling -- rehit-only pairs
-6.5%, but lazy converts only 16.1% of 20.1M opportunity tokens despite 80%
timing coverage (est. ceiling -15..-30%). The leak is NOT
eviction-before-reuse (gap-length split flat); suspects: chain truncation at
dropped chunks, watermark batch eviction (one purge/11.7 s), lookup gating.
30 G smoking gun: residence 20 s vs 2 s p50 gaps should cover ~76%, measured
3 retrieves in 30 min.

## Record trail (read in this order)

| file | what it settles |
|---|---|
| `records/2026/08/26/10_*.md` | design + every verdict, chronological; superseded sections carry withdrawal markers |
| `records/2026/08/26/12_*.md` | **the agentx measurement collapse**: all conc-8 agentx verdicts withdrawn, what survives |
| `records/2026/08/26/13_*.md` | why agentx as configured could not resolve the question; correct config (b32 onward) |
| `records/2026/08/27/1_*.md` | agentx reconfigured, the L1 reversal, lazy loses at 180 G |
| `records/2026/08/27/2_*.md` | **technical authority**: drop root cause; sections 11-14a design+predictions, **15-16 the e60A/e180A scorecards and verdict** |
| `records/2026/08/27/3_*.md`, `5_*.md` | session logs (idle-drain round; scenario map, announce-then-admit, e60A launch) |
| `records/2026/08/27/4_*.md` | the lazy-offload performance model (user's own reference; never goes into the repo) |
| `records/2026/08/27/6_*.md` | the stale-resume incident that produced the 11:07 refresh |

## Verdict status, short form

- **Survives**: hot/cold-40G. Lazy cold TTFT 432-486 ms vs eager 811-813 ms
  over three seeds, ext 0.54-0.58, zero drops, zero trials; knob correctly a
  no-op there.
- **Survives**: controller safety as a *ledger* property -- every transition
  trial- or probe-verified, volume neutrality in every round.
- **Survives**: 60 G win at conc 32 -- b32 -3.4/-4.3%, d60H base medD -7172,
  e60A off-arm -7155 reproduced in-round. Round-unstable, honest range
  -3k..-7k ms medD.
- **Stands as the failing edge**: 180 G loss (+16.8% c32L), cause
  dropped_evicted = 44% of admitted -- hits make the system ~50% faster and
  blocks recycle before deferred stores emit.
- **Dead**: idle draining as a drop fix (d60H). idle64/idle8 both ties with
  store p50 256 = disablement signature; no tunable middle ground.
- **Dead as implemented, 08-27 morning**: announce-then-admit (22be2125).
  Design: connector holds the first ready lookup one step (`(None, True)`),
  announces ceil(hit_tokens/block) into `_danger_depth`, drain emits the
  endangered front, next query admits; retract on scheduled/finished/reset;
  gate `lmcache.mp.lazy_offload_announce_hits` (**code default still True**),
  harness env ANNOUNCE, counter `announced_bursts`. Wiring proven perfect
  (one announcement per retrieve at both scales) yet **4 of 5 pre-stated
  predictions falsified** (record 2 sections 15-16): e60A@60G on-arm medD
  -730 vs off-arm -7155 (90% of the win gone) for a drop cut of only
  8.1%->5.7%; e180A@180G drops 29.2% vs the <2% target (was 44%), volume
  did not recover (stored 4.26M vs eager 5.45M, retrieved 21.6M vs 25.9M).
  Per pre-commitment: substantive design failure, not mistuning.
  Where the 60 G win went (hypotheses with counter support, NOT concluded):
  on-arm reused 22% less while storing more; free_queue_blocks_read x4.9;
  leading -- forced emissions pin the shallow front, bursts dig past the pins
  and evict other conversations' still-warm GPU prefix cache (local hits ->
  misses; covered_prefix_tokens -33%); alternative -- earlier emission ages
  chunks into L1's LRU, watermark purges before reuse. 180 G residual-drop
  location unknown: throttled_drains=0 rules out drain-bandwidth starvation.
- **Withdrawn** (08-26): every conc-8 agentx `sumD` verdict, both directions.
  Conc 32 is the floor for agentx claims and also the legal ceiling (42
  eligible traces, no dataset wrap, think time fixed).
- **Never observed in any round**: DEGRADED -> NORMAL recovery
  (`degrade_probe_recoveries` = 0 campaign-wide); unit tests only.

## Open state (as of 11:20 -- nothing in flight)

e60A and e180A are DONE and scored; no round is running. The user was given
three directions and has NOT decided -- **ask before launching anything**:

1. Forensics: log free-queue rank at drop time, replay one round, locate the
   451 surviving 180 G drops before any redesign.
2. Redesign candidates: multi-step hold until the front clears;
   announce-without-pin; allocation-side exemption (burst allocation skips
   blocks with pending stores).
3. Park drops, attack the 60 G conversion leak (bigger bounded upside per the
   map): which stored chunks were gone at lookup time; watermark purge cadence.

Pending regardless: flip `lazy_offload_announce_hits` default to False
(verdict says off; code still defaults True). Queue behind: 45/90 G band
sweep, eager@30-vs-eager@180 replication (largest effect, one round only),
conc-16 point, hot/cold seeds beyond 3, GSM8K coverage probe.

## Measurement rules earned the hard way

- Never `sumD` as a primary metric. Median of paired TTFT (medD) plus a stall
  count; control band ~650 ms.
- Rotate config-to-slot assignment every round; a fixed mapping hid a slot
  bias for a whole campaign.
- Always carry a same-config control arm, and >= 3 replicates before a verdict.
- Pre-state predictions with explicit falsifiers in the record before a round
  lands, then score them, misses included. This caught both the collapse and
  the announce-then-admit failure.
- No source edits while a round is in flight (`records/*.md` exempt).

## Harnesses

- **agentx (par)**: `cd /tmp/claude-1016/-home-bo-LMCache-worktrees-lazy-offloading/84352f47-e330-4d19-88ee-0abf7e23352a/scratchpad`
  then `par/round.sh <mode>:<tag>:<KEY=VAL,...> ...` (slots -> GPUs 0/1/5/6,
  ports 27100+SLOT*10, `SLOTS="0 1 3"` override exists). Env:
  `L1_GB CONC ENTRIES DUR GRACE SEED`; per-arm overrides after the tag
  include `DEGRADE_SECS` and `ANNOUNCE`. Compare:
  `python3 cmp2.py par/<baseline> par/<arms...>`. Per-arm archives in
  `par/<tag>/` (`snapshot.txt`, `vllm.log.gz`); ledger greps from
  `par/logs/slot<N>/lazy_vllm.log`. Chain examples: `par/chain_e60A.sh`,
  `par/chain_e180A.sh`. ~35 min/round at DUR=1800. Constraints: the aiperf
  scenario `inferencex-agentx-mvp` refuses duration < 900 s; served model
  name is `agentx`. Smoke recipe: a fresh pool has no danger window and
  nothing emits -- saturate with filler requests first, then vLLM
  `/reset_prefix_cache` to force external hits.
- **hot/cold (longdoc)**: `cd /tmp/claude-1016/-home-bo-LMCache-worktrees-lazy-offloading/716b0498-fe27-43d8-9271-8673da1c54bc/scratchpad/hotcold`
  then `env SMOKE_REPO=<worktree> SMOKE_PYTHON=/home/bo/venvs/vllm-lazy/bin/python
  SMOKE_VLLM=/home/bo/venvs/vllm-lazy/bin/vllm PATH=/home/bo/venvs/vllm-lazy/bin:$PATH
  CPATH=/raid/data/hub/pr4499_agentic/pydev/usr/include/python3.12:/raid/data/hub/pr4499_agentic/pydev/usr/include
  SMOKE_GPU=<free gpu> SMOKE_DEGRADE_SECS=<n> setsid nohup ./run_hot_cold.sh > log 2>&1 &`.
  System python3 lacks torch -- the SMOKE_* env is mandatory. The script has a
  GPU settle guard. Prior rounds archived under `logs/knob_v1/`, `logs/knob_v2/`,
  `logs/pre_degrade_baseline/`.
- Both live in /tmp session scratchpads; if a path vanished, the records
  describe the setup well enough to rebuild.

## Operational rules (user's, standing)

- Everything stays **local**. No push (fork `BoJiang03/LMCache` only, and only
  when told), **no PRs ever**.
- Git author/signoff is always `Bo Jiang <bo.jiang@temple.edu>` via the
  repo-local config. No `Co-Authored-By: Claude`, never `-c user.email`.
- Replies to the user: short, a few lines. Detail goes in records, not chat.
  Commit and record text: terse, no AI flavor, no em dashes. Docs for the
  user's own reference go in records/, NOT into the repo (corrected once on
  08-27; the offending commit was reverted).
- Shared box: never rebuild `lmcache/*.so`, venvs, `/raid`, `/usr/local`.
  GPUs 2/3/4/7 belong to other lines and GPU 5 was claimed by the
  multi-modal line on 08-27; historically safe set 0/1/6 (slots 0/1/3) --
  but ALWAYS check `nvidia-smi` first.
- Never bare `git stash` (tagged push + apply by SHA only; the stash stack is
  shared across worktrees and sessions).
- Launch long jobs under `setsid` and kill by session group; scope `pkill`
  patterns to script paths, never to session-id strings (a session-id pattern
  once matched the shell running it and self-killed a cleanup pass).
- Arm a watcher on every long round; watchers die across session restarts but
  the nohup'd measurement survives -- re-arm, do not relaunch.
- `/records` skill: commit code state + dated conversation record. records/
  is gitignored (`.git/info/exclude`).
