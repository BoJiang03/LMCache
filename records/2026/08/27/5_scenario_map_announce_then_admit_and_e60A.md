# The scenario map, announce-then-admit, and the e60A round

Session segment following `3_dropped_evicted_root_cause_and_the_idle_drain_result.md`
(same day). Technical continuity lives in `2_why_dropped_evicted_is_44_percent.md`
sections 11-14a; this file is the conversation log.

## Code state

One commit this segment: `22be2125` "Announce hit admissions to the danger window"
(8 files, +358/-6) on `lazy-offload-publish`, atop `22c46cb6`. All lazy offload
suites green: 283 tests, 17 of them new (9 policy, 5 manager, 3 connector).
ruff check/format and isort clean. Not pushed. `lo_temp_ctx.md` remains the only
untracked file, deliberately unversioned.

A second commit was made and REVERTED this segment: a performance-model design
doc briefly landed as `d58b76d9` under `docs/design/integration/vllm/`. The user
corrected the intent ("这个文档是你自己看的，不是写进lmcache的") and the commit
was reset away; the doc now lives here as
`4_lazy_offload_performance_model.md`, gitignored. HEAD history contains no
trace of it.

## The questions, in order

**"为什么 l1=180 有压力吗？lazy 需要在 L1 有压力时表现才好"** -- answered: 180 G
has no L1 pressure at all (residence 449-738 s vs an 85 s turn, 7 watermark
events). The pressure at 180 G is on the GPU block pool, manufactured by L1's
own success: 76.7% hit rate means ~2800-block admissions, invisible to a
token-derived window that excludes external computed tokens. Two different
memories, two different pressures; conflating them was the trap.

**"哪些场景有利、哪些不利？60G 的 4% 合理吗？我觉得有点低"** -- the gap
distribution answered both (record 2 section 11). Agentic inter-turn gaps are
p50 2 s / p75 16 s against ~85 s turns, so lazy (turn-end writes) needs seconds
of residence where eager (turn-start writes) needs ~90-117 s. Favorable band:
residence between gap and turn+gap, roughly L1 20-130 GB at this load. Parity
already holds at the band's floor (30 G: -0.05%) and on no-reuse traffic
(first-turn pairs: -57 ms); the only failed parity edge is big-L1, and that is
the drop bug. And no, 4.3% is NOT reasonable: on rehit-opportunity pairs alone
the win is already -6.5%; recall converts only 16% of opportunity tokens
against a timing coverage of 80%; the ceiling estimate is -15% to -30%. The
user's instinct was right.

**"写个文档，然后推进"** -- the doc went to the wrong place first (see above),
then to records. 推进 became: d60H scoring, the announce-then-admit
implementation, and the e60A round.

## d60H verdict (record 2 section 13)

idle64 +245 ms, idle8 -189 ms, base **-7172 ms** against the in-round eager.
Both idle arms are the 180 G disablement signature (store p50 256, retrieves
19/27 vs base 128). Scorecard: predictions 1, 2, 4 hit; 3 a letter-hit whose
mechanism claim was wrong (idle8 is not intermediate -- there is no tunable
middle ground in idle draining); 5 falsified (base doubled its b32 win; the
60 G effect is -3k..-7k across rounds, not a point). The pre-committed
criterion fired: no shipped knob fixes drops without discarding the win.

## Announce-then-admit (record 2 section 12, commit 22be2125)

The design insight: every after-the-fact wiring loses the race, because the
scheduler consumes a ready lookup result and allocates the burst inside the
same schedule() call, before that call's drain. The deterministic fix is a
one-step hold owned by the connector: first ready query with external hit
tokens returns (None, True) once more and announces ceil(hit/16) blocks; the
policy floors its danger depth at the announced sum; the drain between the two
queries emits the endangered front, whose pins force the burst to dig past
them; the next query admits. Retraction on scheduled/finished/reset, three
nets. Config gate `lmcache.mp.lazy_offload_announce_hits` (default on) exists
for A/B. `announced_bursts` joins the ledger.

Cold admissions never announce: chunked prefill allocates step-by-step and the
token model already covers it -- measured 9-13% drops at 60 G come from the
hit-admission share, 44% at 180 G.

## The smoke saga

The scripted smoke was refused by the scenario (`duration >= 900`), and two
hand-driven attempts produced zero emissions for the RIGHT reason: an idle
24 GiB pool has 16k free blocks and nothing is ever in danger. The loop closed
on the third attempt: saturate the pool, store A, churn a full turnover, reset
vLLM's prefix cache, re-request A -- announced_bursts=1, retrieves=1, request
completed. The mechanism is live on a real engine.

## e60A (in flight at time of writing)

GPU 5 was taken by the multi-modal line mid-morning, so the round runs three
60 G arms on slots 0/1/3 (round.sh gained a SLOTS override): eager,
lazy+ANNOUNCE=false, lazy+ANNOUNCE=true. The 180 G leg follows as a single
arm. Launched 09:23:51, expected ~10:05. Liveness confirmed at 09:31:
announced_bursts=5 in the on-arm's live ledger. Predictions pre-stated in
record 2 section 14: (1) on-arm keeps medD below -2000; (2) drops < 3%;
(3) store p50 stays > 20000 -- the not-a-disablement check, pre-committed this
time; (4) 180 G drops < 2%; (5) 180 G stored volume within 15% of eager.

## Process notes

- The doc placement mistake: "写个文档" after two exchanges about analysis
  quality meant a working document for the user, and the repo convention
  (docs/design mirroring lmcache/) pulled the interpretation the wrong way.
  Reverted cleanly on "退回". Cost: one commit created and reset.
- The idle-pool smoke false starts were not wasted: they are the mechanism's
  premise demonstrated negatively, and they surfaced the aiperf scenario's
  duration floor before the real round could trip on something similar.
- Slot collision avoidance held again: checked GPUs before launch, found GPU 5
  occupied, adapted the round rather than queuing onto a busy device.
- Waiters: done-marker until-loops on the chain log, plus a bounded liveness
  poll on the live ledger. Both patterns worked; no self-matching pgrep.

## Open state

- e60A lands ~10:05; score predictions 1-3, then run e180A_lazy_on and score
  4-5. If prediction 1 holds, the goal's two legs (win in the band, parity
  outside) are both carried by announce-then-admit and the remaining agenda is
  the conversion leak (watermark purge forensics) and the 45/90 G band sweep.
- Queue behind, unchanged: eager@30-vs-eager@180 replication, a conc-16 point,
  hot/cold replication, GSM8K coverage probe.
