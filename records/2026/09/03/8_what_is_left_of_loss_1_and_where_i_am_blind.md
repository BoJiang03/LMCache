# What is left of loss #1, and where I am blind

Follows record 7, which shipped the delta fix and measured it at 4.18 of
7.97 ms/step. This one squeezes the existing pstats for free — no GPU — to see
what the remaining `+3.78 ms/step` is made of, and ends by naming the part I
cannot see at all.

Nothing here was measured today. It is a re-reading of chain23's dumps
(`pmp.*.pstats`, `pns.*.pstats`, window 2400:3000, 8 workers), which profile
the **unfixed** build.

---

## The free decomposition: `mp` minus `nostore`, spin rows excluded

Tool: `harness/scripts/nonspin_diff.py`. It drops rows matched by name against
a spin list rather than by threshold, because at 3–5k calls/step cProfile's own
per-call overhead dominates their milliseconds — their call counts are the
signal, their times are noise.

Directly attributable connector work:

| frame | Δtot ms/step | Δcalls/step |
|---|---|---|
| `msgspec._core.msgpack_encode` | 1.013 | 5.75 |
| `_create_key` | 0.243 | 0.96 |
| `socket.py:638(send)` | 0.094 | 5.75 |
| `event_ipc.py:218(export_event)` | 0.008 | 0.96 |
| **total** | **~1.36** | |

That 1.36 ms/step of Python bought 6.19 ms/step of step time
(`mp` 91.90 − `nostore` 85.71). **Amplification 4.5×.**

Cross-check against the shipped fix: it removed roughly
`(0.243 − 0.03) + (1.013 × 6/7) + (0.094 × 6/7) ≈ 1.16 ms/step` and delivered
4.18. **Amplification 3.6×.** The two estimates agree to within the precision
either deserves, which is the first independent confirmation of the
amplification factor — record 6 had it from one intervention only.

## What the same table rules OUT as targets

Two classes of large deltas here are **not** optimisation targets, and reading
them as such is how this session would waste its next day:

- **Spin-driven builtins.** `time.monotonic` +0.228 ms/step at **+1948
  calls/step**, `_thread.lock.__exit__` +0.217 at **+1954 calls/step**. These
  are called *from* the busy-wait; `mp` spins 61% more than `nostore` (5115 vs
  3167 `sched_yield` calls/step, record 6). They measure the delay, they do not
  cause it.
- **Unchanged work running slower.** `_launch_kernel` +0.880 ms/step at
  **Δcalls 0.00**; likewise `torch.mm` +0.341, `matmul_ogs` +0.194,
  `torch.addmm` +0.169. Same call counts, more time — the amplification landing
  on the GPU launch path. Nothing to remove.

`nonspin_diff.py` prints that caveat under every table so the next reader does
not have to rediscover it.

## The implication for what is left

After the fix, the store path's attributable Python is roughly:

```
encode        ~0.15   (24 KB instead of 180 KB)
_create_key   ~0.03
socket.send   ~0.014
export_event   0.008
              ─────
              ~0.20 ms/step
```

At 3.6× that is **~0.7 ms/step** of the residual `+2.01` (`nostore` → `mpfix`).
So **most of the remaining store-side cost is not the payload any more.** It is
the round trip, the hook machinery, or something not yet named. Lever (c) —
having the scheduler send the delta once and workers ship only a reference —
would chase that 0.7 and no more. It is no longer the biggest remaining item.

## Where I am blind

`+1.77 ms/step` of the remaining `+3.78` is the `none` → `nostore` step, and
**I have never profiled `none`.** chain23 dumped `mp` and `nostore` only. That
is 47% of what is left, completely unattributed.

The single datum I have is negative: `_pickle.loads` is 1.25 ms/step in `mp`
and 1.29 in `nostore` — equal, so it is not the `mp`-vs-`nostore` delta. Whether
it constitutes the `none`-vs-`nostore` delta cannot be answered without a `none`
profile. `none` loads no connector, so it has no connector metadata to unpickle
at all; the difference could be most of the 1.77 or none of it.

I am not designing against that gap again. This session has already been wrong
twice from reasoning past missing data — the median-as-measurement error
(record 6) and the single-rank test model (record 7).

## Proposed next step

One cProfile chain over the **current** build, three arms: `none`, `nostore`,
`mpfix`. It decomposes both open items at once — `none`→`nostore` (+1.77, never
measured) and `nostore`→`mpfix` (+2.01, now known not to be payload). About 40
minutes; profiled arms run ~7.5% slow, which record 6 showed does not distort
the comparison (709.5/660.3 profiled vs 686.0/639.0 unprofiled, +7.5% vs +7.4%).

Candidate levers, to be chosen **after** that data, not before:

| if the profile shows | lever |
|---|---|
| metadata pickle/broadcast dominates | metadata carries block-id deltas only; server derives ranges |
| hook-call overhead dominates | early-out on steps with no store, before `_get_connector_metadata` |
| the zmq round trip dominates | lever (c): scheduler submits, workers ship a reference |
| no attributable Python, pure amplification | move the LMCache client out of the worker process entirely |

The framing that holds regardless: at TP=8 the multiplier is 3.6–4.5×, so
**any per-step Python left in the worker process is priced at four times its own
cost**, and the structural end state is that the worker executes none of it.

## Housekeeping: seven stale monitors

Seven Monitor pipelines from earlier chains were still running, the oldest
since 09-02 10:53 (`logs/phase1b.out`). `TaskList` reported none of them — the
task records were lost across compaction while the processes survived.

The one that nearly got missed was an **orphaned `tail`** whose `ugrep` stage
had already died, so searching for the filter stage did not find it. The
reliable sweep is to walk the `claude bg-pty-host` / `bg-spare` process tree and
list every descendant, rather than grep for expected command names.

All seven stopped by explicit pid (never `pkill -f`). On exit each flushed its
last backlogged line as a fresh notification — `CHAIN21: batch done 08:22:10`
and similar — which are historical output, not new events.

## State

- `fix_mp_store_key_prefix_resend_pr` @ `88ccf635`, pushed to fork. PR not opened.
- `vast_repro_dev` @ this commit; `50c6cf7f` carries the fix here too.
- No GPU, lane, vllm, lmcache, tail or grep processes of mine running.
- Still owed from the original directive: loss 2 (`vllm_v1_adapter.py:1102`,
  the 480 KB pageable H2D copy at 33.7 ms/call) and reproducing VAST's second
  reported problem. Both need GPU time.
