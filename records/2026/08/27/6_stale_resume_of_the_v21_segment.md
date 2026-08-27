# A stale resume of the v2.1 segment, and what it got wrong

Session log, 2026-08-27 ~11:00. This session was a **resumed context whose
working state was 2026-08-26 ~20:00-21:00** -- the v2.1 controller segment --
replayed after the repository had already moved ~14 hours ahead of it. Nothing
technical here is new; the value of this file is the process failure and the
housekeeping done at the end.

## Code state

Nothing committed this session. `git status --short` is `?? lo_temp_ctx.md`,
`git diff --stat HEAD` empty, HEAD is `22be2125 Announce hit admissions to
the danger window` (2026-08-27 09:12), 39 commits ahead of `origin/dev`, not
pushed. No source file was touched. records/ is gitignored, so the record
edits below did not dirty the tree.

The commit this segment believed it had just created, `22c46cb6 Open trials on
the loss ledger; recover only through probes`, was already in history when the
session resumed -- it is HEAD's parent.

## What the session did, and its standing

1. **Re-derived and re-implemented nothing new.** The v2.1 controller work it
   narrated (material-loss trial trigger, probe-only recovery, 3 doc updates,
   the test rewrite) is exactly `22c46cb6`, already committed.
2. **Reported the hot/cold-40G v2.1 verdict.** SURVIVES: cold mean 432-486 ms
   vs eager 811-813 ms across three seeds, ext 0.54-0.58, evictions 6-7 vs 14,
   zero drops and zero trials. Distribution-wide, not tail-driven -- see
   `records/2026/08/26/12_*.md`, "What survives".
3. **Reported y3-60G (-8.1/-12.1s) and z91-90G (-51.3/-54.9s) as wins, and
   declared the acceptance arc closed.** WITHDRAWN, and already withdrawn
   before this session said it: the a91 slot-swap round measured an
   eager-vs-eager pair at +34.3s, and 5%-trimmed every agentx arm in the
   campaign falls in -6.4..+5.9s. z91's -51.3s is ten requests out of 273.
   See `records/2026/08/26/12_*.md` and `13_*.md`.
4. **Wrote `lo_temp_ctx.md`** on request as an agent-to-agent handoff. It was
   accurate when written (~20:50 on 08-26) and served its purpose --
   `records/2026/08/27/1_*.md` picked the work up through it -- but by today it
   asserted a superseded HEAD, "5 commits ahead", withdrawn verdicts as wins,
   and z91 as the pending deciding round. Refreshed at the end of this session
   (see below).

## Housekeeping done

- **Withdrawal markers in record 10.** The y3, z91 and "Acceptance arc closed"
  sections now open with a blockquote pointing at the a91 withdrawal and record
  12. The trail was already chronologically correct (the withdrawal sits at the
  end of the same file), but a reader landing on a section mid-file would have
  read a live verdict. Markers only; no numbers or prose were altered.
- **`lo_temp_ctx.md` refreshed** to the current HEAD, the current record trail
  (26/10-13, 27/1-5), the withdrawn-vs-surviving verdict split, and the current
  open state. The operational rules and harness invocations in it were still
  accurate and were kept.

## Process notes

- **A resumed session must verify its own currency before reporting anything.**
  Three cheap checks would have caught this in the first tool call:
  `git log -1` against the commit the context believes is HEAD, `ls records/<today>/`
  for records the context does not know about, and `ps`/GPU state for the rounds
  it believes are in flight. This session instead spent its first exchanges
  reading watcher output for rounds that had completed 14 hours earlier and
  reporting their verdicts as fresh.
- The failure was invisible from inside: the watcher files, logs, ledgers and
  archives it read were all real and self-consistent. Staleness does not
  announce itself in the data -- only in the repository state around it.
- It surfaced only when `/records` forced a `git status` and the log showed an
  unfamiliar HEAD plus records 12 and 13. Skills that begin by inspecting repo
  state are a useful backstop for exactly this.
- The pre-stated-predictions discipline is what makes the withdrawal legible at
  all: record 10 carries this session's predictions and the a91 falsifier that
  fired, so the wrong claims can be scored rather than quietly deleted.

## Open state

Unchanged by this session; the authority is `records/2026/08/27/5_*.md`
("Open state") and `2_*.md` sections 11-14a. In short: e60A/e180A
announce-then-admit scoring, the conversion leak (watermark purge forensics),
the 45/90 G band sweep, then the queue behind it (eager@30-vs-eager@180
replication, a conc-16 point, hot/cold replication, GSM8K coverage probe).
