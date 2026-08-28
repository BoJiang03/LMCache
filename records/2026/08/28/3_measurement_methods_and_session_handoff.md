# 3 — Measurement methods and session handoff

Records 1 and 2 carry this session's findings. This one carries the
**methods** those findings depend on, because all three analyses live only
in a session-scoped scratchpad
(`/tmp/claude-1016/.../3d34c28a-*/scratchpad/par/`) that will not survive,
and re-deriving them from the records alone would be guesswork. It closes
with the state of the branch and the one open decision.

## 1. Why these three analyses mattered

Every methodological correction this session came from re-reading logs we
already had, not from new rounds:

- the NUMA asymmetry (record 1 §1) came from `nvidia-smi topo -m` plus the
  per-slot drop history, and cut the medD spread between identical arms
  from 2029 ms to 92 ms;
- the transient-vs-steady-state problem (record 2 §1) came from replaying
  archived ledgers, and invalidated the operating point of every earlier
  round;
- the reuse-distance CDF (record 2 §1) came from the aiperf artifacts, and
  predicted i60N's cliff before the round finished.

None cost a GPU-minute. When a result looks stable, replay the archive
before running anything.

## 2. Reuse distance (LRU stack distance) from aiperf artifacts

The workload property that governs everything, and the only one that is
independent of L1 size, policy, and run length. For each consecutive turn
pair of a conversation, count the KV tokens every request in between
wrote; the CDF of that distance *is* hit rate as a function of cache size.

Inputs, all from `<arm>/artifacts/profile_export.jsonl`, one JSON object
per request: `metadata.conversation_id`, `metadata.turn_index`,
`metadata.request_start_ns`, `metrics.input_sequence_length.value`.

```python
recs = []                       # (start_ns, conversation_id, turn_index, isl)
for line in open(f"{arm}/artifacts/profile_export.jsonl"):
    r = json.loads(line); md = r["metadata"]; m = r["metrics"]
    isl = m.get("input_sequence_length", {}).get("value")
    if isl is None:
        continue
    recs.append((md["request_start_ns"], md["conversation_id"],
                 md["turn_index"], isl))
recs.sort()
cum, W = 0, []                  # W[i] = KV tokens written before request i
for _, _, _, isl in recs:
    W.append(cum); cum += isl
convs = {}
for i, (t, cid, ti, isl) in enumerate(recs):
    convs.setdefault(cid, []).append((ti, i))
gaps = []                       # reuse distance in tokens, per turn pair
for v in convs.values():
    v.sort()
    for (_, i1), (_, i2) in zip(v, v[1:]):
        gaps.append(W[i2] - W[i1])
# hit rate for a cache of `gb` gigabytes at KV_BYTES_PER_TOKEN bytes/token
cap = gb * (1 << 30) / KV_BYTES_PER_TOKEN
rate = sum(1 for d in gaps if d <= cap) / len(gaps)
```

Use an **eager** arm as the source: it stores everything, so its ISL stream
is the closest available proxy for KV produced. Caveats: ISL approximates
tokens written per request; the CDF is the *full-prefix* hit rate, so
chunk-level partial hits make the real curve smoother than the step; and
server-side content dedup means the resident working set is smaller than
the written volume (at 120 G, 533 GB written occupied 89 GB).

Result on `agentx-std` (364 turn pairs, 23.84 M tokens): p50 distance
1038 k tokens = 95 GB, p50 time gap 89 s; 60 GB → 4.7%, 120 GB → 72.3%.
See record 2 §1 for the full table. This does not need re-deriving unless
the dataset, `ENTRIES`, `CONC`, or `max-context-length` change — **it is a
property of the workload definition**, so recompute it whenever any of
those move, and pick L1 from it rather than by habit.

## 3. Convergence replay (`converge.py`)

Answers "is this metric stationary, or am I measuring a transient?" from
an archived arm, no re-run needed. Parses `[YYYY-MM-DD HH:MM:SS,mmm]`
stamps out of `vllm.log.gz` (`Lazy offload counters:` lines, after
stripping ANSI with `\x1b\[[0-9;]*m`) and `server.log.gz` (`Stored N
tokens` / `Retrieved N tokens`), accumulates both, and prints admitted,
drops, drop rate, mean deferral, stored, retrieved and utilization at
t ∈ {300, 450, 600, 900, 1200, 1500, 1800, 2100} s from first activity.

The tell that started record 2: utilization falling monotonically
65% → 18.6% while stored rose linearly and retrieved flattened.

## 4. Windowed medD (`early.py`)

Answers "how long must the load run for this verdict to hold?" Pairs
requests on `(conversation_id, turn_index)` — the only stable key, per
`cmp2.py`'s header — and restricts each window to requests whose
`request_start_ns` is within τ of that run's own first request, then
reports median TTFT delta and pair count per τ.

Two things it showed that no single-number comparison could:

- at 60 G the signal peaks at t≈1200 s (−8528 ms) and is then **diluted to
  −3520 ms** by a tail of hit-free requests;
- the arm-to-arm spread shrinks with n (4396 ms at 600 s → 92 ms at full),
  so short rounds buy speed with precision — which is only acceptable when
  the effect is large, as it is above the knee.

Run it on every round from now on; record 2 §4 makes it part of the
standard report.

## 5. Branch state

`lazy-offload-dev`, nothing pushed, tree clean at `a2ffad06`:

| commit | what |
|---|---|
| `a2ffad06` | record 2 — reuse-distance cliff, `agentx-std` |
| `9f99db74` | record 1 — three ceilings, the off arm |
| `0b75fa3b` | `emitted_deferral_drains` / `dropped_deferral_drains` |
| `aed6c7bc` | stop tracking `lo_temp_ctx.md` (mis-added in `55612abb`) |
| `0436a131` | `dropped_evicted_tokens` |
| `80cca26f` | record 12 (previous session) |

All three code commits are pure observability — no policy behaviour
changed, so no earlier round is invalidated by them. 225 lazy tests green,
ruff clean. Rounds i60J through i60N (five rounds, fifteen arms) are
scored in records 1 and 2.

## 6. The open decision

Everything scored before i60N — eager vs lazy vs off, the danger floor,
announce, the whole simplification — was measured at L1 = 60 G, which
record 2 §1 shows is the worst point on the reuse curve: a cache with a
~5% steady-state hit rate, where all observed benefit is the filling
transient. Above the knee the system is a different machine: writes drop
75%, the read/write ratio crosses 1.0, preemptions fall 122 → 25.

**The relative verdicts may not survive the move.** The next round should
therefore be eager vs lazy vs off on `agentx-std` at 120 G — the same
three-way comparison as i60M, re-run where the cache actually works, at
20 minutes instead of 41. Proposed and not yet started; the user was asked
and has not answered, so nothing is running.

Carried forward unchanged: warm-cache preload for the harness (record 2
§4), X4's falsification (drops peak at 120 G, fall at 240 G — record 2
§6), `max_drain_blocks_per_step` never tried, and the gate-3
`min_prefix_tokens` sweep the user deferred until the store line closes.
