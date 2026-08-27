# Knob verdicts, the code-change audit, and the correctness gap

Date: 2026-08-26 (continued from record 4)
Code: `1c43ca02`, working tree clean throughout this segment.
Workload as record 4: agentx-mvp compliant, L1 = 90 GiB unless stated.

## 1. Knob sweep, in-round paired (`m_*` round)

Baseline `m_eager_s0` (GPU 0), n=273, sumTTFT 283.5 s.

| arm | knob | sumD | p99 | stores | dropped_evicted | backlog_emitted | GPU prefix hit | retrieved |
|---|---|---|---|---|---|---|---|---|
| m_lazy_s1 | default | **-30.3 s** | 7696 | 224 | 127 | 0 | 24.6% | 1.62 M |
| m_cap8_s2 | `max_pending_ops=8` | -2.6 s | 8817 | 907 | 28 | 846 | 16.5% | 1.56 M |
| m_hz10_s3 | `horizon_steps=10` | -26.3 s | **7663** | 250 | 63 | 0 | 10.0% | **1.78 M** |

### `max_pending_ops=8` collapses lazy into eager

The serial-run verdict was right; the mechanism is now visible.
`backlog_emitted=846` out of 907 emissions: 93% of stores were forced out by the
cap rather than timed by the eviction forecast. Store count goes 224 -> 907, next
to eager's 1021. The cap does fix the leak (`dropped_evicted` 127 -> 28) and pays
for it with the entire benefit: p99 back to 8817, GPU prefix hit down to 16.5%,
sumD down to -2.6 s.

`eviction_aware.py` already documents the sizing rule -- size the cap against
`dropped_evicted`, because "a backlog deep enough to lose operations is deeper
than the workload can defend". 8 is far below that. Observed end-of-run `pending`
is 8-40 across arms, and the leak is ~127 ops, so a cap that bounds burst damage
without retiming the policy is order 64-128. 8 is not a bad cap, it is a
different policy.

### `horizon_steps=10` halves the leak for free

Storing 4x earlier: `dropped_evicted` 127 -> 63, latency unchanged (-26.3 vs
-30.3 s, inside the +-5 s replicate band), best p99 in the round (7663), highest
retrieved volume (1.78 M), and `pending` drains from 40 to 1.

This is the first knob that improves the ledger without costing latency. It needs
replicates before it becomes the default, but it is the leading candidate for the
`dropped_evicted` leak that record 4 listed as unfixed.

### Fifth compliant lazy replicate

`m_lazy_s1` at -30.3 s. The compliant set is now -28.8 / -26.4 / -24.1 / -42.3 /
-30.3 s, mean -30.4 s, 5/5 wins.

## 2. Code-change audit over the measurement campaign

Asked directly: did the code move while we were measuring agentx? It did, twice,
and both are inert on the configuration we ran. Verified rather than assumed:

| commit | when | effect on our runs |
|---|---|---|
| `5ea3cc6e` | 08-25 16:28 | the code state record 1's L1 sweep ran on |
| `66c64116` | 08-26 08:20 | none. All of `max_pending_ops` is gated behind `if budget > 0 and self._config.max_pending_ops > 0`; every round used 0. `backlog_emitted=0` in every default-config ledger is the commit's own sensor for exactly this, documented as "zero means the cap never bound and the policy behaved exactly as an unbounded backlog would". |
| `1c43ca02` | 08-26 10:28 | none on the default path. The single behavioural change is a conjunct added to an existing condition: `if self._store_release is EVICTION_HEAD and (_free_blocks_accepts_prepend(...))`. With the default `EVICTION_HEAD` this reduces to the pre-commit expression. |

Round-to-code mapping: `p_*` (09:48) ran on `66c64116`; `g_*` / `h_*` / `k_*` /
`m_*` (10:15 onward) ran on the `1c43ca02` tree. So the five compliant lazy
replicates span a commit but not a code path.

Corrected along the way: `throttled_drains` (1-2 per run) is the sizing sensor
for `max_drain_per_step`, a pre-existing knob, not for `max_pending_ops`. Nonzero
there does not indicate the new cap was active.

### Process defect this exposed

`/home/bo/venvs/vllm-lazy` is an editable install pointed at this worktree, so
the live code is the working tree at each server's start time. `round.sh` staggers
its four arms by 20 s, so an edit inside that window would give arms in the *same
round* different code. The `1c43ca02` edits landed near the `g_*` round's stagger;
they happened to be inert, so nothing was lost.

Rule from here on: **no working-tree edits while a round is in flight.** Code
changes go between rounds.

## 3. The correctness gap, and closing it

The performance campaign cannot see a correctness bug, by construction. Record 1
already states it: agentx content is "deterministic synthetic tokens reconstructed
from hashes ... useless for anything content-dependent". Sixteen arms of output
text carry no correctness signal. That is why record 1 kept the GSM8K run.

GSM8K was last verified on 08-19 (`af0525ec` / `d419ab3b`), when the code branch
was at `924e2c1c`. All three commits above landed after it, so no correctness
check covers the current tree.

Of the three, `5ea3cc6e` is the one that carries risk: "Make a deferred store
visible to the next request over its prefix", +566 lines across the manager,
pending store, eviction-aware policy and `lmcache_mp_connector.py`.

Reading the call site narrowed what that risk actually is, and my first framing
of it was wrong. I wrote that it "makes a store that has not been physically
written yet answer a lookup as available". It does not touch the lookup path.
`LMCacheMPConnector._skip_pending_covered_prefix` advances the *store* watermark
`num_stored_tokens` past a range a buffered operation already covers, so the
request does not re-stage a prefix its predecessor has staged. The retrieval side
is unchanged. Two consequences:

- If the operation it trusted never lands (`dropped_evicted`, `dropped_failed_store`),
  the skipped range is stored by nobody and a later request gets a *shorter* hit
  than it should. That is a performance loss, not a wrong answer -- and it is a
  second, independent reason to care about `dropped_evicted`, which
  `horizon_steps=10` halves.
- It corrupts only if the pending index misidentifies content. The index is keyed
  `(cache_salt, block_id, hash snapshot)` specifically so a stale snapshot cannot
  pass as current; a defect in that key would put wrong KV under a real key, which
  is exactly what `accuracy.py` exists to catch:

> A wrong offset, a chunk filed under the wrong position, or a prefix rebuilt to
> the wrong length all return *correct bytes* attached to the *wrong tokens*.

Launched 11:27 on the idle GPU 2, `SMOKE_REPO` pointed at HEAD: Qwen3-8B, 20-shot
greedy, 120 questions, concurrency 4, L1 = 68 GB, pool 2048 blocks, off / eager /
lazy x 3 reps. The probe's strength is the within-config comparison -- same
engine, same prompts, pass 1 computed and pass 2 served out of L1, so any drop is
retrieval corruption with no configuration difference to blame. Vacuity guards:
pass-2 `apc` must be near zero (else the GPU served the prefill) and `ext` well
above zero.

Reference band from the last verified sweep (strict score, cold -> cached):

| mode | cold | cached | delta |
|---|---|---|---|
| off | 0.925 / 0.925 / 0.917 | 0.917 / 0.917 / 0.933 | -0.008 / -0.008 / +0.017 |
| eager | 0.925 / 0.925 / 0.933 | 0.917 / 0.917 / 0.925 | -0.008 / -0.008 / -0.008 |
| lazy | 0.917 / 0.933 / 0.917 | 0.925 / 0.925 / 0.925 | +0.008 / -0.008 / +0.008 |

+-0.017 is +-2 questions of 120, and all three modes sit inside it. A prefix
misalignment collapses the score rather than moving it two questions, so the
resolution is sufficient.

### Verdict: clean, with one coverage gap that matters

Nine cells, all guards clean, finished 11:51. Cross-config strict delta against
the matching `off` recompute reference:

| tag | strict delta | same answer | same text | divergence p50 |
|---|---|---|---|---|
| eager rep 0 / 1 / 2 | -0.008 / -0.017 / -0.017 | 119 / 118 / 118 of 120 | 89 / 89 / 80 | 67 / 58 / 74 |
| lazy rep 0 / 1 / 2 | -0.017 / -0.008 / -0.017 | 118 / 119 / 118 of 120 | 88 / 79 / 81 | 79 / 76 / 67 |

The decisive comparison is not the delta against `off`, it is eager against lazy,
and the two delta sets are *identical*: each is {-0.008, -0.017, -0.017}. Eager
does not execute the deferred-store path at all, so whatever produces the one-to-
two-question drift produces it equally without lazy, and it is not caused by these
commits. Within-config repeat noise measured in the same sweep is one question
(eager 0.917/0.908, lazy 0.908/0.917, off 0.925/0.925), so the drift is at the
floor of the instrument.

Every lazy ledger closes exactly -- `admitted` = `emitted` + `dropped_evicted` +
`pending` (198 = 187+7+4, 197 = 185+8+4, 199 = 188+7+4) with zero
`dropped_failed_store`, `dropped_id_reuse` and `rejected_prefix_broken`.

The gap: **`covered_prefix_advances = 0` in all three lazy reps.** The new code
path of `5ea3cc6e` never fired. `covered_blocks_probed` is 21.4-21.7 k, so the
probe walked blocks and never found a covering pending operation -- GSM8K's pool
is 2048 blocks, eviction pressure arrives fast, and `pending` ends at 4 out of
198 admitted. On agentx the same counter fires hard, 126-149 advances and 1.9-6.1
Mtok skipped per run. So the mechanism is heavily exercised where only performance
is measured, and not at all where correctness is measured.

What this sweep therefore establishes: the retrieval path at HEAD is sound, and
`66c64116` / `1c43ca02` are clean on the configuration we ship. What it does not
establish: that the `5ea3cc6e` skip is correct when it fires. To close that, the
probe needs the skip to fire -- raise the GPU pool so admitted operations stay
buffered long enough for pass 2 to walk into them, confirm
`covered_prefix_advances > 0`, then read the pass-2 score. Until a run shows a
nonzero count, the correctness evidence for that commit is absent rather than
negative.

Co-residency noted for honesty: the accuracy run shares the host with the
in-flight performance rounds. Its host-memory write rate is ~2.9 GB/s against a
machine with hundreds of GB/s, and it occupies a GPU no round is using, so the
perturbation is small -- but the `m_*`, `n30_*` and `n60_*` rounds ran with it
present.

## 4. Queue

- `n30_*` (L1 = 30 GiB, 2 eager + 2 lazy, slots rotated) launched 11:32.
- `n60_*` (L1 = 60 GiB) chained after it.
- GSM8K at HEAD, 2 of 9 cells done at the time of writing (`off` reps 0 and 1,
  both passing their non-vacuity guards).

Rationale for the L1 rounds is in record 4 section 7: record 1's 90 G latency cell
has since been re-measured properly and flipped sign (p99: lazy 59% worse serially
vs 10-13% better in-round), so the 30 G and 60 G latency cells carry no weight.
The volume and mechanism rows of that sweep stand -- they are server-side counters
with 2-3x effects, not latency samples.

## 5. Still open

- Replicate `horizon_steps=10`, then decide whether it becomes the default.
- Re-test `max_pending_ops` at 64-128 rather than 8.
- L1 30 G / 60 G latency, in flight.
- The load ramp. Unblocked as of record 4 but not started. Concurrency 8 gives
  effective 2.4, so saved prefill becomes GPU idle rather than queue relief; this
  is also where lazy's pin pressure could reverse the verdict.
