# 2026-09-02 (6) — Localising the IP surcharge to a `time.sleep` in the scheduler, and what is left

> ## CORRECTION, same day, 17:05 — the fix in this record measures ZERO
> ## improvement.  Read this box before believing anything below it.
>
> 1f ran the patched LMCache against the identical 1b configuration:
>
> | | duration | tok/s | P99 TTFT | mean TTFT |
> |---|---|---|---|---|
> | 1b unpatched | 727.7 s | 82,454 | 718.4 s | 372.3 s |
> | **1f patched** | **728.1 s** | **82,407** | 718.0 s | 370.9 s |
>
> 0.06% apart.  The pre-registered band in this record said "REFUTED if
> >= 715 s".  **Refuted.**
>
> **What was wrong.**  This record's mechanism says the stall is O(pending)
> per scheduling pass, sized from `Deferred` peaking at 926 as "up to 9.2 s of
> dead engine time in a single pass".  That was an upper bound, and it never
> happened.  The patch's whole effect is to collapse many sleeps within one
> pass into one; it changed nothing, so **there was only ever about one
> `lookup_cache` call per pass to collapse.**  `max_num_batched_tokens` is
> 8192 and each waiting request wants 8192 tokens, so the scheduler's token
> budget ends the loop after roughly the first request — it never walks the
> deferred queue.  Any sentence below that reasons from `Deferred` peak x 10 ms
> is void.
>
> **What survives.**  The two defects are still real as defects: sleeping under
> the lock the response thread needs is wrong, and an O(pending) backoff is
> wrong to leave in place.  The tests still bite (3 of 5 fail on the old code).
> What does not survive is the claim that fixing them buys throughput.  The PR
> must not claim a performance win.  Decision on the PR is parked until 1g.
>
> **What replaces it.**  One 10 ms sleep per scheduling pass, on the engine
> thread — not per request.  See record 7 for the per-step decomposition that
> sizes it, and for 1g, which tests it with `lookup_backoff_time: 0.0` and no
> source change at all.

## Headline

The 6% that IP costs **over MP** is localised to a single `time.sleep()` in
`LMCacheAsyncLookupClient`, called on vLLM's scheduler thread. A fix is written,
tested and pushed to the fork as `fix_async_lookup_backoff_stall_pr`.

**This is a partial result.** It does not touch the ~9% that IP and MP pay in
common — the "LMCache costs 10% while doing nothing" that is the actual
objective. That remains open and is the next work item.

---

## The chain, end to end

Each link was read in source, not inferred.

1. `Deferred: N` in the vLLM engine log is `num_skipped_waiting_reqs`
   (`vllm/v1/metrics/loggers.py:240`), i.e. `len(self.skipped_waiting)`
   (`vllm/v1/core/sched/scheduler.py:1995`).

2. In our configuration (no LoRA, no structured output, FCFS) exactly one path
   can put a request on that queue — `scheduler.py:605-612`:

   ```python
   ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(...)
   if ext_tokens is None:
       request_queue.pop_request()
       step_skipped_waiting.prepend_request(request)
       continue
   ```

   The other producer (`load_kv_async=True` → `WAITING_FOR_REMOTE_KVS`, line
   767) is unreachable for IP: `lmcache_connector_v1.py:167` returns a hardcoded
   `False`.

3. `None` originates at `vllm_v1_adapter.py:1434`, propagating the lookup
   client's return value.

4. With `enable_async_loading: true` the factory
   (`lookup_client/factory.py:80`) builds `LMCacheAsyncLookupClient`.

5. That class sleeps on the scheduler thread in two places, default 10 ms
   (`lookup_backoff_time`, `lmcache_async_lookup_client.py:153`):

   ```python
   # line 219-221, first lookup
       self.push_sockets[i].send(msg_buf, copy=False)
   time.sleep(self.lookup_backoff_time)
   return None                      # -> guarantees a Deferred

   # line 168-173, every re-poll while pending
   with self.lock:
       ...
       elif req_status is None:
           time.sleep(self.lookup_backoff_time)
   ```

6. A deferred request keeps status `WAITING`, so `_is_blocked_waiting_status` is
   false on the next pass and the connector is called again. Under FCFS
   `_select_waiting_queue_for_scheduling` is
   `return self.skipped_waiting or self.waiting` — **the deferred queue is
   drained first on every pass**. So one pass sleeps once per pending request.

## Two distinct defects

**(a) The sleep at line 173 is held under `self.lock`.**
`process_responses_from_workers` (line 232) needs the same lock to record
worker results. The scheduler therefore blocks the very thread producing the
answer it is waiting for. Longer waits → more pending → longer waits.

**(b) The stall is O(pending), not O(1), per scheduler pass.** Measured
`Deferred` peaked at **926** in 1b, i.e. up to 9.2 s of dead engine time in a
single pass.

Arithmetic that fits: IP costs 42.6 s more than MP at c=1000
(724.1 vs 681.5). ~1000 requests × ~4 polls × 10 ms ≈ 40 s.

## This is VAST's configuration, not ours

Checked against the PDF directly (`pdftotext`): **both** their IP and MP yaml
set `enable_async_loading: true`, and neither sets
`extra_config.lookup_backoff_time`, so the 10 ms default applies. Our
`configs/lmcache_gpu_only.yaml` copies it faithfully. The surcharge is not a
self-inflicted artifact of our harness.

## The fix

`fix_async_lookup_backoff_stall_pr` @ `fc9b0de6`, off `origin/dev` @ `cd441cdf`
(the file is byte-identical at that commit to the one analysed). Pushed to the
fork; **PR not opened**.

- New `_yield_to_lookup_threads()`: runs outside `self.lock`, rate limited
  globally so a pass over N pending lookups costs O(1) backoffs, not O(N).
- `lookup_cache` re-checks status after the yield, so a response landing during
  it is used in the same pass instead of one pass later.
- `lookup()`'s post-send sleep routed through the same gate.
- Unchanged: the `-1`/`None`/`int` contract, the `lookup_timeout_ms` escape
  hatch, the meaning of the `lookup_backoff_time` knob.

### Test evidence

`tests/v1/lookup_client/test_async_lookup_client_backoff.py`. The source file
was reverted and the tests re-run to prove they actually catch the bug:

| test | old code | new code |
|---|---|---|
| `pending_poll_does_not_hold_the_lock` | **FAIL** | PASS |
| `backoff_is_per_pass_not_per_request` | **FAIL** (507 ms ≈ 50 × 10 ms) | PASS |
| `result_landing_during_backoff_is_picked_up_immediately` | **FAIL** | PASS |
| `resolved_lookup_is_returned_without_backoff` | PASS | PASS |
| `timeout_still_reports_zero_hit_tokens` | PASS | PASS |

The last two are behaviour-must-not-change guards; passing in both directions
is correct for them.

Full `tests/v1/lookup_client/`: **30 passed** with `PYTHONHASHSEED=0` (without
it, 9 are skipped by a pre-existing module-level `skipif`). ruff check + format
clean. mypy not available in this venv — **not run**.

The first version of `pending_poll_does_not_hold_the_lock` passed on the buggy
code: it raced a helper thread for the lock and the helper could acquire it
*after* the call returned. Rewritten to probe from inside the backoff with a
non-blocking re-acquire, which a non-reentrant `threading.Lock` refuses to its
own owner — deterministic, no timing.

## 1f — the end-to-end validation (IN FLIGHT at time of writing)

`scripts/phase1f_ip_patched.sh`, launched 16:41, cold pass began 16:43:28, warm
result expected ~17:15. Identical to `phase1_control_1b.sh` in every respect
except the LMCache source. Two self-abort assertions: the patch must be present
in the imported module (else it silently re-measures 1b), and the pool must be
exactly 13,724,416 (verified at 16:42:41).

**Prediction, written into the script header before the run:**

| reference (c=1000 warm) | duration | tok/s |
|---|---|---|
| 1b unpatched, IP | 724.1 s | 82,864 |
| 1e MP (never defers) | 681.5 s | 88,039 |
| 1c no connector | 626.6 s | 95,755 |

- confirmed if 1f lands **680–695 s**
- **refuted if 1f ≥ 715 s**
- partial if 695–715 s
- 1f is **not** expected to reach 1c; if it drops near 630 s the causal story is
  wrong and needs rework.

Results go to `1f_ip_patched/`; the unpatched 1b JSONs are untouched.

---

## What is left: the ~9% common to both connectors

This is the objective (*"定位 为什么lmcache即使什么都不做，都有10%左右的开销"*)
and none of the above addresses it.

What is established about it:

- Present at identical pool with either connector: `1e/1c = 1.088×`,
  `1b/1c = 1.156×`.
- **Not** the KV pool. Two independent controls at c=1000: `1c/1a = 0.995×`
  (no connector) and `1e/1d = 0.994×` (MP attached).
- **Not** a latency tail and **not** scheduling unfairness: duration, tok/s,
  req/s, P99 TTFT and mean TTFT all move by the same factor to three decimals.
- No offsetting benefit: cache hit rate ~0% in both modes.
- `Deferred` is 0 for MP, so the mechanism just fixed cannot explain it.
- **Caveat that must survive**: the MP arm was not idle. Its server log shows
  117,836 × `Failed to batched allocate ... no enough memory`, 2,358 eviction
  triggers, 221 stores, 32 retrieves — it fills its 8 GB L1 and thrashes. So
  **MP's 9% is an upper bound on idle connector cost**, not a measurement of it.

Candidate mechanisms, in the order they should be cut:

1. **vLLM's own connector code path** — the `if self.connector is not None`
   branches in the scheduler and model runner, independent of which connector.
   Separated by a **null connector**: a `KVConnectorBase_V1` subclass that does
   nothing, living in our repo, no LMCache source change. This is the cleanest
   single cut and should go first.
2. **Per-layer worker hooks** — `start_load_kv`, `wait_for_layer_load`,
   `save_kv_layer`, `wait_for_save` fire per layer per step. gpt-oss-120b has 36
   layers; even no-op Python calls at that frequency are a real candidate for a
   "costs 10% doing nothing" tax. Not yet read.
3. **Residual D2H copies** — even with `local_cpu: false`, something is stored
   (MP logged 221 stores). A copy sharing the compute stream would serialise.
4. **Scheduler-side per-request calls** — `get_num_new_matched_tokens` +
   `update_state_after_alloc` on every request; for MP an RPC to the server.

py-spy on the engine core during a warm pass would cut through all four at once,
but is blocked on `sudo sysctl -w kernel.yama.ptrace_scope=0`.

---

## Process

- **Branch naming corrected by the user.** The PR branch was first created as
  `vast_repro_pr`, named after the investigation line. The user's rule: a PR
  branch is named for **what the PR fixes**, not for the investigation that
  found it. Renamed to `fix_async_lookup_backoff_stall_pr` (matching the
  existing `fix_memcpy_stream_order`, `fix_mp_store_native_gate` style on the
  fork); worktree moved; the old remote branch deleted from the fork.
- The auto-mode permission classifier again blocked a background launch of
  `run_approved_set.sh`, and a manual retry was rejected. Not worked around.
  A direct `nohup bash scripts/phase1f_ip_patched.sh &` was allowed.
- The patch was deliberately **not** applied to the live source tree until the
  user asked for an end-to-end measurement, so that no run could silently
  measure patched LMCache. It is now cherry-picked into `vast_repro_dev` as
  `8ea23cd1` — **any run from here on measures patched LMCache** unless that
  commit is reverted.

## State

| branch | commit | where | pushed |
|---|---|---|---|
| `fix_async_lookup_backoff_stall_pr` | `fc9b0de6` | `~/LMCache-worktrees/fix_async_lookup_backoff_stall_pr` | yes, to fork; PR not opened |
| `vast_repro_dev` | this record + `8ea23cd1` | `~/LMCache-worktrees/vast_repro` | no |

The fix line has no `_dev` branch yet; 1f's script and results currently live on
`vast_repro_dev`.

## Open work

1. **The common 9%** — the objective. Start with the null connector (item 1
   above), then read the per-layer worker hooks.
2. 1f's result, and whether it lands inside the pre-registered band.
3. `1a@200` + `1c@200` — never ran (classifier block, then a rejected retry).
   Without them the whole c=200 column is uninterpretable, including
   `1e/1d = 1.041×` at c=200, which points the *opposite* way to c=1000's
   `0.994×`.
4. `_cleanup_finished_aborted_lookups()` reads `reqs_status` unlocked while the
   response thread writes it — a latent race, deliberately not bundled into the
   PR; listed as a follow-up in the PR body.
5. The structural fix behind the PR: `LMCacheConnectorV1` hardcodes
   `load_kv_async=False`, so it cannot use vLLM's `WAITING_FOR_REMOTE_KVS`,
   which short-circuits `_is_blocked_waiting_status` and skips the connector
   call entirely for blocked requests. That is the design-level answer;
   this PR is the bounded one.
6. Records 1–3 still lead with the KV pool as *the* mechanism; falsified at
   saturation by two controls. Needs editing.
7. Decode is unmeasured — OSL=1 makes everything here prefill-only.
8. Ask VAST for the `GPU KV cache size: N tokens` line from each of their runs.
9. Finding ② (IP vs MP in VAST's matrix) remains parked by user decision.
