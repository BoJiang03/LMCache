# temp_ctx — handover for a new node (written 2026-09-03)

Everything a cold session needs to resume the VAST/LMCache performance
investigation on a different machine. Read this first, then
`records/2026/09/03/11_loss_2_handed_off_and_the_residual_of_loss_1.md`.

---

## 0. Branches

Two branches on the fork `https://github.com/BoJiang03/LMCache.git`, one base:

| branch | what it is |
|---|---|
| `fix_mp_store_key_prefix_resend_pr` @ `88ccf635` | **the PR.** 9 files, +300/-14. No `records/`, no diagnostics, no scratch. Verified: `SLOTPROBE` does not appear. **Do not open the PR** — a title/body draft is in §6. |
| `fix_mp_store_key_prefix_resend_dev` | **everything else.** `records/` (all session notes + results), the harness snapshot, the SLOTPROBE diagnostic, the same fix as commit `50c6cf7f` (byte-identical to `88ccf635` across all 5 fix files — verified by `git diff --numstat`). |

The dev branch is **28 ahead / 11 behind `origin/dev`**. It has never been
rebased onto upstream `dev`; do that only if you need upstream changes, and
never rebase the PR branch without re-verifying the diff stat above.

## 1. The task, as given

> 自己定位我们找到的两个性能损失。如果找到了就修。修完了就接着去复现 vast 里面的第二个问题。

(a) locate the two performance losses, (b) fix them, (c) reproduce VAST's
second reported problem.

**Status:** (a) done for both. (b) loss 1 half fixed and shipped; loss 2
diagnosed then **dropped by owner's decision** — "I guess I am gonna 放弃 loss2
because it is in in process path which is not important. so lets focus on loss
1 (mp mode now)". (c) **not started.**

Owner asked that the work continue **in English**.

## 2. Loss 1 — the MP connector (the live work)

`LMCacheMPConnector`, out-of-process LMCache server, loaded through
`kv_connector_module_path` + `PYTHONPATH` — the same path VAST uses.

### Root cause

The STORE key was built from the **whole grown prompt prefix**
(`token_ids[0:end]`) even though the chunk being stored was only the last 8192
tokens. At ISL=60000 that is a msgpack array of 60,000 ints, ~255 KB,
re-encoded and re-sent on **every store**.

It costs ~8 ms/step while the Python itself is ~1.1 ms because:

- the encode runs on the `mq-client-shared-loop` daemon thread **inside the
  worker process**, and `msgspec._core.msgpack_encode` **does not release the
  GIL**, so it steals from the model-execution thread in the same process;
- at TP=8 all 8 ranks do it independently and the all-reduce takes the **max**
  of 8 ranks' jitter.

Amplification is measured, not inferred: same connector costs **+1.10 ms/step
(0.83%) at TP=4** vs **+7.97 (9.5%) at TP=8**.

### Numbers (TP=8, 1000 prompts, ISL=60000, c=1000, `scripts/probe_report.py`)

    arm            loop     exec      cpu   blocked   Δloop vs none
    tp8_none      83.94    68.76    68.75      0.00        --
    tp8_nostore   85.71    68.22    68.19      0.04      +1.77
    tp8_mpfix     87.72    72.30    71.74      0.56      +3.78
    tp8_tinykey   88.53    71.19    70.63      0.56      +4.59
    tp8_mp        91.90    73.89    72.04      1.85      +7.97

Reproduce with `.venv/bin/python scripts/probe_report.py results/phase1_v2`.

**Recovered: 4.18 ms/step = 52% of loss 1** (~30 s off a 625 s job).
`88ccf635` ships the **delta with an absolute offset**, keeping the prefix hash
chain so the key still identifies the same thing. ~1.1 ms/step of Python
removed returned 4.18 → amplification **3.8×**, demonstrated by intervention.

### The remaining +3.78, and the one structural fact that is new

`tp8_nostore` adds **+1.77 to `loop` while `exec` goes DOWN 0.54**. That cost
is **not inside `execute_model` at all** — ~+2.31 ms/step of it is worker idle
*between* steps, i.e. the next step's command arriving late. That points at the
**scheduler / EngineCore process, which no profile in this investigation has
ever covered** — every cProfile arm so far was worker-side. This is the single
best lead for the remainder and it was sitting in the table.

Split of the remaining +3.78:

- **+1.77 (47%)** — outside `execute_model`: scheduler-side LOOKUP and
  connector metadata construction. **Never profiled. No live hypothesis.**
- **+2.01 (53%)** — inside the surviving store submission: CUDA event IPC
  export, MQ round trip, futures. Only ~0.20 ms/step of attributable Python →
  ~0.7 at 3.8×. **~1.3 ms/step does not reconcile.**

⚠️ **Known defect in that split:** `tp8_nostore` was measured on the
**unfixed** build, whose metadata still broadcast the whole prefix. So
`+2.01 = mpfix − nostore` subtracts across two builds and `+1.77` is an
old-build number. **Re-baseline before quoting either as final.**

### Refuted hypotheses (do not re-propose)

Seven mechanisms have been proposed from reading source and seven refuted. The
most recent, killed before it cost a run:

- **`block_ids` still O(request).** False. `lmcache_mp_metadata.py:276` slices
  STORE `block_ids` to the stored range (128 ids, not 938); the RETRIEVE op at
  `:350` is gated on `num_lmcache_hit_tokens > start_token_idx`, which is 0 on
  this cold all-unique workload, so it is never emitted.

**Do not offer an eighth guess. Measure.**

### Next action — the chain (proposed, NOT approved, and was GPU-blocked)

Three arms, TP=8, **300 prompts** (profiles do not need the full workload; the
timings are frozen in records 6 and 7), **~25 min**:

1. `none` — floor
2. `nostore` **on the fixed build** — fixes the cross-build defect, gives
   `+1.77` an honest number
3. `mpfix` — current state

with cProfile on **both the worker and the EngineCore/scheduler process**.
Profiling the scheduler is the change from every prior run and the only way
`+1.77` gets decomposed.

Then, contingent on what it shows:

- **Raw int32 buffer instead of a msgpack array of ints.** Even the delta is
  still 8192 msgpack-encoded ints; as a raw buffer the encode is a memcpy.
  Local, low risk, independently measurable, and at 3.8× a small win multiplies.
- **Get the encode off the GIL** — release it around the encode, or move
  submission so it cannot contend with the forward pass. Highest ceiling,
  largest change: it attacks the amplifier, not the payload.
- **Cache the CUDA event IPC handle export** if the profile charges it.

## 3. Loss 2 — the IP connector (diagnosed, bounded, deliberately NOT fixed)

`LMCacheConnectorV1`, in-process. Owner dropped it as unimportant. Keep the
findings; do not spend GPU on it without a new instruction.

- `wait_for_save` costs ~76 ms/step at TP=4. **85% of it is stream drain**
  (64.47 ms), not the copy: the 480 KB `slot_mapping` H2D is **0.089 ms**.
  Pageable H2D is both host-blocking *and* stream-ordered, so the copy does not
  return until the current stream drains — a cProfile frame that blocks is
  charged for everything it blocks on.
- **Record 2's "33.7 ms pageable copy" was 99.9% drain.** LMCache's own TODO
  for a pre-allocated pinned buffer is worth 0.089 ms/call.
- `use_layerwise: true` **spreads the block, it does not remove it**:
  `wait_for_save` 76.15 → 2.43 ms/step but `save_kv_layer` 0.27 → 67.70; block
  total 76.42 → 70.13; end-to-end 324.4 s → 320.4 s against `none` 296.9 s.
  Per-chunk throughput collapsed 24.03 → 0.809 GB/s.
- **Real loss is only +11–12 ms/step (+9.3% at TP=4);** the rest is absorbed
  because the GPU is genuinely busy. CUDA default sync busy-waits, hence
  `exec_wall == exec_cpu` even while blocked on the device.
- **Unimplemented fix:** event-record instead of `store_stream.synchronize()`;
  defer `batched_put` by one store; **add the missing
  `store_stream.wait_stream(current_stream)`**.

### ⚠️ Latent correctness bug found on the way — independent of performance

`VLLMPagedMemGPUConnectorV2.from_gpu` (`lmcache/v1/gpu_connector/gpu_connectors.py:374`)
runs the D2H inside `with torch.cuda.stream(self.store_stream):` with **no
`wait_stream(current_stream)`**. V3's `from_gpu` (`:599`) has the same gap.
Only `VLLMPagedMemLayerwiseGPUConnector` (`:1043`) takes the barrier.

Today only the pageable copy's incidental device drain orders that D2H against
the forward pass writing the KV it reads. **This is why an earlier pinned-buffer
attempt died with a CUDA illegal access** — removing the drain removed the only
thing providing ordering. Worth reporting upstream on its own merits.

## 4. Measurement conventions — read before trusting any number

1. **`median Avg prompt throughput` is a CLASSIFIER, not a measurement.** It has
   two quantised attractors (~95,99x and ~89,99x) and the median reports which
   one an arm sits on. It said `tinykey` and `mp` were identical (91.03 vs
   91.04) when they were 27.4 s apart end-to-end. **Every `ms/step =
   8192000 / median tok_per_s` figure in records 4 and 5 is this classifier.**
2. **Trust two instruments only:** the step probe differenced
   (`scripts/probe_report.py` — each STEPPROBE line is cumulative, so the
   steady state is the *difference* between consecutive lines) and the
   end-to-end client duration. They reconcile to ~1 s across four arms.
3. **Step counts must match** before comparing `ms/step`.
4. **The no-op sync probe:** inserting
   `torch.cuda.current_stream().synchronize()` immediately before a pageable
   copy changes no semantics (the copy already waits for exactly that), so it
   separates "drain" from "DMA" without perturbing the run.

## 5. Reproducing on the new node

    REPRO_ROOT=/home/bo/vast_profiling_problem      # harness (NOT a git repo)
    LMCACHE_SRC=/home/bo/LMCache-worktrees/vast_repro
    MODEL=/raid/rui/gpt-oss-120b                    # 36 layers, mxfp4, block 64
    VENV=$REPRO_ROOT/.venv ; CUDA_HOME=/usr/local/cuda-13.0
    PORT=8765  MP_PORT=5765

**The harness is not under git.** A complete snapshot is committed at
`records/harness_snapshot/` on the dev branch:

    scripts/  configs/  timedconn/  nullconn/  sitecustomize.py
    analysis/  vast_LMCache_collab.pdf         # VAST's problem statement
    chains/                                    # chain2..chain27 + apply*.py,
                                               # these lived only in the job
                                               # tmp dir and are otherwise lost

Restore with `rsync -a records/harness_snapshot/ $REPRO_ROOT/` (then fix the
absolute paths in `scripts/env.sh`), and `pip install -e $LMCACHE_SRC`.

Run one arm:

    env GPUS=0,1,2,3,4,5,6,7 TP=8 PROBE=1 NPROMPTS=1000 CONC=1000 \
        LANE_OUT=results/phase1_v2 ARM=mp TAG=tp8_mp bash scripts/lane.sh

Arms available in `lane.sh`: `none null mp timed nostore tinykey nolookup
nowait storeprobe ip timedip ipstoreprobe`. `lane.sh` gates on free VRAM,
pre-flights the connector class, and asserts the pool size, so a bad launch
fails in seconds rather than after 10 minutes.

`LMC_SLOTPROBE=1` arms the split H2D probe (dev branch only —
`vllm_v1_adapter.py`; **it must never reach a PR branch**).
`LMC_TIMER_DIR` + `scripts/timer_report.py` give per-hook timers
(**takes one timer file, not the `timers/` directory**).

**`records/` is excluded** via `.git/info/exclude:19`, so committing it needs
`git add -f`.

## 6. PR draft (already given to the owner; the PR must NOT be opened)

**Title:** `[MP] Stop resending the whole prompt prefix on every store`

**Body, in short:** the MP store key was derived from the entire grown prompt
prefix rather than the chunk being stored, so every store re-encoded and
re-sent O(prompt) tokens. Ship the delta with an absolute offset and keep the
prefix hash chain. Measured at TP=8 / ISL=60000: **91.90 → 87.72 ms/step,
−4.18 (52% of the connector's overhead)**, ~30 s off a 625 s job. Includes
`tests/v1/multiprocess/test_session_token_delta.py`.

## 7. Operating rules carried over

- **Experiments run only on explicit user command.** Propose with a time cost,
  wait for approval, never auto-chain. (memory:
  `experiments-run-only-on-user-command`)
- **PR branches are named for the fix,** not the investigation. (memory:
  `pr-branch-named-for-the-fix`)
- Push to the **fork**, never upstream. **Never open the PR** — hand over a
  title and body draft.
- **Shared box:** touch only processes this session started; never
  `pkill -f`. A `pgrep` pattern must not match its own caller — writing a
  heredoc and launching it in the *same* shell command puts the script text on
  the launcher's command line, and `chain27` hung forever on exactly that.
- `records/` lives on the `_dev` branch and nowhere else.

## 8. Box state when this was written (old node — will not apply on the new one)

GPUs 0–3 free (GPU 0 held 3 idle foreign CUDA contexts, ~2.4 GB, 0%).
GPUs 4–7 taken by neighbour `shihao` from 14:26 (4× `ray::CacheGRPOWorker` +
4× `VLLM::Worker_TP`, 52–82 GB each, 100% util). **TP=8 was blocked, which is
why the profile chain never ran** — and TP=4 cannot substitute, because the
TP-degree amplification *is* the phenomenon (0.83% vs 9.5%).

No processes of this session were left running.
