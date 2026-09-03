# Loss #2 is a stream drain, not a copy -- and the fix it implied is worth 0.089 ms

Record 2 found the line and got the mechanism wrong, and the wrong mechanism
had already cost two reverted patch attempts. One arm settles it.

## The probe

`slot_mapping.to(self.device)` is a PAGEABLE host-to-device copy, so it is both
host-blocking and stream-ordered: the calling thread does not return until the
current stream has drained. At that moment the stream holds an entire
forward pass. So the 33.7 ms cProfile charged to `{method 'to'}` is the sum of

    (a) draining the forward-pass kernels already queued, and
    (b) the 480 KB DMA, which should be ~50 us.

Synchronising the current stream immediately BEFORE the copy separates them and
**cannot perturb what it measures**: the pageable copy already waits for exactly
that stream, so the sync is a no-op in every sense except accounting. That is
the whole design of the probe -- `LMC_SLOTPROBE=1`, three lines.

## The split

TP=4 lane, GPUs 0-3, 300 prompts x 60,000 tokens, c=300, pool pinned at 30,000
blocks (1,920,000 tokens -- the lane's POOL_REF asserted it against record 2's
runs). Arm `ipstoreprobe`, identical to record 2's. All four workers:

    SLOTPROBE calls=2200  sync_ms/call=64.47  copy_ms/call=0.089
                          store_ms/call=4.96  n_store=1930  reqs/call=1.00

| component of `wait_for_save` | ms/call | share |
|---|---|---|
| **draining the forward pass** | **64.47** | **85%** |
| the 480 KB DMA itself | **0.089** | **0.12%** |
| `lmcache_engine.store()` | 4.96 | 6.5% |
| the rest of the loop body | 6.52 | 8.5% |
| `wait_for_save` total (hook timer, 2195 calls / 167.53 s) | 76.32 | 100% |

Per-worker spread is 64.10-65.10 on sync and 0.086-0.092 on copy.

**The pre-allocated pinned buffer that LMCache's own TODO asks for, one line
above the call, is worth 0.089 ms/call.**

Record 2's "the copy costs more than the transfer it prepares -- 33.7 ms for
480 KB against 11-15 ms for 144 MB" inverts: 0.089 ms against 5.2 ms.

## The instrument did not move

    arm                       exec_wall   exec_cpu   end-to-end
    record 2 ipstoreprobe      140.47     140.25       326.0 s
    this arm, probe armed      140.42     140.19       324.4 s

Two decimals on both probe columns, 1.6 s on the client clock.

## And this time the store path is live

Every phase1 MP arm stored 0.2% of what it was asked to (record 2, "three
things that were already true"). This arm did not: **8,400 `Stored 8192 out of
total 8192` lines, zero allocation failures**, mean `offload_time` 5.23 ms for
0.1406 GB = 24.0 GB/s. So the 4.96 ms/call of `store()` is a real 141 MB D2H
running at pinned speed, not a fast failure. Nothing here rests on an inert
path.

## Why the fix record 2 implied nets zero

The pinned-staging fix (pinned host buffer, `non_blocking=True`, plus the
`store_stream.wait_stream` barrier the first attempt was missing) removes
0.089 ms of copy and relocates the 64.47 ms of drain. It cannot do otherwise:

1. `VLLMPagedMemGPUConnectorV2.from_gpu` (gpu_connectors.py:374) runs
   `multi_layer_kv_transfer` D2H on its own `store_stream` with **no**
   `wait_stream(current_stream)`. V3's `from_gpu` (line 599) has the same gap;
   only the layerwise connectors take the barrier (line 1043).
2. That D2H reads the KV this step's forward pass has just written, so the
   barrier is not optional. **Today the pageable copy's device drain is the
   only thing supplying that ordering** -- which is exactly why attempt 2's
   reusable pinned buffer died with a CUDA illegal access after a few hundred
   stores.
3. Add the barrier, and `from_gpu`'s `store_stream.synchronize()`
   (gpu_connectors.py:410, fired for every host-resident memory object) now
   waits for the forward pass. The block moves from `.to()` into `store()`.

The bytes were never the problem. The **host block** is.

## What the block actually costs

`wait_for_save` is called from the `finally` of vLLM's
`_get_kv_connector_output` context manager
(vllm/v1/worker/kv_connector_model_runner_mixin.py:107), which wraps
`_model_forward` **and nothing else** -- it closes before `compute_logits`,
before sampling, at the point gpu_model_runner.py comments "Now the batch has
been launched we can wait for corrections ... without breaking async
scheduling". It is precisely where vLLM intends to run ahead.

    arm     loop     exec     cpu    end-to-end
    none   131.78    81.82   81.82     296.9 s
    ip     143.28   140.40  140.19     324.4 s

`exec` grows by +58.6 ms/step but the loop by +11.5 and the client clock by
**+27.5 s = +12.5 ms/step (+9.3%)**. The other ~52 ms is time the CPU would
have spent waiting for the GPU anyway, merely relocated into `execute_model`;
`exec_wall == exec_cpu` throughout because CUDA's default sync busy-waits.

**So loss #2's fixable prize is +12.5 ms/step, and the mechanism is destroyed
CPU run-ahead, not a slow copy.** Any fix has to leave the host unblocked:

  (i)   `store_stream.wait_stream(current_stream)` in `from_gpu` -- required for
        correctness the moment anything upstream stops draining the device.
  (ii)  pinned staging + `non_blocking=True` for `slot_mapping`, with an
        event-gated buffer pool so a buffer is not overwritten while its DMA is
        in flight (attempt 2 reused one buffer and corrupted it).
  (iii) replace `store_stream.synchronize()` with an event recorded on
        `store_stream` and carried by the memory object, waited on by whoever
        first reads those host bytes. **This is the one that recovers the
        12.5 ms**; (i) and (ii) only make it safe.

The source itself flags (iii): "NOTE: for better performance, we may not want
to sync for every memory object."

## Method note

This is the second time in this investigation that a per-call cProfile number
was read as the cost of the work in the frame rather than the cost of what the
frame waits for. The first was `poll.py:80(poll)` and the msgspec encode's
caller table (record 7). The general form: **a frame that blocks is charged for
everything it blocks on, and cProfile has no column that tells you which.** A
sync inserted immediately before the suspect call, where it is semantically a
no-op, separates the two for three lines of diff.

## Reproduce

    bash records/2026/09/03/harness/scripts/chain26.sh      # the split
    LMC_SLOTPROBE=1 ...                                     # env gate, off by default

The probe lives on `vast_repro_dev` only. It must not reach a PR branch.
