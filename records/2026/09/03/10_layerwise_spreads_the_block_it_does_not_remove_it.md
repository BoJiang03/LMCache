# Layerwise spreads loss #2's block, it does not remove it

Record 9 established that IP's `wait_for_save` is 85% stream drain and 0.12%
copy, and that the fix has to leave the host unblocked. Layerwise is LMCache's
own overlap mechanism and costs one config line, so it was worth one arm
before touching the transfer path.

It works, it is not the answer, and the way it fails is more useful than the
result.

## Validity

`configs/ip_layerwise.yaml` differs from `configs/lmcache_gpu_only.yaml` by
exactly one line (`use_layerwise: true`); the diff is in the harness. Same TP=4
lane, same 300 x 60,000 / c=300, same pinned 30,000-block pool (POOL_REF
asserted). The adapter's layerwise copy site carries its own probe, so the log
says which branch ran: **252 `SLOTPROBE-LAYERWISE` lines** and
`use_layerwise': True`. Both arms stored the same work: **8,400
`Stored ... 0.1406 GB` chunks each**.

## The result

Per worker, ms/step over 2,200 steps.

| | non-layerwise | layerwise |
|---|---|---|
| `wait_for_save` | 76.15 | **2.43** |
| `save_kv_layer` | 0.27 | **67.70** |
| connector block, total | 76.42 | 70.13 |
| `exec_wall` / `exec_cpu` | 140.42 / 140.19 | 139.57 / 139.35 |
| end-to-end | 324.4 s | **320.4 s** |
| (`none`) | | 296.9 s |

And the probe, which is what makes the mechanism legible:

    non-layerwise   calls= 2,200   sync=64.47 ms/call   copy=0.089
    layerwise       calls=12,600   sync= 2.15 ms/call   copy=0.025

**The drain per call collapsed 30x. Per worker across the run the host block
went from 141.8 s to 27.1 s -- 114.8 s removed -- and end-to-end bought 4.0 s
of the 27.5 s gap (15%).**

It did not disappear, it moved: `save_kv_layer` picked up 67.7 ms/step, having
been 0.27. The single full-forward drain became 36 per-layer synchronisations,
and the per-chunk transfer efficiency collapsed with the granularity:

    non-layerwise   cost   5.85 ms   0.1406 GB   24.03 GB/s
    layerwise       cost 173.79 ms   0.1406 GB    0.809 GB/s

Same bytes, 30x the wall time, spread thinner. Net 4 s.

## What the pair establishes

Two very different placements of the same work -- one big block at the end of
the forward, or 36 small ones interleaved with it -- cost within 6 ms/step of
each other and within 4 s end-to-end. Combined with record 9's copy split, the
shape of loss #2 is now:

> **The worker thread spends ~70 ms/step spinning inside a CUDA sync waiting
> for the GPU, and ~11-12 ms/step of that is real loss; ~85% is time the CPU
> would have spent waiting anyway. Neither the copy nor the layer granularity
> moves it. Only not blocking at all can.**

`exec_wall == exec_cpu` in both arms, to two decimals, because CUDA's default
sync busy-waits.

Pre-registration: three readings were written down before the run
(`dur ~300 / sync collapses`, `dur ~324 / sync still ~60`, `dur >> 324`). None
matched. What happened was the sync collapsing *and* the duration not moving --
which is the case that separates "the block is the cost" from "the block is
mostly absorbed", and neither branch had been written to expect it. Recording
the reading in advance still did its job: it made the surprise visible instead
of letting it be narrated away afterwards.

## Revised fix, and why it is smaller than record 9 said

Record 9 asked for three changes and called (iii) a transfer-path rewrite. With
this arm's data it can be a localised deferral:

  1. `from_gpu`: replace `store_stream.synchronize()` (gpu_connectors.py:410,
     V3's at 645) with `event.record(store_stream)`, and return.
  2. `cache_engine.store`: hold `(keys, memory_objs, event)` as pending; on the
     NEXT store, wait that event before `storage_manager.batched_put`.
  3. `from_gpu`: add `store_stream.wait_stream(current_stream)`, which V2 and
     V3 both lack. Required for correctness the moment nothing upstream drains
     the device -- and the reason attempt 2 died with a CUDA illegal access.

The cost is that a cache write lands one step later, which for a cache is
nothing. The target is the 11-12 ms/step. At TP=4 that is +8-9%; the number
that matters is TP=8, where phase1 measured IP at 97.5 against `none`'s 85.34
ms/step (+14.2%) -- **the largest single loss in this investigation, larger
than loss #1 was before the delta fix.**

## Harness defect: the self-matching pgrep, second occurrence

`while pgrep -u bo -f "bash scripts/lane[.]sh"` matched its own ancestor and
the arm waited on itself forever. Record 2 logged this class already ("pgrep
patterns must not match their own caller"), but it recurred in a new shape:
the script was written with a heredoc and launched **in the same shell
invocation**, so the launching process's command line contained the script
text, which contains the pattern.

Two defences now in `chain27.sh`: launch by path so the command line is short,
and skip every ancestor of `$$` when scanning. Nothing leaked -- `pgrep` was
empty before and after the kill, and no GPU process survived.
