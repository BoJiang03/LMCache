# Loss #2 handed off; and what is actually left of loss #1

Two decisions and one dead lead. No GPU time was spent on this record -- the
box's second half went to a neighbour at 14:26 and TP=8 is blocked.

## Loss #2 is stopped here, deliberately

Owner's call: it lives in the in-process connector path, which is not the
deployment that matters. It is **diagnosed and bounded, not fixed**, and
records 9 and 10 leave it in a state someone can pick up cold:

* The mechanism is a per-step host block on the worker's model-execution
  thread, ~70 ms/step of CUDA busy-wait, of which ~11-12 ms/step is real loss
  (+9.3% at TP=4; phase1 measured +14.2% at TP=8).
* It is **not** the pageable `slot_mapping` copy (0.089 ms/call against 64.47
  ms of stream drain) and **not** layer granularity (`use_layerwise` moved the
  block from `wait_for_save` into `save_kv_layer` and bought 4.0 s of a 27.5 s
  gap).
* The fix, unimplemented and unverified: event-record instead of
  `store_stream.synchronize()` in `from_gpu`, defer `batched_put` by one store,
  and add the `store_stream.wait_stream(current_stream)` that V2 and V3 both
  lack. That missing barrier is a real latent bug independent of performance --
  today only the pageable copy's device drain orders the D2H against the
  forward pass that writes the KV it reads.
* The `LMC_SLOTPROBE` diagnostic stays on `vast_repro_dev`, env-gated, off by
  default. It must not reach a PR branch.

## What is left of loss #1

The delta fix recovered 4.18 of 7.97 ms/step. The residual, at TP=8 / c=1000 /
1000 prompts / ISL=60,000, by step probe:

| | ms/step | vs `none` |
|---|---|---|
| `tp8_none` | 83.94 | -- |
| `tp8_nostore` | 85.71 | +1.77 |
| `tp8_mpfix` | 87.72 | +3.78 |
| `tp8_mp` (before the fix) | 91.90 | +7.97 |

    +1.77   none -> nostore    47% of the residual   NEVER PROFILED
    +2.01   nostore -> mpfix   53%                   ~0.20 ms/step of attributable
                                                     Python -> ~0.7 at 3.6x;
                                                     ~1.3 unexplained

## Defect in that split: it crosses two builds

`tp8_nostore` was measured on the **unfixed** build. `NoStoreMPConnector`
patches only the worker's `batched_submit_store_requests`; the scheduler still
runs `build_connector_meta`, and on the old build that metadata carried the
whole prefix `token_ids`, which vLLM pickles and broadcasts to all 8 workers
every step. Record 7 already showed this is not free -- the shipped fix beat
the `tinykey` upper bound precisely because the broadcast metadata shrank too.

So `+2.01 = mpfix - nostore` subtracts across two different builds, and `+1.77`
is an old-build number. **The next run must re-baseline `nostore` on the fixed
build before either figure is quoted again.**

## A lead, raised and killed in the same hour

Proposed: the connector metadata still carries an O(request length) payload --
`block_ids`, 938 ids for a 60,000-token request at block_size 64 -- pickled and
broadcast to 8 workers every step, which would be the same disease the delta
fix cured in the store key. Supporting number: `_pickle.loads` costs 1.25
ms/step in **both** `mp` and `nostore` (chain23), so it is not the difference
between those two and is therefore unconstrained against `none`.

Killed by reading the code it accuses:

* `lmcache_mp_metadata.py:276` -- the STORE op's `block_ids` goes through
  `slice_block_ids_per_group(..., start_token_idx, end_token_idx)`, i.e. the
  range being stored. One 8192-token chunk is **128 block ids**, not 938.
* `lmcache_mp_metadata.py:350` -- the RETRIEVE op *does* ship
  `tracker.get_token_ids()`, the full list. But it is gated on
  `end_token_idx = tracker.num_lmcache_hit_tokens > start_token_idx`, and on a
  cold all-unique-token pass that is 0. **No RETRIEVE op is emitted at all.**

There is no O(request) payload left in the per-step metadata on this workload,
and the `_pickle.loads` equality says nothing. **I have no candidate for the
+1.77.**

Worth naming the pattern: this is the seventh mechanism proposed from reading
source in this investigation, and the seventh refuted -- this one before it
cost a run, which is the only improvement. The standing rule holds: on this
problem, reading generates hypotheses and only the profiler settles them.

## Plan, when the GPUs come back

Three arms, all cProfile'd, TP=8: `none`, `nostore` **on the fixed build**,
`mpfix`. Profiles do not need the full workload -- the timings are already
frozen in records 6 and 7 -- so 300 prompts instead of 1000: **~25 min instead
of ~55.** Goal is to decompose `+1.77` for the first time and to close the
~1.3 ms/step that does not reconcile in `+2.01`.

Levers stay unchosen until that data exists. Two blind picks have already cost
this investigation a day (the median-as-measurement error, and the single-rank
unit test that missed the truncation race).

## Box state

    GPU 0-3   free (GPU 0 holds 3 idle foreign CUDA contexts, 2.4 GB, 0%)
    GPU 4-7   106-110 GB, shihao, ray::CacheGRPOWorker + VLLM::Worker_TP*, from 14:26

TP=4 lanes still run and stay comparable to records 9 and 10. TP=8 is blocked,
and TP=4 cannot substitute: MP costs +0.8% there against +6.7% at TP=8, and the
amplification is the entire reason the loss is measurable.

No processes of mine are running; monitors from chain26 and chain27 are stopped
and one orphaned `tail` was killed.
