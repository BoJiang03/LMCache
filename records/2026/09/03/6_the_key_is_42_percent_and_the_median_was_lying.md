# The store key is 42% of loss #1 -- and the instrument that said otherwise

`tinykey` is `mp` with the STORE key built from the slice it actually stores
(`token_ids[start:end]`, `start=0`) instead of the whole grown prompt prefix.
Store path only -- `submit_retrieve_request` shares `_create_key`, so the
truncation is gated on a flag set for the duration of
`batched_submit_store_requests`.

Validity, all 8 workers, identical:

    TINYKEY batches=7000 TRUNCATED=7000 FELLBACK=0
            mean_stock_tokens=60000  mean_tiny_tokens=8192

Nothing fell back on the alignment guard, and the key really did shrink 7.3x
(255 KB -> 35 KB on the wire).

## The two instruments disagreed, and the median was the wrong one

    median Avg prompt throughput:  tinykey 91.03 ms/step   mp 91.04
    client wall clock:             tinykey 658.6 s         mp 686.0

27.4 s apart, and the "in-engine" number says they are identical. The
distribution says why:

    arm             n   median     mean      p25      p75
    tp8_none       62    95993    96407    95988   101483
    tp8_nostore    64    95990    93136    95982    95995
    tp8_tinykey    66    89997    90426    89990    95984
    tp8_mp         68    89987    87135    89336    89993

`Avg prompt throughput` has two quantised attractors here, ~95,99x and
~89,99x. The MEDIAN reports which of the two an arm sits on -- it is a
classifier, not a measurement -- and `mp` and `tinykey` sit on the same one.
The mean separates them (87,135 vs 90,426, +3.8%) and so does everything else.

**Every `ms/step = 8192000 / median tok_per_s` number in records 4 and 5 is
this classifier.** Use the step probe (`scripts/probe_report.py`, which
differences consecutive STEPPROBE lines) or the client duration.

## With the right instrument, all four arms reconcile

    arm            probe ms/step    delta   x7200   end-to-end     delta
    tp8_none            83.94         --       --     625.2 s        --
    tp8_nostore         85.71      +1.77    12.7 s    639.0 s    +13.8 s
    tp8_tinykey         88.53      +4.59    33.0 s    658.6 s    +33.4 s
    tp8_mp              91.90      +7.97    57.4 s    686.0 s    +60.8 s

Every row's `delta x 7200` matches its end-to-end delta within ~1 s. Two
independent instruments, four arms, no residue. There is no ramp/drain
discrepancy to explain -- that was an artefact of comparing the classifier
against the wall clock.

## Result: the key is 42% of the loss, and the amplification is real

`tinykey` removes **3.38 of the 7.97 ms/step (42%)**; end-to-end, 27.4 s of
60.8 s (45%).

The Python it actually removed is ~1.1 ms/step -- `tuple()` 0.248 -> 0.031,
msgpack encode 0.688 -> 0.085, zmq send proportional to a 7.3x smaller
payload. It bought 3.38. **Amplification ~3x, now demonstrated by
intervention rather than inferred from spin counts.** Removing host-side
Python from the worker's execution thread at TP=8 is worth three times its
own cost, which is what the desync-plus-lockstep picture predicted.

Decomposition of the +7.97:

    +1.77   scheduler-side LOOKUP and connector metadata  (nostore keeps it)
    +3.38   the key's size                                 (tinykey removes it)
    +2.82   the rest of the store submission: event IPC export, MQ, futures

## Three corrections to records 4 and 5

1. **"+5.70 ms/step is the steady-state tax; +9.7% is job-level; the
   difference is 19.8 s of ramp and drain."** Wrong. +5.70 was the
   classifier. The honest whole-run figure is **+7.97 ms/step, which IS the
   +9.7%**. Nothing needs to be attributed to ramp or drain.

2. **"Pool depth is irrelevant across 1.92M / 13.7M / 25.8M -- 85.3 / 91.0 at
   all three, four significant figures."** The four-figure agreement was the
   classifier reporting the same attractor. By probe: pool 1,920,000 gives
   **+7.28** ms/step, pool 13,724,416 gives **+7.97** -- same magnitude, ~9%
   apart. The conclusion (pool depth is not the driver) survives; the
   precision claim does not.

3. **"At TP=4 the in-engine steady state shows no loss at all -- 136.54
   against 136.54."** By probe, TP=4 c1000 is **+1.10 ms/step (0.83%)**
   against TP=8's +7.97 (9.5%). The TP contrast stands and is ~7x, but it is
   not 0 vs 6.7%.

## What to fix, revised

The 3.38 ms/step is reachable without changing what the key IDENTIFIES:

  (a) Ship `token_ids` as a raw int32 buffer instead of a msgpack array of
      ints. The encode becomes a memcpy: recovers the 0.25 tuple and the 0.69
      encode; the wire stays ~240 KB so the zmq cost stays. Local, and
      independently verifiable.
  (b) Ship the delta, not the prefix. The server already keeps a per-request
      token buffer (`session.set_tokens`, engine_context.py:299), so it can
      take only the new tokens. Encode and wire both collapse. Protocol
      change, and the full 3.38 is in reach.

(a) first -- low risk and it can be measured on its own. Given the ~3x
amplification, ~0.9 ms/step of Python removed should return ~2.5 ms/step.

`tinykey` itself is NOT the fix: truncating changes which key the KV lands
under, which is harmless only because every lookup misses on this cold
all-unique-token workload.

## Instrument note

Third watchdog, third miss -- but a small one. `chain24` takes a baseline of
the compute pids already on the GPUs and reports only pids that are neither in
the baseline nor attributable to this lane. `nvidia-smi` returned an empty
list at the instant the baseline was taken, so the three genuine neighbours
(1816225 rui/lmcache, 3203853 and 3204611 root/lmcache -- 522-592 MiB of idle
CUDA context each, all on GPU 0, 0% util) were not recorded as baseline, and
our own lane's lmcache server 3051328 matched no ownership pattern. No foreign
COMPUTE process appeared during the run; those three were equally present
during tp8_none / tp8_mp / tp8_nostore and chain23, so the arms are comparable.
