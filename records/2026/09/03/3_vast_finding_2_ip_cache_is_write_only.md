# VAST finding #2 reproduces: IP's cache is write-only

2026-09-03.  Companion to record 2, which covers the two per-step losses.

## What VAST reported

PDF page 2, "Performance comparison: LMCache-MP vs. LMCache-IP (performance
degradation observed). Tested on AMD GPUs."  Their chart is ISL 120k, TP8,
OSL=1, warm passes, and shows LMCache MP w/ VAST at ~250-290k tok/s against
LMCache IP w/ VAST at ~400-500k, with mean TTFT 100-140 s against 40-90 s.
Without an L2, MP collapses to ~30k and IP peaks near 600k at c=100-200 before
falling off a KV-cache cliff at c>=300.

That is the RETRIEVE path.  Every phase1 arm was cold, prefill-only, no hits
and no L2, so none of them could have seen it.

## What was run here

TP=4 on GPUs 0-3, 100 prompts x 60,000 tokens, cold pass then warm pass over
the SAME prompts (same seed), pool pinned at 1,920,000 tokens in both arms,
~500 GB of cache on each side (MP `--l1-size-gb 500`; IP `local_cpu: true,
max_local_cpu_size: 120.0` which is per rank, so 480 GB at TP=4).

**vLLM prefix caching is OFF.**  It has to be: vLLM's own paged pool here is
8 x 117.8 GiB of KV, larger than any LMCache tier this box can host, so with
APC on the warm pass is served entirely by vLLM and never reaches LMCache at
all.  This is a deviation from VAST's flags and it is the one caveat below.

## The result

| arm | cold | warm | speedup |
|---|---|---|---|
| MP | 105.6 s / 56,794 tok/s | **27.7 s / 216,784 tok/s** | **3.8x** |
| IP | 112.6 s / 53,270 tok/s | **109.9 s / 54,620 tok/s** | **1.03x** |

On the warm path MP is **4.0x faster than IP**.  So the MP-vs-IP degradation
reproduces, at a different sign from their chart -- they had IP ahead of MP
with a VAST L2 attached, on AMD, at ISL 120k; here, with no L2 and on NVIDIA,
IP is the one that loses.

## The mechanism is not a slow retrieve.  There is no retrieve.

    all 200 requests (100 cold + 100 warm) logged "LMCache hit tokens: 0",
    under APC off AND APC on
    the step probe still has IP at exec 140.02 / cpu 139.80 ms/step in the warm
    arm -- the same ~74 ms/step of wait_for_save it pays everywhere

IP pays the full store cost on every step and then never finds any of it.  Its
warm pass is within 3% of its own cold pass.  MP, on the same prompts and the
same pool, stored 2,800 chunks and got 3.8x.

The store cost IP is paying for nothing is the pageable `slot_mapping.to()`
profiled in record 2, at 33.7 ms per call.

## The caveat is closed: it is not an artifact of disabling APC

Turning APC off is what makes a hit attributable to LMCache, but it could in
principle have been what broke IP's retrieve rather than what exposed it.  Arm
`warm_ip_apc` reran with prefix caching ON -- VAST's actual flag:

| IP arm | cold | warm | speedup |
|---|---|---|---|
| APC off | 112.6 s / 53,270 | 109.9 s / 54,620 | 1.03x |
| **APC on** | **112.5 s / 53,315** | **111.4 s / 53,859** | **1.01x** |

Identical.  And with APC on, all 200 requests still logged
`LMCache hit tokens: 0`, while vLLM's own prefix cache managed a 4.0% hit rate
-- so neither tier served the warm pass.  The write-only behaviour is real and
independent of the flag.

## Also worth flagging to VAST

Their MP config is `--l1-size-gb 1600 --eviction-policy noop`.  With `noop`,
once L1 is full every subsequent store fails to allocate and the cache stops
taking new entries for the rest of the run -- silently, at WARNING level.  On
this box with 8 GB that happened after 113 chunks; at 1600 GB it happens later
but it still happens, and their w/o-L2 MP curve collapsing to ~30k is what a
full noop cache would look like.  `--eviction-policy LRU` is the arm `bigl1`
ran here, and it kept every store succeeding for the whole run.
