# Phase 1 closed out, and two things that turned out to be wrong

Continues `1_vast_lmcache_gpu_only_regression_repro.md`.  Drafts live in
`2_supports_hma_issue_and_vast_reply.md`.  Written 2026-09-02 ~14:10, while the
1a control is still running.

## Headline: finding (1) reproduces

VAST, c=1500, warm, MI355X: GPU-only **761,774 ms** vs GPU-only + LMCache
**885,142 ms** = **1.162x**.

Us, same workload on H200: **928.2s** vs **1072.1s** = **1.155x**.

Absolute values differ ~20% (different silicon); the ratio matches to within
0.6%.  The mechanism is a halved GPU KV pool on a hybrid-attention model:
gpt-oss-120b is 36 layers, exactly 18 `sliding_attention` (window 128) and 18
`full_attention` (verified from `config.json`).  With the hybrid KV cache
manager on, the sliding layers need a 128-token ring buffer instead of a
131072-token reservation; with it off, all 36 layers are provisioned for the
full sequence.  Measured pools: **25,798,626** vs **13,724,416** tokens =
**1.880x**, and that ratio is byte-for-byte reproducible across sessions.

vLLM turning the allocator off is correct, not a bug: `SupportsHMA` hands the
connector `block_ids` as one list per KV cache group, and a connector that
assumes a flat list would index the wrong blocks -- silent KV corruption.  vLLM
picks the slow-but-correct fallback and refuses to let you force it back on
(`vllm/config/vllm.py:1483-1490` raises).

The LMCache-side gap is documented in LMCache's own source.
`lmcache/integration/vllm/vllm_v1_adapter.py:180-186`:

```python
# According to the vLLM code (.../sched/scheduler.py#L943),
# only one KVCacheGroup is supported in connector for now.

# TODO: Please support multiple KVCacheGroup in connector.
# NOTE: Also, `update` method in RequestTracker should be updated accordingly.
unfolded_block_ids = new_request.block_ids[0].copy()
```

The MP path got the port that this TODO describes; the IP path did not.  The
group machinery already exists and is not MP-specific by location --
`kv_cache_groups.py` and `kv_cache_group_edits.py` sit in the same package; the
IP adapter simply does not import them.  Fixing it means threading groups
through the IP adapter's request tracking and store/retrieve paths (`block_ids`
appears ~70 times), then declaring `SupportsHMA` on `LMCacheConnectorV1` -- which
lives in **vLLM's** tree, so the two halves must land in the right order.  Adding
the mixin first would trade a 1.16x slowdown for silent data corruption.

## Wrong thing #1: I could not actually tell whether VAST used IP or MP

I asserted the page-1 chart was the in-process connector.  It is not stated
anywhere on that page -- not in the text, not in the chart legend.  The
"Configurations Used / 1. IP / 2. MP" block sits under item **2**, the MP-vs-IP
comparison.

My reasoning had been that the legend's knobs (`chunk_size`, `save_chunk_meta`,
`LMCache-fs`, sharding) are the IP YAML surface.  That does not hold: the MP
server CLI exposes `--l2-adapter JSON`, `--l2-store-policy`, `--chunk-size`, and
`save_chunk_meta` reaches `fs_connector` through `extra_config` either way.

It matters, because the two modes reach the same halved pool by different routes
and therefore need different fixes:

| | how the pool halves | fix |
|---|---|---|
| IP | vLLM disables HMA *for them*, silently | code change in LMCache |
| MP | *their own* `--disable-hybrid-kv-cache-manager` | delete one flag |

If it is MP, they can have the full 25.8M pool today: the module their config
names, `lmcache.integration.vllm.lmcache_mp_connector`, already subclasses
`SupportsHMA` (`:273`).  Only the legacy `_0201` / `_0180` modules need that
flag, and `_0201:81` is where that requirement is written down.

Also worth flagging: **the NVIDIA half of finding (1) is a different experiment.**
The blog VAST links in item 1a benchmarks Mistral Medium 3.5 128B on vLLM
**0.20.0** and 8x H100 -- not gpt-oss-120b on 0.22.1 and MI355X.  If Mistral
Medium 3.5 is not a sliding-window or Mamba hybrid, the halved-pool explanation
cannot cover that platform at all, and the constant connector overhead below is
the likelier cause there.  Neither the PDF nor the blog says.  Both are now
questions 1 and 6 in the draft.

## Wrong thing #2: half my cost numbers were cross-session comparisons

The c=300 point had 1b (connector attached, storing nothing) coming out **24%
faster** than 1c (plain vLLM, allocator forced off) on a byte-identical pool,
with bench arguments verified equal field by field.  The control re-run shows the
**original 1c was simply a bad measurement** -- taken 15 minutes after the OOM
incident, while root's k8s pods were restarting and re-claiming 200 GB each:

| c=300 warm | when | tok/s |
|---|---|---|
| 1c original | Sep 1 17:53 | 113,470 |
| 1c re-run | Sep 2 13:16 | **143,628** |
| 1b | Sep 2 11:04 | 148,395 |

Then the 1a control showed 1a had drifted too, in the **opposite** direction:

| config, c=300 warm | Sep 1 | Sep 2 | drift |
|---|---|---|---|
| 1a plain | 211,141 | 182,298 | 0.86x |
| 1c allocator off | 113,470 | 143,628 | 1.27x |

One config got 14% slower between sessions, the other 27% faster.  That is not a
"busy machine" story -- it is two-directional noise, and it means a **single run
at the knee carries roughly +-20%**.  The 1.86x I had published for the
pool-halving cost was inflated by about 50% purely by pairing a Sep 1 number
with a Sep 1 number from a different hour.

Same-session numbers so far (all Sep 2):

| | c=300 | c=600 |
|---|---|---|
| pool-halving (1a_rerun / 1c_rerun), warm | **1.27x** | pending |
| pool-halving, cold | 0.99x | **1.27x** |
| connector (1c_rerun / 1b), warm | 0.97x | 1.28x |

I told the user mid-session that the cold pass showed no pool cost at all and
that the penalty was therefore purely a prefix-reuse effect.  **The c=600 cold
point (1.27x) contradicts that** -- the c=300 cold result does not generalise.
A plausible reading is that at c=600 the pool also limits how many 60k prefills
can be in flight at once, independent of reuse, but the passes are not clean
(each server session runs c=300 before c=600, so the "cold" pass at c=600 starts
with c=300's blocks resident).  Do not quote a cold/warm decomposition yet.

### What is and is not safe to quote

Safe -- verified, session-independent:
- 1.880x pool ratio, byte-identical across sessions (25,798,626 / 13,724,416)
- The mechanism and every code citation
- 18 sliding + 18 full layers, from the model's own `config.json`

Not safe yet:
- Any single throughput cost figure at the knee (c=300-600)
- The 14.2% "saturation tax": it was 1b (Sep 2) vs 1a/1c (Sep 1).  A 1c control
  at c=1000 is queued to pair same-session with 1b's c=1000.
- The 1.155x headline itself pairs 1a (Sep 1) with 1b (Sep 2).  The saturation
  plateau looks far more stable than the knee -- 1a's and 1c's c=1000 points
  agree to 0.7% *across* the OOM incident -- but it should be confirmed.

## Phase 2 redesigned

The original A/B/C design asked whether MP was disadvantaged by being forced to
`--disable-hybrid-kv-cache-manager` while IP kept the allocator.  That question
is void: 0.22.1 auto-disables for IP too, so both VAST arms ran at 13.7M.
Confirmed by extracting the configs from the PDF rather than trusting my
transcription.

The real asymmetry is the storage tier, verbatim from their configs:

- MP: `lmcache server --l1-size-gb 1600` -- a 1.6 TB CPU cache doing real work
- IP: `local_cpu: false` -- no tier at all; stores nothing, retrieves nothing

Their "MP vs IP" may therefore be closer to "LMCache working" vs "LMCache idle".
New four-arm design (`scripts/phase2_mp_vs_ip.sh`, ISL=120k, c=100/300/600):

| arm | config | pool | tier |
|---|---|---|---|
| A `ip_notier` | IP as VAST ran it | 13.7M | none |
| B `ip_cputier` | IP with 62 GB x 8 = 496 GB L1 | 13.7M | yes |
| C `mp_vast` | MP as VAST ran it | 13.7M | yes |
| D `mp_hma` | MP without the flag | ~25.8M | yes |

B vs C is the fair comparison; A vs B is how much IP's tierless config flatters
it; C vs D is the 1.88x they give up for free.

`max_local_cpu_size` is **per TP rank**, so matching MP's 496 GB means 62.0 in
the IP YAML, not 496.  That arithmetic is what OOM-killed the box on Sep 1; the
script now aborts if `free -g` available is below `L1_GB + 250`.

## Harness bugs hit today

1. **Missing exec bit.** `phase1_control_1a.sh` was generated with `sed > file`,
   which does not set +x.  The chain hit `Permission denied`, logged it to a file
   nobody was reading, and **silently skipped the step**, moving on to the next.
   The 1a control simply did not happen.  Fixed: every chain step now checks its
   return code and stops the chain on failure, and exec bits are verified before
   launch.
2. **`pkill -f` matched my own shell, again.** `pkill -f "chain5.sh"` killed the
   Bash tool process running it (exit 144) because the pattern appears in that
   process's own command line.  I hit exactly this on Sep 1 with
   `pkill -f phase1_gpu_only` and had switched to explicit pids; I regressed.
   Rule, for real this time: **never `pkill -f` from a tool shell** -- capture the
   pid and `kill "$pid"`.
3. Two chain generations briefly overlapped and both logged "starting 1a
   control".  Only one driver actually launched (verified by pid count and by
   `pgrep -cf "vllm serve"`), but the window existed.  Chains are now created
   only after the previous one is confirmed dead by `kill -0`.

## State at the time of writing

Phase 1: 1a, 1b, 1c complete at c=100..1500 (1c's c=1500 warm lost to an engine
crash).  Controls: 1c re-run at c=300/600 done, 1a re-run at c=300 done and
c=600 in flight.  Queued behind it (`$CLAUDE_JOB_DIR/tmp/q.sh`, waiting on pid
798408): 1a @1000, 1c @1000, then Phase 2.

Nothing filed, nothing sent, nothing pushed.  No LMCache source changes -- this
whole line is investigation only.
