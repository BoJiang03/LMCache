# Chart misread, the NVIDIA citation does not say what the PDF claims, scope narrowed to NVIDIA

2026-09-02, afternoon session. Continues records 1-3 of the same day.

This session produced two corrections to numbers I had already reported, one
structural finding about the PDF itself, and three scope decisions from Bo.
The corrections come first because they invalidate the headline in record 3.

---

## 1. CORRECTION: I misread VAST's chart. Their gap is 9-18%, not 16.2%

At x=1500 the page-1 chart stacks three labels: `885142`, `831574`, `761774`.
Record 3 quoted 885142 as curve 1b and derived "VAST 1.162x vs mine 1.155x,
0.6% apart".

Zooming the render shows **`885142` carries the orange square marker, i.e. it
belongs to curve `2. GPU + LMCache-CPU`**. Curve 1b (thin pink, no markers)
ends at `831574`.

I digitized both curves by pixel to settle it independently of the labels,
calibrating on the two unambiguous points at x=1500 (red circle 761774, orange
square 885142). Script committed at `harness/analysis/digitize_pdf_chart.py`,
page render at `analysis/pdf_page1_chart.png` in the working dir.

| conc | 1a (red) | 1b (pink) | 1b/1a |
|---|---|---|---|
| 500 | 84,797 | 83,742 | **0.988x** |
| 700 | 264,053 | 294,632 | 1.116x |
| 800 | 335,756 | 367,390 | 1.094x |
| 900 | 399,023 | 454,909 | 1.140x |
| **1000** | 466,508 | 549,809 | **1.179x** (peak) |
| 1200 | 580,388 | 659,472 | 1.136x |
| 1400 | 701,650 | 773,353 | 1.102x |
| 1500 | 761,774 | 831,574 | **1.092x** (both from printed labels) |

Digitization accuracy check: the pixel read at c=1000 gives 549,809 against the
printed label 541,947 -- 1.4% off, so treat the derived ratios as +/-2%.

**Corrected reproduction claim.** The matching point is c=1000, not c=1500:

| conc | VAST 1b/1a | mine 1b/1a |
|---|---|---|
| 100 | -- | 1.004x |
| 500 / 600 | 0.988x | (600) 1.469x |
| **1000** | **1.179x** | **1.164x** |
| 1500 | 1.092x | 1.155x |

Finding (1) still reproduces, and the *shape* reproduces too (no gap at low
concurrency, gap opens under load). But "0.6% apart at c=1500" is retracted.

---

## 2. CORRECTION: at saturation the halved pool costs nothing; the tax is the connector

Decomposing warm P99 TTFT across the three arms (1a = full pool no connector,
1c = halved pool no connector, 1b = halved pool + IP connector):

| conc | 1a | 1c | 1b | pool 1c/1a | connector 1b/1c | total 1b/1a |
|---|---|---|---|---|---|---|
| 100 | 23.0 | 21.1 | 23.1 | 0.919x | 1.093x | 1.004x |
| 300 | 81.4 | 155.6 | 118.9 | 1.912x | 0.764x | 1.461x |
| 600 | 295.9 | 368.9 | 434.6 | 1.247x | 1.178x | 1.469x |
| **1000** | 612.7 | 616.1 | 713.0 | **1.005x** | **1.157x** | 1.164x |
| 1500 | 928.2 | crash | 1072.1 | -- | -- | 1.155x |

**At c=1000 the pool halving is free (1.005x) and the entire 1.16x is the
connector.** The "LMCache halves your KV pool" story, which records 1-3 lead
with, only holds at the knee (c=600). This is the opposite emphasis from what
those records give.

Caveat that still stands: 1a/1c are Sep 1, 1b is Sep 2 morning. The same-session
c=1000 set was being collected when the session was re-scoped.

---

## 3. The connector stores nothing and still charges 16%

`1b`'s server log, all 501 sampled lines:

```
External prefix cache hit rate: 0.0%
```

Never non-zero. With `local_cpu: false` the LocalCPUBackend is created as an
allocator only (`use_hot=False`, excluded from `get_non_allocator_backends()`),
so LMCache stores no token and serves no token -- and still costs 16% at c=1000.
This is pure overhead with zero benefit, which is a worse story than the pool.

A second signal, present only with the connector attached:

| | max `Deferred` reqs |
|---|---|
| 1a | 0 |
| 1c | 0 |
| **1b** | **926** |

vLLM's scheduler defers requests when the connector is attached. Not yet chased.

---

## 4. STRUCTURAL: the PDF's config block belongs to item 2, not to the chart

Extracted the text of all five pages (`pypdf`). Page 2 opens with:

> `2. Performance comparison: LMCache-MP vs. LMCache-IP (performance degradation
> observed). Tested on AMD GPUs` / `Configurations Used` / `1. IP` ...

So the IP command line, the IP YAML (p3), the MP command line + `lmcache server`
line (p4) and the MP YAML (p5) are **all documented under finding (2)**.
**The page-1 chart has no configuration documented at all.**

This closes the question record 3 left open: we cannot tell whether the chart's
curve 1b is IP or MP, and the config block is not evidence either way.

Two attempts to infer the mode from the legend both failed:
- `save_chunk_meta` reaches `fs_connector` via `extra_config` in both modes.
- `GDS-hipifile` is not MP-only: IP has `storage_backend/gds_backend.py` +
  `hipfile_shim.py`; MP has `--gds-l1-backend {auto,cufile,hipfile,ugds,phx}`.

---

## 5. The NVIDIA blog is IP -- but it is a different experiment, and it is not GPU-only

`https://www.vastdata.com/blog/vastdata-lablup-kvcache-offload-benchmark`

**Mode: IP.** The article never writes `LMCacheConnectorV1`, `LMCacheMPConnector`,
`kv_connector`, `kv-transfer-config` or `lmcache server`. It does expose two
LMCache settings, and both are IP-only surface:

- `LMCACHE_CUFILE_BUFFER_SIZE` -- `lmcache/v1/config.py:991` `_update_config_from_env`
  builds env names as `f"LMCACHE_{attr_name.upper()}"` over `_CONFIG_DEFINITIONS`,
  the `LMCacheEngineConfig` field table. `cufile_buffer_size` was such a field
  historically (renamed to `gds_buffer_size` by `40adcc9b`, PR #2858).
- `gds_path` -- `config.py:336`; its only effect is `storage_backend/__init__.py:235`
  creating `GdsBackend`, in the in-process storage manager.

MP's equivalent is `lmcache server --gds-l1-path/--gds-l1-backend` in
`lmcache/v1/distributed/config.py`, which greps clean for `environ`/`getenv`/
`env_prefix` -- the MP server reads no environment variables. Under MP both blog
settings would be inert. **Inference, not stated in the article.**

**But it does not support finding (1) as worded.** The two arms are:

| turn | Without Offloading | With Offloading | |
|---|---|---|---|
| first (cold) | ~22.3s | ~26.3s | **+4s** |
| subsequent (warm reuse) | ~22.0s | ~6.6s | 3.3x faster |

average TTFT 22,104 ms -> 10,573 ms = **2.09x faster**.

"With Offloading" is `gds_path` pointing at a VAST mount with a GDS cuFile
buffer -- KV written to external storage. **That is not a "GPU-only KV cache
configuration".** The ~4s is an external-storage write cost on cold turns, and
the article's own headline is that LMCache is 2x faster.

Setup: Mistral Medium 3.5 128B, vLLM 0.20.0 (`vllm-openai:0.20.0-cuda12.9-ubuntu22.04`),
8x H100, 10 agent contexts x 5 turns = 50 requests **in sequence, not parallel**.

So PDF item 1 claims degradation "across both Nvidia and AMD platforms", but
citation (a) is a different phenomenon on a different configuration. **The only
GPU-only evidence in the PDF is the AMD chart.** Our H200 runs are, as far as
this repro can tell, the first NVIDIA GPU-only data point for the claim.

Two facts that make the pool mechanism *possible* over there anyway:
- **vLLM v0.20.0 already defines `SupportsHMA` and `supports_hma()`** (fetched
  `base.py` at tag v0.20.0), so the auto-disable existed in the blog's version.
- Mistral Medium 3.5 is 88 Ministral-3 decoder layers, and Ministral uses
  interleaved sliding-window attention (ragged 128k/32k/32k/32k) -> hybrid model
  -> would trigger the same switch. *Second-hand from NVIDIA NeMo docs; confirm
  against the HF `config.json` before using.*

We reproduced the blog's *shape* in a GPU-only config at c=100, where the pool
provably cannot bind:

| | 1a | 1b | |
|---|---|---|---|
| cold | 67.2s | 78.0s | **1.16x** |
| warm | 23.0s | 23.1s | **1.004x** |

Cold penalty, warm parity -- the same shape as "first turn +4s, later turns fine",
minus the 3.3x because we configured no storage tier to win it back.

---

## 6. What binds concurrency, and a residual that is still unexplained

At ISL=60000 with `block-size=64`: 938 blocks = 60,032 tokens per sequence.

| | pool | seqs it holds | `--max-num-seqs` | binds on |
|---|---|---|---|---|
| 1a | 25,798,626 | 429 | 256 | max_num_seqs |
| 1b / 1c | 13,724,416 | **228** | 256 | **the pool** |

vLLM prints this directly as `Maximum concurrency for 131,072 tokens per request`:
196.83x vs 104.71x.

**If "how many sequences fit" were the whole mechanism, 1c/1a would be a constant
256/228 = 1.123x.** Measured: 1.247x at c=600, 1.005x at c=1000. Neither matches.
**A third mechanism is unaccounted for.** This is the sharpest open question.

MI355X estimate (288 GB HBM x 8, `gpu_memory_utilization` 0.9, ~60 GB mxfp4
weights sharded): ~58.7M tokens with the allocator on, ~29.3M with it off ->
978 vs 489 sequences, both far above `max_num_seqs=256`. If that estimate holds,
**the pool difference is invisible on VAST's hardware** and their 9-18% is pure
connector cost -- which would explain their 0.988x at c=500 against our 1.469x
at c=600. One log line settles it: `GPU KV cache size: N tokens` from each of
their runs. That is now the first question to ask VAST.

---

## 7. Scope decisions taken this session (Bo)

1. **Finding (2) (IP vs MP) is parked.** Close (1) first. The four phase-2 arms
   are preserved but gated behind `RUN_PHASE2=1`.
2. **The AMD chart is no longer the reproduction target** -- we only have NVIDIA
   hardware. It stays as background. Consequence: concurrency points no longer
   need to match VAST's; pick what is stable on our box.
3. **Do not switch models to match the blog.** Mistral Medium 3.5 + vLLM 0.20.0
   would cost a second venv and a weight pull to measure a GDS write cost we
   have already reproduced the shape of in GPU-only form.
4. **1e (MP + `--disable-hybrid-kv-cache-manager`) dropped.** It existed only to
   resolve the chart's IP/MP wording ambiguity; with the chart out of scope it
   has no purpose. Ask VAST instead of spending machine time guessing.

### Working agreement

**No benchmark runs without Bo's explicit approval, one at a time, no auto-chaining.**
Reason given: a full day of self-directed sweeps felt like it produced little --
knee-region points where the box drifts +/-20%, cross-session pairings that had
to be retracted, and a queued 5-hour sweep for a question then parked. Recorded
to memory as `experiments-run-only-on-user-command`.

---

## 8. Harness changes

New:
- `scripts/phase1d_mp_gpu_only.sh` -- MP connector *without*
  `--disable-hybrid-kv-cache-manager`, tiny L1 (8 GB, `eviction-policy noop`) as
  the analogue of IP's `local_cpu:false`. Asserts the pool is >= 20M and aborts
  otherwise, so a failed `SupportsHMA` cannot be mistaken for a measurement.
  Readiness is `ss -ltn | grep 127.0.0.1:$MP_PORT`, **fatal on timeout** -- the
  earlier version grepped the log for `listening|serving|ready|started`, wording
  that has never been observed, and on no match would fall out of the loop and
  proceed anyway.
- `scripts/phase1_control_1b.sh` -- same-session 1b re-measure into `1b_rerun/`.
- `analysis/digitize_pdf_chart.py` -- the chart digitizer.

Gotcha worth keeping: **a running bash script must not be edited in place** --
bash re-reads by byte offset and would execute garbage. `q.sh` was live, so the
new steps were chained from `phase2_mp_vs_ip.sh`, which `q.sh` only invokes by
path later. When `q.sh` was later killed, that chaining became moot.

### Wall clock is roughly 1.6x the JSON `duration`

Planning was off because `vllm bench serve` tokenizes the whole prompt set before
the timed window. Measured on this box:

| point | JSON `duration` | actual wall clock per pass |
|---|---|---|
| c=300 | 99 s | 5.3 min |
| c=1000 | 624 s | **17 min** |

Server start is only ~105 s (`launching 14:11:38` -> first pass `14:13:23`) with
a warm torch.compile cache -- much faster than the ~10 min assumed earlier.

---

## 9. Process discipline

- Stopped the queue by **pid** (`kill 829889`), never `pkill -f`. That pattern
  matches the tool shell's own command line and has self-killed this session
  twice already (Sep 1 `phase1_gpu_only`, Sep 2 `chain5.sh`).
- Left the in-flight `1a@1000` warm pass running -- it is the baseline every
  design needs and its own EXIT trap tears down its vLLM cleanly.
- Never touched pid 3483837 (root's DeepSeek-V2-Lite on port 8024).
- Monitors cleaned: stopped `bccj182sc`, `b1tul3kn7`, `bw3c6vauj` (all watching
  dead chains); one left, `bmhvuc0hc`, on `logs/1d_mp.out`.

---

## 10. State at the end of this record

Running: `1a_rerun` c=1000 warm (started 14:30:23, ETA ~14:48).
Armed: runner pid 949970 waits for it, then runs `CONC="200 1000"
phase1d_mp_gpu_only.sh` -- the first MP measurement on this box. ETA ~15:35.
Nothing is queued after that.

Two early checkpoints in that run:
1. `GPU KV cache size` must print **25,798,626**. `13,724,416` means
   `LMCacheMPConnector`'s `SupportsHMA` did not take effect and the script aborts.
2. `lmcache server` must listen on 5765 or the script exits.

### Safe to quote

- Pool 25,798,626 vs 13,724,416 = 1.880x. Three sessions, byte-identical.
- The mechanism, the `vllm.py:1471` warning text, all source references.
- gpt-oss-120b = 18 `sliding_attention` (window 128) + 18 `full_attention`.
- `External prefix cache hit rate: 0.0%` across 501 samples in 1b.
- The binding arithmetic (429 vs 228 seqs against `max_num_seqs=256`).
- The blog's own numbers and that its enabled arm writes to VAST storage.

### Not safe to quote

- Any single cost multiplier at the knee (c=300, c=600): +/-20% session drift.
- The 14.2% "saturation tax" from record 1 -- 1b Sep 2 against 1a/1c Sep 1.
- **The 1.155x / 1.164x headline** -- still a cross-session pairing. The
  same-session c=1000 set is incomplete.
- Any cold/warm decomposition (retracted in record 3, still retracted).
- The MI355X pool estimate in section 6 -- arithmetic on an assumed HBM budget,
  not a measurement.

### Records 1-3 need edits once the c=1000 set lands

- Record 1 and 2 lead with the pool as *the* mechanism. Section 2 above says it
  is free at saturation. Rewrite the emphasis.
- Record 3's "VAST 1.162x, 0.6% apart" is wrong (section 1).
- Record 2's issue/reply drafts still carry the
  `NUMBERS UNDER REVISION -- do not file or send this section yet` banner.
  Leave it until the same-session numbers exist.
