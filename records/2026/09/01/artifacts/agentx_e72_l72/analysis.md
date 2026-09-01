# Agentx CONC=72 pair rerun on shipped PR defaults (2026-09-01)

Same setup as the CONC=40 rerun (../agentx_e40_l40/analysis.md): PR code
1f037adc, shipped keys only (EVICTION_AWARE, horizon 2.5, deferral 30 s),
fresh MP server and engine per arm, aiperf inferencex-agentx-mvp, seed 1234,
1800 s + 600 s grace. Purpose: verify the sweep's high-load flagship row
(CONC=72, where the original table showed the largest lazy wins) still
holds without the deleted danger-floor knob.

## aiperf (client view)

| metric | e72 eager | l72 lazy | lazy delta |
|---|---|---|---|
| TTFT avg (ms) | 46019 | 41822 | -9.1% |
| TTFT p50 | 44963 | 39946 | -11.2% |
| TTFT p90 | 114995 | 92908 | -19.2% |
| TTFT p99 | 150816 | 110606 | -26.7% |
| e2e avg (ms) | 161712 | 148056 | -8.4% |
| e2e p50 | 90071 | 91906 | +2.0% |
| e2e p99 | 1229940 | 1164122 | -5.4% |
| ITL avg (ms) | 116.6 | 110.6 | -5.1% |
| ITL p99 | 339.1 | 359.1 | +5.9% (worse) |
| output tok/s (total) | 198.3 | 213.3 | +7.6% |
| tok/s/user avg | 10.9 | 12.1 | +11.0% |
| requests completed | 442 | 472 | +6.8% |

## Cache side

Lookup hit rate: e72 17.24M / 53.07M = 32.5%, l72 21.33M / 56.21M = 37.9%
(original sweep: 30% vs 41%). Prompt sources: l72 serves 18.81M external
tokens vs e72 15.23M (+23%) and computes 33.15M vs 35.71M (-7%). L1 writes
341,478 vs 1,045,532 chunks: lazy writes 33% of eager's volume into the
same 250 GB, slowing L1 turnover so entries survive to reuse; end usage
0.65 vs 0.72.

Ledger (l72, closes: 2056+1431+4+5 = 3496): admitted=3496 emitted=2056
emitted_overdue=1376 dropped_evicted=1431 rejected_unhashed=33
rejected_prefix_broken=1947 dropped_on_request_drop=4 pending=5. 67% of
emissions from the deadline, same share as the CONC=40 pair; 41% of
admitted ops dropped to eviction under this GPU pressure, and those drops
are the filter that saves the write bandwidth.

## Reading

The flagship row reproduces on shipped defaults with the same shape,
slightly softer numbers: original e2e -14% / throughput +9.9% / +13%
requests, rerun e2e -8.4% / +7.6% / +6.8%. TTFT improves at every
percentile (p99 -27%), per-user decode +11%, hit rate +5.4 points. The one
tail regression is ITL p99 +5.9%, consistent with the original sweep's
mid-range ITL observations. Conclusion: the sweep's high-load win did not
depend on the deleted danger-floor knob.
