# Authoritative certificate set — schema 4, commit `dc1590c1`

All 11 registered models, re-certified together on one tree so the whole
support claim rests on a single commit. **These supersede every
`certificate_*.json` in the dated folders above** (14 older files remain
there as history; four of those state things that are no longer true — see
`../9_certificate_correctness_pass_all_11_reissued.md` §1).

| model | verdict | tests | benchmark | n | coverage | suite time |
|---|---|---|---|---|---|---|
| qwen2-vl-2b | SUPPORTED | 29 | MME | 2374 | null (in-process) | 766s |
| qwen2.5-vl-3b | SUPPORTED | 29 | MME | 2374 | null (in-process) | 771s |
| internvl3.5-2b | SUPPORTED | 29 | MME | 2374 | null (in-process) | 776s |
| qwen3-vl-2b | SUPPORTED | 34 | MME | 2374 | null (in-process) | 750s |
| glm-4.6v-flash | SUPPORTED | 29 | MME | 2374 | null (in-process) | 1175s |
| gemma-3-4b | SUPPORTED | 27 | MME | 2374 | 1.0056 | 1360s |
| gemma-4-e4b | SUPPORTED | 27 | MME | 2374 | 1.0076 | 1305s |
| qwen3.5-2b | SUPPORTED | 27 | MME | 2374 | 1.0 | 1873s |
| qwen3.6-27b | SUPPORTED | 27 | MME | 2374 | 1.0563 | 1109s |
| qwen3.8-27b | SUPPORTED | 27 | MME | 2374 | 1.0586 | 1706s |
| qwen3-omni-30b | SUPPORTED | 31 | **MMAU** | 1000 | null (in-process) | 941s |

Every one: `schema_version: 4`, `tested_tree.stable: true`,
`commit dc1590c1`, 0 failures / 0 errors / 0 skips, parity gate pass.
3.5 h of suite time, run as two waves of 5-6 across GPUs 1,2,3,5,6,7.

## Provenance of the parity evidence

Nine reuse recorded reports; **qwen3.8-27b was re-run fresh** because its
Aug-21 MP server log had been lost, so its TTL-cleanliness could not be
checked. The fresh run confirms the old numbers (flips 13 vs 12, scores
within 1.6 on a 2800 scale, hit ratio 0.168 vs 0.164) with 0 failed reads
and 0 `read_lock_expired` — so the Aug-21 report was sound, now by
measurement rather than assumption.

For the other two pre-TTL-fix MP reports (qwen3.5-2b, qwen3.6-27b) the
server logs survive and show **0 failed reads**, which is the direct check.
The five in-process models were never exposed: the TTL lives in the MP
cache server (`MP_SERVER_L1_READ_TTL_S`) and `LocalCPUBackend` has no read
lock. Gemma 3/4's reports post-date the fix.

Note the hit numbers alone could not have settled this — Gemma 4 reported a
hit coverage of 1.0076 while corrupting 1288 of 2374 answers. Only the
failed-read log distinguishes a healthy run from that one.
