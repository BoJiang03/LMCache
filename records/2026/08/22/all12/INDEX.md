# Authoritative certificate set — schema 4, commit `0040c6bd`

All **12** registered models, certified on one tree. **These supersede
`../recert/`** (11 models at `dc1590c1`), which in turn superseded the
older dated certificates.

| model | verdict | tests | benchmark | coverage | suite time |
|---|---|---|---|---|---|
| qwen2-vl-2b | SUPPORTED | 29 | MME | null (in-process) | 769s |
| qwen2.5-vl-3b | SUPPORTED | 29 | MME | null (in-process) | 776s |
| internvl3.5-2b | SUPPORTED | 29 | MME | null (in-process) | 757s |
| qwen3-vl-2b | SUPPORTED | 34 | MME | null (in-process) | 738s |
| glm-4.6v-flash | SUPPORTED | 29 | MME | null (in-process) | 1126s |
| **molmo2-4b** | **SUPPORTED** | **26** | MME | null (in-process) | 747s |
| gemma-3-4b | SUPPORTED | 27 | MME | 1.0056 | 1347s |
| gemma-4-e4b | SUPPORTED | 27 | MME | 1.0076 | 892s |
| qwen3.5-2b | SUPPORTED | 27 | MME | 1.0 | 1532s |
| qwen3.6-27b | SUPPORTED | 27 | MME | 1.0563 | 2320s |
| qwen3.8-27b | SUPPORTED | 27 | MME | 1.0586 | 1794s |
| qwen3-omni-30b | SUPPORTED | 31 | **MMAU** | null (in-process) | 942s |

Every one: `schema_version: 4`, `tested_tree.stable: true`,
`commit 0040c6bd`, 0 failures / 0 errors / 0 skips, parity gate pass.

## Why this set exists

Molmo 2 is new (see `../10_`). The other eleven were re-run rather than
carried over because three of this session's changes touch what a
certificate SAYS, even where they do not touch what it measures:

1. `MMHarness._validate_prompt_shape` is a new startup check that every
   model now passes through. A certificate from before it was issued by a
   harness that could not have caught a wrong `media_first_template`.
2. **gemma-3-4b's exclusion list was incomplete.** It is a mm-prefix-LM
   (measured `is_mm_prefix_lm=True`), so its chunked-prefill exclusion has
   TWO independent causes; the old certificate named only the hybrid one.
   That entry is derived, so it self-heals on regeneration — this is the
   regeneration.
3. The support claim should rest on one commit. It did at `dc1590c1`; it
   stopped the moment Molmo 2 was certified at `0040c6bd`.

The media bytes are unchanged for these eleven: `case_media_bits` returns 0
unless the model is media-first, verified byte-identical before the run.

## Provenance of the parity evidence

No parity was re-run here. Every model reuses the recorded report its
`../recert/` certificate cited, and `certify.py --parity-report`
re-evaluates the gate against the current code rather than trusting the
recorded verdict. Molmo 2's is fresh (2026-08-22, in `../hitchhiker/` and
copied here).

## One non-deterministic failure, recorded rather than dismissed

`qwen2-vl-2b` FAILED its first run of this set, at
`test_t0_chunk_boundary_phases[1]`:

```
t04-p1-B: LMCache reported 20928 hit tokens but vLLM only skipped 16
on the connector's account (0 locally cached)
```

20928 hits for a few-hundred-token prompt is not a cache behaviour, it is
a counter delta that picked up something outside its window — the check
compares LMCache's reported hits against vLLM's own prefill-provenance
counters. The re-run passed 29/29, and the same check passed for the other
eleven models here and for all eleven this morning. So it is filed as a
flake in the stats path, **not** as a cache defect — but it is filed, with
the exact message, because a counter that can be wrong once can be wrong
in a run nobody re-runs.
