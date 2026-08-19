# Reproduction scripts (branch `multi_modal_repro`)

This branch is `multi_modal` plus reproduction/verification scripts. It is
never merged into `dev`; it exists so PR reviewers can reproduce the issues
and verify the fixes.

## mm_hash_collision_repro.py

Reproduces LMCache issue #3301: the pre-fix 16-bit truncation of multimodal
identifiers (`hex_hash_to_int16`) lets two different images share all KV
cache keys, silently serving the wrong image's KV (~50% probability at ~300
distinct same-shape images, >99% at 800).

```bash
CUDA_VISIBLE_DEVICES=0 python repro/mm_hash_collision_repro.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct --num-images 800
```

Exit codes: `1` = false hit reproduced (buggy build), `0` = no false hit
(fixed build, or no 16-bit collision found — rerun with more images).

The script records the real identifiers the connector sees, finds a pair
colliding under 16-bit truncation with different image colors, then asks the
model for each image's color: on a buggy build the second image is answered
with the FIRST image's color.

## Alternative: the acceptance suite as red/green evidence

The acceptance suite on `multi_modal` (`tests/e2e_mm/`) detects the same bug
via hit counters, without needing an actual answer flip:

```bash
# On the pre-fix code: FAILS (false hit detected by hit-count invariant)
# On the fixed code: PASSES
cd tests/e2e_mm && LMCACHE_MM_E2E=1 LMCACHE_MM_E2E_PRESSURE_N=800 \
    CUDA_VISIBLE_DEVICES=0 pytest . -k collision_pressure
```
