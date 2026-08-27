# SPDX-License-Identifier: Apache-2.0
"""Run one isolated_cases scenario standalone, with the env pytest would set.

usage: python run_isolated.py <scenario> <model_key> <out_json>

Replicates tests/e2e_mm/conftest.py::pytest_configure (the hybrid prompt
shape must be in place before catalog import) and harness.configure_environment,
then delegates to isolated_cases.main.
"""

import os
import pathlib
import sys

E2E = pathlib.Path("/home/bo/LMCache-worktrees/multi_modal/tests/e2e_mm")
sys.path.insert(0, str(E2E))

scenario, model_key, out_json = sys.argv[1:4]
os.environ["LMCACHE_MM_E2E_MODELS"] = model_key

from specs import MODEL_SPECS  # noqa: E402

spec = MODEL_SPECS[model_key]
if spec.hybrid_block_tokens and not os.environ.get("PROBE_NO_PAD"):
    import conftest  # noqa: E402
    from catalog import BOUNDARY_PHASES  # noqa: E402

    wpb = int(spec.hybrid_block_tokens * conftest.HYBRID_WORDS_PER_TOKEN)
    os.environ["LMCACHE_MM_E2E_PRE_PAD_WORDS"] = str(wpb * conftest.HYBRID_PRE_PAD_BLOCKS)
    os.environ["LMCACHE_MM_E2E_POST_PAD_WORDS"] = str(
        wpb * conftest.HYBRID_POST_PAD_BLOCKS
    )
    os.environ["LMCACHE_MM_E2E_MID_PAD_WORDS"] = str(wpb * 2)
    os.environ["LMCACHE_MM_E2E_BOUNDARY_STEP"] = str(max(1, wpb // BOUNDARY_PHASES))

from harness import configure_environment  # noqa: E402

configure_environment()

import isolated_cases  # noqa: E402

# Probe overrides, so trials do not need a spec edit each time.
if os.environ.get("PROBE_BLOCKS"):
    blocks = int(os.environ["PROBE_BLOCKS"])
    isolated_cases.PREEMPTION_GPU_BLOCKS = blocks
    import dataclasses

    spec = dataclasses.replace(spec, preemption_gpu_blocks=blocks)
    isolated_cases.MODEL_SPECS[model_key] = spec
if os.environ.get("PROBE_MAXLEN"):
    isolated_cases.PREEMPTION_MAX_MODEL_LEN = int(os.environ["PROBE_MAXLEN"])
if os.environ.get("PROBE_CAP_GB"):
    isolated_cases.EVICTION_CAPACITY_GB_MP = float(os.environ["PROBE_CAP_GB"])
if os.environ.get("PROBE_BUDGET"):
    _budget = int(os.environ["PROBE_BUDGET"])
    _orig = isolated_cases.run_preemption

    def _patched(spec_arg):
        import harness as _h

        _hek = _h.hybrid_engine_kwargs

        def _wrap(block_tokens, family=None):
            kw = _hek(block_tokens, family) if family else _hek(block_tokens)
            if kw:
                kw["max_num_batched_tokens"] = _budget
            return kw

        _h.hybrid_engine_kwargs = _wrap
        try:
            return _orig(spec_arg)
        finally:
            _h.hybrid_engine_kwargs = _hek

    isolated_cases.run_preemption = _patched
    isolated_cases.SCENARIOS["preemption"] = _patched

sys.exit(isolated_cases.main([scenario, model_key, out_json]))
