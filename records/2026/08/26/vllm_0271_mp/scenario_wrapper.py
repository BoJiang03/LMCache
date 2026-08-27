"""Run one isolated_cases scenario standalone, with stack dumps on SIGUSR1.

Mirrors what tests/e2e_mm/test_isolated_paths.py does before spawning the
scenario: configure_environment() plus the non-hybrid prompt-shape knobs
pytest_configure sets, so a standalone run is not silently a different
experiment.

Usage: scenario_wrapper.py <e2e_dir> <scenario> <model_key> <out.json>
"""
import faulthandler
import os
import runpy
import signal
import sys

e2e_dir, scenario, model_key, out_json = sys.argv[1:5]
sys.path.insert(0, e2e_dir)

from harness import configure_environment  # noqa: E402
from specs import MODEL_SPECS  # noqa: E402

spec = MODEL_SPECS[model_key]
if spec.hybrid_block_tokens:
    raise SystemExit("hybrid prompt shape not replicated here; run via pytest")
if not spec.supports_system_role:
    os.environ["LMCACHE_MM_E2E_NO_SYSTEM_ROLE"] = "1"
if spec.media_first_template:
    os.environ["LMCACHE_MM_E2E_MEDIA_FIRST"] = "1"
configure_environment()
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
print(f"[wrapper] pid={os.getpid()} scenario={scenario} model={model_key}", flush=True)
sys.argv = ["isolated_cases.py", scenario, model_key, out_json]
runpy.run_path(os.path.join(e2e_dir, "isolated_cases.py"), run_name="__main__")
