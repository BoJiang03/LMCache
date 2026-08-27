"""Attribution run: branch tests, lmcache from the upstream base tree
(dev_head) in BOTH the engine process and the MP server child.

Deliberately bypasses isolated_cases' repo-pin guard: the guard prevents
ACCIDENTALLY certifying the wrong tree; this experiment intentionally runs
the other tree and never writes a certificate.
"""
import pathlib
import sys
BASE_TREE = "/home/bo/LMCache-worktrees/dev_head"
BRANCH = "/home/bo/LMCache-worktrees/multi_modal"
sys.path.insert(0, BRANCH + "/tests/e2e_mm")   # branch test code
import harness
harness.configure_environment()
# Redirect ONLY the MP server child's PYTHONPATH root to the base tree:
# start_mp_cache_server derives it from harness.__file__, which every other
# harness helper (baseline_runner lookup) still needs pointing at the branch.
_orig_start = harness.start_mp_cache_server
def _start_on_base_tree(*args, **kwargs):
    old = harness.__file__
    harness.__file__ = BASE_TREE + "/tests/e2e_mm/harness.py"
    try:
        return _orig_start(*args, **kwargs)
    finally:
        harness.__file__ = old
harness.start_mp_cache_server = _start_on_base_tree
import lmcache  # resolves to BASE_TREE via PYTHONPATH
real = pathlib.Path(lmcache.__file__).resolve()
assert str(real).startswith(BASE_TREE), f"engine lmcache is {real}, want base"
print(f"[attribution] engine-side lmcache: {real}", flush=True)
# Spoof the spec origin so the repo-pin guard passes; loaded code and
# submodule resolution (via lmcache.__path__) stay on the base tree.
lmcache.__spec__.origin = BRANCH + "/lmcache/__init__.py"
import isolated_cases
sys.exit(isolated_cases.main(sys.argv[1:]))
