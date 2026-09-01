"""Gate-only import guard: keep `lmcache` resolving to this worktree.

The vllm-lazy venv carries an editable install of lmcache pointing at the
development worktree, through three mechanisms: a meta_path finder, a path
hook plus a `__path_hook__` placeholder on sys.path, and the cached importer
for that entry. Strip all three so the engine imports the PR tree. Untracked;
deleted after the gate.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if "__editable___lmcache" not in type(finder).__module__
    and "__editable___lmcache" not in getattr(finder, "__module__", "")
]
sys.path[:] = [entry for entry in sys.path if "__editable___lmcache" not in entry]
sys.path_hooks[:] = [
    hook for hook in sys.path_hooks if "__editable___lmcache" not in repr(hook)
]
sys.path_importer_cache.clear()
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
