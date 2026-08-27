"""Strip the lmcache editable-install finder from this process.

The vllm-lazy venv carries a PEP 660 editable install of lmcache pointing
at /home/bo/LMCache-worktrees/lazy_offloading. Its meta-path finder sits
AFTER the path-based finder, so with PYTHONPATH pointing at another
LMCache tree the package itself resolves correctly — but any submodule
that tree lacks (lmcache.cuda_ops, lmcache.c_ops, ...) silently falls
through to the FOREIGN tree's compiled extensions, mixing two branches in
one process. This guard removes that finder so imports resolve from
PYTHONPATH alone and missing extensions fail cleanly (torch fallback).

Loaded automatically (sitecustomize) by every Python process whose
PYTHONPATH includes this directory — engine, MP server, and baseline
subprocesses alike. The venv itself is untouched.
"""

import sys

sys.meta_path = [
    finder
    for finder in sys.meta_path
    if "__editable___lmcache" not in (getattr(finder, "__module__", "") or "")
]
