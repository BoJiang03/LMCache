# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the multimodal acceptance suite.

Opt-in guard: every test is skipped unless ``LMCACHE_MM_E2E=1``. The engine
harness is session-scoped per model: baselines are computed first in a
subprocess, then one LMCache engine serves all tests of that model.
"""

# Standard
import os
import pathlib
import sys

# Third Party
import pytest

# Pin THIS repo's lmcache package. The suite runs from tests/e2e_mm (to
# escape the global tests/conftest.py), so the repo root is not on sys.path
# and a stray editable install could silently resolve `import lmcache` to a
# DIFFERENT source tree -- the suite would then certify the wrong code.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def pytest_collection_modifyitems(config, items):
    # Modality-gated tests (e.g. T2.3 video) are DESELECTED -- not skipped --
    # for models whose spec does not declare the modality: a skip would
    # poison certification (certify.py treats any skip as failure), while a
    # deselect keeps the run's claim exactly as wide as the spec.
    from specs import MODEL_SPECS

    deselected = []
    kept = []
    for item in items:
        marker = item.get_closest_marker("requires_modality")
        if marker is not None:
            model_key = getattr(item, "callspec", None)
            model_key = model_key.params.get("harness") if model_key else None
            if (
                model_key is not None
                and marker.args[0] not in MODEL_SPECS[model_key].modalities
            ):
                deselected.append(item)
                continue
        kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept

    for item in items:
        item.add_marker(pytest.mark.mm_e2e)
    if os.environ.get("LMCACHE_MM_E2E") != "1":
        skip = pytest.mark.skip(
            reason="multimodal acceptance suite is opt-in: set LMCACHE_MM_E2E=1"
        )
        for item in items:
            item.add_marker(skip)


def _model_keys() -> list[str]:
    # Local import: keep collection working without the package installed.
    from specs import selected_model_keys

    return selected_model_keys()


@pytest.fixture(scope="session", params=_model_keys())
def harness(request, tmp_path_factory):
    """Session harness for one model: baselines + LMCache engine."""
    from catalog import catalog, pressure_requests, video_requests
    from harness import MMHarness, compute_baselines, configure_environment
    from specs import MODEL_SPECS

    configure_environment()
    # Fail loudly if `import lmcache` resolved outside this repo (see the
    # sys.path pinning at the top of this file).
    import lmcache

    lmcache_root = pathlib.Path(lmcache.__file__).resolve().parents[1]
    if lmcache_root != _REPO_ROOT:
        raise RuntimeError(
            f"lmcache resolved to {lmcache_root}, expected {_REPO_ROOT}; "
            "the suite would certify the wrong source tree"
        )
    spec = MODEL_SPECS[request.param]
    all_requests = list(catalog().values()) + pressure_requests(pressure_n())
    if "video" in spec.modalities:
        all_requests += list(video_requests().values())
    workdir = tmp_path_factory.mktemp(f"mm_e2e_{spec.key}")
    baselines = compute_baselines(spec, all_requests, workdir)
    h = MMHarness(spec, baselines)
    yield h
    h.close()


def pressure_n() -> int:
    """Number of distinct images for the collision pressure test (T0.2)."""
    return int(os.environ.get("LMCACHE_MM_E2E_PRESSURE_N", "64"))
