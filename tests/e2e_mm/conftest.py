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


# Hybrid prompt shape, in unified blocks. The pre-pad gives requests that
# differ only in image content a shared, cacheable prefix; the post-pad puts
# whole blocks AFTER the image span so such requests also differ there --
# without it, block-granular hit counts would be identical for different
# images and the suite's primary cross-image detector would be blind. The
# mid-pad (between consecutive multimodal items) lets a partial-sharing hit
# reach past the first image.
HYBRID_PRE_PAD_BLOCKS = 2
HYBRID_POST_PAD_BLOCKS = 4
# ~1 token per pad word (verified on the Qwen tokenizers).
HYBRID_WORDS_PER_TOKEN = 1.0
# The MP cache server backing a hybrid run. GDN state pages are fat (~13 MB
# per block on Qwen3.5-2B), and the pressure test stores 64 padded prompts.
# Deeper hybrids cost far more per block (~205 MB on Qwen3.6-27B) and
# override this via ``ModelSpec.mp_server_l1_gb``; too small a capacity
# evicts inside a test and fails its store-conservation audit.
HYBRID_MP_SERVER_L1_GB = 60.0


def pytest_configure(config):
    """Set the prompt shape for hybrid models BEFORE test modules import.

    ``test_mm_acceptance`` builds its catalog at import time, so the pad
    environment must be in place before collection imports it; a later
    change would give the tests different salts than the baselines.

    Args:
        config: The pytest config (unused).

    Raises:
        RuntimeError: If a hybrid model is selected together with other
            models — the prompt shape is global, so one run certifies one
            hybrid model.
    """
    from specs import MODEL_SPECS

    keys = _model_keys()
    hybrid = [k for k in keys if MODEL_SPECS[k].hybrid_block_tokens]
    if not hybrid:
        return
    if len(keys) > 1:
        raise RuntimeError(
            f"hybrid model {hybrid[0]!r} needs a padded prompt shape that "
            f"applies to the whole run; select it alone (got {keys})"
        )
    block = MODEL_SPECS[hybrid[0]].hybrid_block_tokens
    words_per_block = int(block * HYBRID_WORDS_PER_TOKEN)
    os.environ["LMCACHE_MM_E2E_PRE_PAD_WORDS"] = str(
        words_per_block * HYBRID_PRE_PAD_BLOCKS
    )
    os.environ["LMCACHE_MM_E2E_POST_PAD_WORDS"] = str(
        words_per_block * HYBRID_POST_PAD_BLOCKS
    )
    os.environ["LMCACHE_MM_E2E_MID_PAD_WORDS"] = str(words_per_block * 2)
    # Sweep the T0.4 alignment phases across one whole block period.
    from catalog import BOUNDARY_PHASES

    os.environ["LMCACHE_MM_E2E_BOUNDARY_STEP"] = str(
        max(1, words_per_block // BOUNDARY_PHASES)
    )


def pytest_collection_modifyitems(config, items):
    # Spec-gated tests (modality, special-architecture add-on suites) are
    # DESELECTED -- not skipped -- for models whose spec does not declare
    # the capability: a skip would poison certification (certify.py treats
    # any skip as failure), while a deselect keeps the run's claim exactly
    # as wide as the spec.
    from specs import MODEL_SPECS

    deselected = []
    kept = []
    for item in items:
        model_key = getattr(item, "callspec", None)
        model_key = model_key.params.get("harness") if model_key else None
        spec = MODEL_SPECS[model_key] if model_key is not None else None
        gated_out = False
        if spec is not None:
            # ALL requires_modality markers must be satisfied, not just the
            # closest one: a cross-modal test needs two modalities at once,
            # and reading only the closest marker would silently run it on a
            # model missing the other.
            modalities = {
                arg
                for marker in item.iter_markers("requires_modality")
                for arg in marker.args
            }
            extra = item.get_closest_marker("requires_extra_suite")
            gated_out = not modalities <= spec.modalities or (
                extra is not None and extra.args[0] not in spec.extra_suites
            )
        if gated_out:
            deselected.append(item)
            continue
        kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept

    # Run tests that do NOT use the session engine BEFORE those that do.
    # The isolated scenarios spawn their own engine subprocess at the same
    # gpu_memory_utilization on the same GPU; the session harness — once
    # created by the first harness-using test — stays resident until
    # session end, and two engines do not fit. This was previously an
    # accident of file-name order (test_isolated_paths < test_mm_acceptance)
    # that the deepstack add-on file broke; the sort is stable, so order
    # within each group is unchanged.
    items.sort(key=lambda item: "harness" in getattr(item, "fixturenames", ()))

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
    """Session harness for one model: baselines + LMCache engine.

    A Mamba/GDN hybrid model runs on the MP deployment path with a cache
    server started here: vLLM's hybrid KV cache manager is only offered to
    connectors that advertise support for it, and the in-process
    ``LMCacheConnectorV1`` does not — it fails engine init outright
    ("Hybrid KV cache manager is disabled but failed to convert the KV
    cache specs to one unified type").
    """
    from catalog import (
        audio_requests,
        catalog,
        cross_modal_requests,
        pressure_requests,
        video_requests,
    )
    from harness import (
        MMHarness,
        MPHarness,
        compute_baselines,
        configure_environment,
        start_mp_cache_server,
    )
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
    if "audio" in spec.modalities:
        all_requests += list(audio_requests().values())
    if {"image", "audio"} <= spec.modalities:
        all_requests += list(cross_modal_requests().values())
    workdir = tmp_path_factory.mktemp(f"mm_e2e_{spec.key}")
    baselines = compute_baselines(spec, all_requests, workdir)
    if not spec.hybrid_block_tokens:
        h = MMHarness(spec, baselines)
        yield h
        h.close()
        return

    server = start_mp_cache_server(
        zmq_port=24000 + (os.getpid() % 1000),
        http_port=24000 + (os.getpid() % 1000) + 1000,
        chunk_size=spec.hybrid_block_tokens,
        log_path=workdir / "mp_server.log",
        l1_size_gb=spec.mp_server_l1_gb or HYBRID_MP_SERVER_L1_GB,
        separate_object_groups=True,
    )
    # The hybrid engine settings themselves come from the spec via
    # MMHarness (shared with the baseline engine).
    h = MPHarness(
        spec,
        baselines,
        zmq_port=server.zmq_port,
        http_port=server.http_port,
    )
    try:
        yield h
    finally:
        h.close()
        server.process.terminate()


def pressure_n() -> int:
    """Number of distinct images for the collision pressure test (T0.2)."""
    return int(os.environ.get("LMCACHE_MM_E2E_PRESSURE_N", "64"))
