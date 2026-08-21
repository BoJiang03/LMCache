# SPDX-License-Identifier: Apache-2.0
"""DeepStack add-on suite (``ModelSpec.extra_suites`` flag ``"deepstack"``).

DeepStack models (Qwen3-VL family) inject multiscale vision features into
the first few LLM layers through a per-step side buffer that lives OUTSIDE
the paged KV. For LMCache the risky path is a hit whose boundary lands
INSIDE an image placeholder span: vLLM resumes prefill mid-span and the
resumed image tokens need their deepstack payload scattered at the right
offsets. Natural LRU eviction produces such boundaries only by accident;
these tests produce them surgically (evict the stored tail, replay).

Why a KV-level oracle: measured on Qwen3-VL-2B (2026-08-21, records
2026/08/21), fully disabling the injection changes no output bytes on the
synthetic color probes — the injected features have per-level norms in the
hundreds, yet the answers are insensitive, so every output-based oracle in
the base suite is blind to a missing or misaligned payload. Comparing the
KV that the mid-span resume re-stores against the KV the original full
prefill stored DOES see it: recompute regime noise measured rel-Frobenius
0.02-0.04 per chunk, while a deepstack-blinded resume measured 0.55-0.70
(15-25x separation). The thresholds below sit between the two bands with
>3x margin each way; the negative control keeps the oracle honest.
"""

# Standard
from dataclasses import dataclass
import contextlib
import importlib

# Third Party
import pytest

# First Party (test-local)
from catalog import (
    MMRequest,
    color_request,
    multi_color_request,
    video_color_request,
)
from harness import LMCACHE_TEST_CHUNK_SIZE as CHUNK
from harness import (
    MMHarness,
    clone_resident_kv,
    evict_resident_keys,
    resident_chunk_keys,
    resident_kv_tensor,
)

pytestmark = pytest.mark.requires_extra_suite("deepstack")

# Per-chunk relative-Frobenius bands (see module docstring for the
# measurements they are calibrated against).
NORMAL_MEAN_REL_FRO_MAX = 0.15
BLIND_MEAN_REL_FRO_MIN = 0.30

# vLLM model classes whose deepstack side-buffer write the negative control
# can disable. Extend when a new deepstack architecture joins the registry.
_DEEPSTACK_MODEL_CLASSES = [
    ("vllm.model_executor.models.qwen3_vl", "Qwen3VLForConditionalGeneration"),
    ("vllm.model_executor.models.qwen3_vl_moe", "Qwen3VLMoeForConditionalGeneration"),
]


@contextlib.contextmanager
def deepstack_blindness():
    """Disable the deepstack side-buffer write on known model classes.

    While active, the multiscale payload is never written, so the LLM adds
    zeros at the injection layers — the exact failure mode the add-on
    suite must be able to detect (a resume path that forgets or misplaces
    the payload). The main embedding merge is untouched.
    """
    patched: list[tuple[type, object]] = []
    for module_name, class_name in _DEEPSTACK_MODEL_CLASSES:
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            continue
        # Test-only reach into vLLM internals: the buffer write is the one
        # seam that disables injection without touching anything else.
        original = cls._set_deepstack_input_embeds
        cls._set_deepstack_input_embeds = lambda self, embeds: None
        patched.append((cls, original))
    if not patched:
        raise RuntimeError("no deepstack model class importable; control is vacuous")
    try:
        yield
    finally:
        for cls, original in patched:
            cls._set_deepstack_input_embeds = original


@dataclass(frozen=True)
class ResumeOutcome:
    """Result of one surgical-eviction resume experiment.

    Attributes:
        first_text: Miss-pass generated text.
        replay_text: Resume-pass generated text.
        prompt_tokens: Prompt length per the lookup counter.
        chunk_count: Chunk keys the miss pass stored.
        keep_chunks: Chunks kept resident for the replay.
        replay_hits: Lookup hits of the replay.
        restored: Evicted chunks that were re-stored by the replay.
        missing: Evicted chunks NOT resident after the replay.
        mean_rel_fro: Mean per-chunk relative Frobenius distance between
            re-stored and original KV.
        max_rel_fro: Maximum per-chunk relative Frobenius distance.
    """

    first_text: str
    replay_text: str
    prompt_tokens: int
    chunk_count: int
    keep_chunks: int
    replay_hits: int
    restored: int
    missing: int
    mean_rel_fro: float
    max_rel_fro: float


def midspan_resume(
    harness: MMHarness, request: MMRequest, keep_chunks: int, blind: bool
) -> ResumeOutcome:
    """Store a request, evict its tail chunks, replay, and diff the KV.

    Args:
        harness: The session harness (engine must be otherwise idle so the
            resident-key delta is exactly this request's chunk chain).
        request: The request to run; must use a fresh, case-unique salt.
        keep_chunks: How many leading chunks survive the cut.
        blind: Run the replay under ``deepstack_blindness()`` (negative
            control) instead of normally.

    Returns:
        The measured ``ResumeOutcome``.
    """
    keys_before = set(resident_chunk_keys())
    first = harness.run(request)
    chain = [k for k in resident_chunk_keys() if k not in keys_before]
    if keep_chunks >= len(chain):
        raise ValueError(
            f"cut at {keep_chunks} chunks needs a longer chain ({len(chain)})"
        )
    cut = chain[keep_chunks:]
    originals = clone_resident_kv(cut)
    evict_resident_keys(cut)

    context = deepstack_blindness() if blind else contextlib.nullcontext()
    with context:
        replay = harness.run(request)

    rel_fros: list[float] = []
    missing = 0
    for key, original in originals.items():
        restored = resident_kv_tensor(key)
        if restored is None:
            missing += 1
            continue
        diff = restored.float() - original.float()
        denom = original.float().norm().item() or 1.0
        rel_fros.append(diff.norm().item() / denom)
    return ResumeOutcome(
        first_text=first.text,
        replay_text=replay.text,
        prompt_tokens=first.lookup_tokens,
        chunk_count=len(chain),
        keep_chunks=keep_chunks,
        replay_hits=replay.lookup_hits,
        restored=len(rel_fros),
        missing=missing,
        mean_rel_fro=sum(rel_fros) / len(rel_fros) if rel_fros else 0.0,
        max_rel_fro=max(rel_fros, default=0.0),
    )


def check_resume_mechanics(outcome: ResumeOutcome) -> None:
    """Assert the cut/resume plumbing behaved (independent of KV content).

    Verifies the replay hit EXACTLY the surviving prefix (the cut took
    effect and nothing else was hit), and that every evicted chunk was
    recomputed and re-stored under its original content-addressed key.

    Args:
        outcome: The experiment to check.

    Raises:
        AssertionError: On any mechanical deviation.
    """
    assert outcome.replay_hits == outcome.keep_chunks * CHUNK, (
        f"replay hit {outcome.replay_hits} tokens, expected exactly the "
        f"{outcome.keep_chunks}-chunk surviving prefix "
        f"({outcome.keep_chunks * CHUNK})"
    )
    assert outcome.missing == 0, (
        f"{outcome.missing} evicted chunks were never re-stored by the "
        f"resume -- the store path dropped recomputed KV"
    )
    assert outcome.restored == outcome.chunk_count - outcome.keep_chunks


def assert_cut_inside_image_span(outcome: ResumeOutcome) -> None:
    """Assert the hit boundary landed inside the (dominant) image span.

    The test images are 448x448 (hundreds of placeholder tokens) behind a
    short text prefix and before a one-chunk question, so a cut at least 4
    chunks in and at least 3 chunks before the prompt end is inside the
    span for every registered model's template.

    Args:
        outcome: The experiment to check.

    Raises:
        AssertionError: If the cut cannot have been inside the span.
    """
    assert outcome.keep_chunks >= 4, "cut landed before the image span"
    assert outcome.keep_chunks * CHUNK <= outcome.prompt_tokens - 3 * CHUNK, (
        f"cut at {outcome.keep_chunks * CHUNK} of {outcome.prompt_tokens} "
        f"tokens is past the image span"
    )


@pytest.mark.parametrize("keep_chunks", [5, 8])
def test_deepstack_midspan_resume_kv(harness, keep_chunks):
    """TD.1: mid-image-span resume recomputes KV equal to the full prefill.

    A hit boundary inside the image span forces the resumed image tokens
    through the deepstack injection path with a nonzero span offset; the
    re-stored KV must match the original full-prefill KV to recompute
    regime noise.
    """
    request = color_request(
        f"tds-kv{keep_chunks}", f"tds-kv{keep_chunks}", keep_chunks % 6
    )
    outcome = midspan_resume(harness, request, keep_chunks, blind=False)
    check_resume_mechanics(outcome)
    assert_cut_inside_image_span(outcome)
    harness.check_text(request, outcome.first_text, "TD.1 miss pass")
    harness.check_replay_text(
        request, outcome.first_text, outcome.replay_text, "TD.1 resume pass"
    )
    assert outcome.mean_rel_fro <= NORMAL_MEAN_REL_FRO_MAX, (
        f"resume-path KV diverged from the full-prefill KV: mean rel-Fro "
        f"{outcome.mean_rel_fro:.4f} (max {outcome.max_rel_fro:.4f}) over "
        f"{outcome.restored} chunks -- deepstack payload lost or misaligned "
        f"on the resume path"
    )


def test_deepstack_midspan_resume_second_image(harness):
    """TD.2: resume inside the SECOND image of a two-image request.

    The resumed multimodal item is not the first in the prompt, so the
    payload scatter must handle a nonzero item offset as well as a nonzero
    span offset.
    """
    request = multi_color_request("tds-2img", "tds-2img", (0, 2))
    probe = harness.run(request)
    chain_estimate = probe.lookup_tokens // CHUNK
    # Land in the tail image: 4 chunks before the prompt end is inside the
    # second image span (the question tail is about one chunk), and past
    # the halfway point of a symmetric two-image prompt.
    keep = chain_estimate - 4
    request2 = multi_color_request("tds-2img-b", "tds-2img-b", (0, 2))
    outcome = midspan_resume(harness, request2, keep, blind=False)
    check_resume_mechanics(outcome)
    assert outcome.keep_chunks * CHUNK > outcome.prompt_tokens // 2 + 2 * CHUNK, (
        "cut did not reach the second image's span"
    )
    harness.check_text(request2, outcome.first_text, "TD.2 miss pass")
    harness.check_replay_text(
        request2, outcome.first_text, outcome.replay_text, "TD.2 resume pass"
    )
    assert outcome.mean_rel_fro <= NORMAL_MEAN_REL_FRO_MAX, (
        f"second-image resume KV diverged: mean rel-Fro "
        f"{outcome.mean_rel_fro:.4f} (max {outcome.max_rel_fro:.4f})"
    )


@pytest.mark.requires_modality("video")
def test_deepstack_midspan_resume_video(harness):
    """TD.3: mid-span resume inside a video placeholder span.

    Videos take a different ingestion path (multi-frame decode, temporal
    merge, timestamped segments) but their tokens receive the same
    deepstack payload; the resume path must align it the same way.
    """
    request = video_color_request("tds-vid", "tds-vid", 0)
    outcome = midspan_resume(harness, request, 8, blind=False)
    check_resume_mechanics(outcome)
    assert_cut_inside_image_span(outcome)
    harness.check_text(request, outcome.first_text, "TD.3 miss pass")
    harness.check_replay_text(
        request, outcome.first_text, outcome.replay_text, "TD.3 resume pass"
    )
    assert outcome.mean_rel_fro <= NORMAL_MEAN_REL_FRO_MAX, (
        f"video resume KV diverged: mean rel-Fro {outcome.mean_rel_fro:.4f} "
        f"(max {outcome.max_rel_fro:.4f})"
    )


def test_deepstack_oracle_negative_control(harness):
    """TD.4: the KV oracle must FIRE when the payload is deliberately lost.

    Replays under ``deepstack_blindness()``: the resumed image tokens get a
    zero payload, exactly the fault class TD.1-TD.3 exist to catch. If the
    measured divergence does not clear the blind band, the oracle has no
    sensitivity and the green TD results above are meaningless.
    """
    request = color_request("tds-blind", "tds-blind", 3)
    outcome = midspan_resume(harness, request, 8, blind=True)
    check_resume_mechanics(outcome)
    assert_cut_inside_image_span(outcome)
    assert outcome.mean_rel_fro >= BLIND_MEAN_REL_FRO_MIN, (
        f"negative control did not trip: deepstack-blinded resume measured "
        f"mean rel-Fro {outcome.mean_rel_fro:.4f} (max "
        f"{outcome.max_rel_fro:.4f}), below the blind band "
        f"{BLIND_MEAN_REL_FRO_MIN} -- the KV oracle cannot detect a lost "
        f"payload and TD.1-TD.3 prove nothing"
    )
