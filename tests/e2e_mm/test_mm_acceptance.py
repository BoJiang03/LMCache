# SPDX-License-Identifier: Apache-2.0
"""Multimodal model support acceptance tests (T0-T2).

See README.md in this directory for the acceptance matrix. All tests share
one session-scoped LMCache engine per model; cache isolation between test
cases comes from case-unique salts at the start of the system message, so
each case's very first token chunk is unique and cases cannot hit each
other's entries.

Hit-count assertion vocabulary (chunk_size = 16):
- "misses" == lookup_hits is 0 (case salt is fresh, chunk 0 cannot match);
- "full hit" == lookup_hits >= lookup_tokens - 2 * CHUNK (trailing partial
  chunk and the recompute-last-token rule);
- "did not reach the image" == at least IMAGE_SPAN_MARGIN fewer hit tokens
  than a full hit (test images are 448x448, spans of hundreds of tokens).
"""

# Third Party
import pytest

# First Party (test-local)
from catalog import (
    BOUNDARY_PHASES,
    catalog,
    color_request,
    pressure_requests,
    video_requests,
)
from conftest import pressure_n
from harness import LMCACHE_TEST_CHUNK_SIZE as CHUNK
from harness import effective_max_tokens

IMAGE_SPAN_MARGIN = 4 * CHUNK

CATALOG = catalog()


def test_t0_cross_image_isolation_and_hit_equivalence(harness):
    """T0.1 + T0.3 + T1.1 + T1.3 on one image pair."""
    req_a, req_b = CATALOG["t01-A"], CATALOG["t01-B"]

    a1 = harness.run(req_a)
    harness.check_output(req_a, a1, "T0.1 first A")
    assert a1.lookup_hits == 0, "fresh case salt must not hit anything"

    b1 = harness.run(req_b)
    harness.check_output(req_b, b1, "T0.1 different image B")

    a2 = harness.run(req_a)
    harness.check_output(req_a, a2, "T0.3 repeat A")
    harness.check_replay_text(req_a, a1.text, a2.text, "T0.3 repeat A")
    # T1.1 reuse depth + T1.3 non-degenerate: the repeat is a full hit.
    assert a2.lookup_hits > 0, "MM requests must actually hit (no bypass)"
    assert a2.lookup_hits >= a2.lookup_tokens - 2 * CHUNK

    # T0.1 counter check: B shares only the text prefix with A; if B's hits
    # reach into the image region, a false hit occurred.
    assert b1.lookup_hits <= a2.lookup_hits - IMAGE_SPAN_MARGIN, (
        f"image B hit {b1.lookup_hits} tokens, too close to A's full hit "
        f"{a2.lookup_hits} -- cross-image false hit"
    )

    b2 = harness.run(req_b)
    harness.check_output(req_b, b2, "T0.3 repeat B")
    harness.check_replay_text(req_b, b1.text, b2.text, "T0.3 repeat B")
    assert b2.lookup_hits >= b2.lookup_tokens - 2 * CHUNK


def test_t0_collision_pressure(harness):
    """T0.2 + T0.7: N distinct same-shape images, with storage conservation.

    T0.2 is the regression for the 16-bit mm_hash truncation (issue #3301):
    with 16-bit identity, ~300 distinct images give ~50% collision
    probability; any collision shows up here as a hit count above the
    text-prefix steady state.

    T0.7 audits the STORE side of the same traffic: every token the lookup
    missed must be store-requested AND land as resident chunk keys in the
    local CPU backend (under-storage = silently dropped KV), and the full-hit
    replay in pass 2 must store ~nothing new (over-storage = unstable keys or
    a lookup/store disagreement) while never losing resident entries.
    """
    requests = pressure_requests(pressure_n())
    n = len(requests)
    decode_budget = sum(effective_max_tokens(harness.spec, r) for r in requests)

    stored_0 = harness.stored_tokens_total()
    resident_0 = harness.storage()
    pass1 = [harness.run(r) for r in requests]

    assert pass1[0].lookup_hits == 0, "fresh case salt must not hit anything"
    steady = pass1[1].lookup_hits
    # The steady state is the shared text prefix only, never into the image.
    assert steady <= pass1[1].lookup_tokens - IMAGE_SPAN_MARGIN
    for i, res in enumerate(pass1[1:], start=1):
        same_identifier = [
            j
            for j in range(i)
            if pass1[j].identifiers and pass1[j].identifiers == res.identifiers
        ]
        assert res.lookup_hits == steady, (
            f"pressure request {i}: hit {res.lookup_hits} tokens, steady "
            f"state is {steady} -- cache-key anomaly (above steady = false "
            f"hit on another image; below = lookup misbehavior). "
            f"identifiers={res.identifiers} "
            f"earlier requests with identical identifiers: {same_identifier}"
        )

    # T0.7 pass-1 conservation: missed tokens -> store requests -> resident
    # keys. Tolerances: chunk alignment (one partial chunk per request) and
    # the decode tokens vLLM may also hand over for storage.
    stored_1 = harness.stored_tokens_total()
    resident_1 = harness.storage()
    missed = sum(r.lookup_tokens - r.lookup_hits for r in pass1)
    stored_delta = stored_1 - stored_0
    assert stored_delta >= missed - n * CHUNK, (
        f"under-storage: lookup missed {missed} tokens but only "
        f"{stored_delta} were store-requested -- LMCache is dropping KV"
    )
    assert stored_delta <= missed + decode_budget + n * CHUNK, (
        f"over-storage: {stored_delta} tokens store-requested for only "
        f"{missed} missed (+{decode_budget} decode budget) -- duplicate "
        f"stores or a lookup/store disagreement"
    )
    key_delta = resident_1.num_keys - resident_0.num_keys
    expected_chunks = missed // CHUNK
    assert key_delta >= expected_chunks - n, (
        f"resident-key deficit: expected ~{expected_chunks} new chunk keys, "
        f"found {key_delta} -- store-requested KV never became resident "
        f"(allocation failure or silent drop)"
    )
    assert key_delta <= expected_chunks + 2 * n, (
        f"resident-key surplus: expected ~{expected_chunks} new chunk keys, "
        f"found {key_delta} -- unstable keys are storing duplicates"
    )
    assert resident_1.total_bytes > resident_0.total_bytes

    # Pass 2: every image repeats, must fully hit and reproduce its output.
    for req, first in zip(requests, pass1, strict=True):
        second = harness.run(req)
        harness.check_replay_text(req, first.text, second.text, "T0.2 replay")
        assert second.lookup_hits >= second.lookup_tokens - 2 * CHUNK

    # T0.7 pass-2 conservation: a full-hit replay has nothing new to cache.
    stored_2 = harness.stored_tokens_total()
    resident_2 = harness.storage()
    assert stored_2 - stored_1 <= decode_budget + n * CHUNK, (
        f"full-hit replay re-stored {stored_2 - stored_1} tokens -- the "
        f"store path does not trust its own lookup"
    )
    new_keys = resident_2.num_keys - resident_1.num_keys
    assert new_keys >= 0, (
        f"{-new_keys} resident chunk keys VANISHED during the replay while "
        f"the cache is far under capacity -- KV loss (eviction or pin bug)"
    )
    assert new_keys <= n, (
        f"full-hit replay added {new_keys} resident chunk keys -- keys are "
        f"not stable across identical requests"
    )
    if key_delta > 0:
        avg_chunk_bytes = (resident_1.total_bytes - resident_0.total_bytes) / key_delta
        bytes_growth = resident_2.total_bytes - resident_1.total_bytes
        assert bytes_growth <= (new_keys + n) * avg_chunk_bytes, (
            f"replay grew resident bytes by {bytes_growth} with only "
            f"{new_keys} new keys -- bytes leaking without keys"
        )


def test_t0_concurrent_batch(harness):
    """T0.8: concurrently scheduled batch with duplicate images.

    All prior T0 tests submit requests one at a time; this one hands vLLM a
    single batch containing two copies of the same image request, a second
    image, and a text-only request, so LMCache sees interleaved lookups and
    stores — including the store of one copy racing the lookup of its
    duplicate. Verification is output-based per batch entry (counters cannot
    be attributed within a batch), plus a single follow-up run that proves
    the batch's stores are usable.
    """
    req_a, req_b = CATALOG["t08-A"], CATALOG["t08-B"]
    req_text = CATALOG["t08-text"]
    batch = [req_a, req_b, req_a, req_text, req_b]

    result = harness.run_batch(batch)
    for i, (req, text) in enumerate(zip(batch, result.texts, strict=True)):
        harness.check_text(req, text, f"T0.8 batch entry {i}")
    assert result.lookup_hits <= result.lookup_tokens

    # The batch's stores must be complete and hittable afterwards.
    a_after = harness.run(req_a)
    harness.check_output(req_a, a_after, "T0.8 single run after batch")
    assert a_after.lookup_hits >= a_after.lookup_tokens - 2 * CHUNK, (
        f"request cached during a batch only hit {a_after.lookup_hits} of "
        f"{a_after.lookup_tokens} tokens afterwards"
    )


def test_detector_sensitivity_negative_control(harness):
    """The suite's tripwire must FIRE when MM identity is deliberately broken.

    A green suite only certifies a model if its detectors are actually
    sensitive. This negative control disables the mm_hash substitution (the
    exact failure mode of issue #3301 taken to its extreme: cache keys blind
    to image content) for a fresh salt and asserts that the T0.1-style
    counter check DOES trip: the second, different image must falsely hit
    into the first image's cache entries. If this test fails, counter-based
    detection is broken and every other green result is meaningless.
    """
    req_a = color_request("t0blind-A", "t0blind", 0)
    req_b = color_request("t0blind-B", "t0blind", 2)

    with harness.identity_blindness():
        a1 = harness.run(req_a)
        assert a1.lookup_hits == 0, "fresh case salt must not hit anything"
        b1 = harness.run(req_b)

    # Under identity blindness B's placeholder tokens are identical to A's,
    # so B must reach A's full prompt depth -- the false hit the real
    # detector (IMAGE_SPAN_MARGIN separation) would flag as contamination.
    assert b1.lookup_hits > b1.lookup_tokens - IMAGE_SPAN_MARGIN, (
        f"negative control did not trip: with identity substitution "
        f"disabled, image B hit only {b1.lookup_hits} of "
        f"{b1.lookup_tokens} tokens -- the counter detector would not have "
        f"seen a real cross-image collision either"
    )


@pytest.mark.parametrize("phase", range(BOUNDARY_PHASES))
def test_t0_chunk_boundary_phases(harness, phase):
    """T0.4: correctness holds for every span-vs-chunk alignment phase."""
    req_a = CATALOG[f"t04-p{phase}-A"]
    req_b = CATALOG[f"t04-p{phase}-B"]

    a1 = harness.run(req_a)
    harness.check_output(req_a, a1, f"T0.4 phase {phase} first A")
    b1 = harness.run(req_b)
    harness.check_output(req_b, b1, f"T0.4 phase {phase} different image B")
    a2 = harness.run(req_a)
    harness.check_output(req_a, a2, f"T0.4 phase {phase} repeat A")

    harness.check_replay_text(req_a, a1.text, a2.text, f"T0.4 phase {phase}")
    assert a2.lookup_hits >= a2.lookup_tokens - 2 * CHUNK
    assert b1.lookup_hits <= a2.lookup_hits - IMAGE_SPAN_MARGIN


def test_t0_mixed_traffic(harness):
    """T0.5: interleaved text-only and multimodal requests stay isolated."""
    text_req = CATALOG["t05-text"]
    req_a, req_b = CATALOG["t05-A"], CATALOG["t05-B"]

    t1 = harness.run(text_req)
    harness.check_output(text_req, t1, "T0.5 text before MM")
    a1 = harness.run(req_a)
    harness.check_output(req_a, a1, "T0.5 MM A")
    t2 = harness.run(text_req)
    harness.check_output(text_req, t2, "T0.5 text repeat")
    harness.check_replay_text(text_req, t1.text, t2.text, "T0.5 text repeat")
    assert t2.lookup_hits >= t2.lookup_tokens - 2 * CHUNK
    b1 = harness.run(req_b)
    harness.check_output(req_b, b1, "T0.5 MM B")
    a2 = harness.run(req_a)
    harness.check_output(req_a, a2, "T0.5 MM A repeat")
    harness.check_replay_text(req_a, a1.text, a2.text, "T0.5 MM A repeat")


def test_t1_prefix_reuse_across_questions(harness):
    """T1.2: same image + different question reuses the shared prefix."""
    req_a, req_q2 = CATALOG["t12-A"], CATALOG["t12-A-q2"]

    a1 = harness.run(req_a)
    harness.check_output(req_a, a1, "T1.2 first question")
    q2 = harness.run(req_q2)
    harness.check_output(req_q2, q2, "T1.2 second question")

    # The shared system+image prefix must hit; only the question tail differs.
    assert q2.lookup_hits >= a1.lookup_tokens - 6 * CHUNK, (
        f"prefix reuse too shallow: hit {q2.lookup_hits} of "
        f"~{a1.lookup_tokens} shared-prefix tokens"
    )
    assert q2.lookup_hits < q2.lookup_tokens, (
        "different question must not be a full hit"
    )


def test_t2_multi_image_order(harness):
    """T2.1: (A, B) and (B, A) must not cross-hit and answer in order."""
    req_ab, req_ba = CATALOG["t21-AB"], CATALOG["t21-BA"]

    ab1 = harness.run(req_ab)
    harness.check_output(req_ab, ab1, "T2.1 order AB")
    ba1 = harness.run(req_ba)
    harness.check_output(req_ba, ba1, "T2.1 order BA")
    ab2 = harness.run(req_ab)
    harness.check_output(req_ab, ab2, "T2.1 repeat AB")

    harness.check_replay_text(req_ab, ab1.text, ab2.text, "T2.1 repeat AB")
    assert ab2.lookup_hits >= ab2.lookup_tokens - 2 * CHUNK
    # Swapped order diverges at the first image: BA must not hit into AB's
    # image region.
    assert ba1.lookup_hits <= ab2.lookup_hits - IMAGE_SPAN_MARGIN


@pytest.mark.requires_modality("video")
def test_t2_video_isolation_and_hit(harness):
    """T2.3: T0.1 + T0.3 + T1 rerun on the video modality.

    Videos travel a different vLLM ingestion path (multi-frame decode,
    temporal merge) but must land on the same LMCache guarantees: content
    identity in the keys, hit-path equivalence, and no cross-item hits.
    Deselected at collection for models whose spec does not declare video.
    """
    videos = video_requests()
    req_a, req_b = videos["t23-A"], videos["t23-B"]

    a1 = harness.run(req_a)
    harness.check_output(req_a, a1, "T2.3 first video A")
    assert a1.lookup_hits == 0, "fresh case salt must not hit anything"
    assert a1.identifiers, "video request produced no multimodal identifiers"

    b1 = harness.run(req_b)
    harness.check_output(req_b, b1, "T2.3 different video B")

    a2 = harness.run(req_a)
    harness.check_output(req_a, a2, "T2.3 repeat video A")
    harness.check_replay_text(req_a, a1.text, a2.text, "T2.3 repeat video A")
    assert a2.lookup_hits > 0, "video requests must actually hit (no bypass)"
    assert a2.lookup_hits >= a2.lookup_tokens - 2 * CHUNK

    # B shares only the text prefix with A; hits into the video span would
    # be cross-video contamination (the video span is hundreds of tokens).
    assert b1.lookup_hits <= a2.lookup_hits - IMAGE_SPAN_MARGIN, (
        f"video B hit {b1.lookup_hits} tokens, too close to A's full hit "
        f"{a2.lookup_hits} -- cross-video false hit"
    )


def test_t2_partial_sharing(harness):
    """T2.2: request [A] then [A, C]: shared prefix hits, C computed fresh."""
    req_a, req_ac = CATALOG["t22-A"], CATALOG["t22-AC"]

    a1 = harness.run(req_a)
    harness.check_output(req_a, a1, "T2.2 single image")
    ac = harness.run(req_ac)
    harness.check_output(req_ac, ac, "T2.2 image pair")

    # The hit must reach well into image A's span (the text prefix alone is
    # a few dozen tokens; the image span is hundreds).
    assert ac.lookup_hits >= IMAGE_SPAN_MARGIN + CHUNK, (
        f"shared image prefix not reused: only {ac.lookup_hits} tokens hit"
    )
    assert ac.lookup_hits < ac.lookup_tokens, (
        "the second image is new; a full hit here is a false hit"
    )
