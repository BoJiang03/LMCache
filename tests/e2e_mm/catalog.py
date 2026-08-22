# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic media and request builders for the MM suite.

Images are solid-color squares with a deterministic per-index pattern, so
that (a) every index yields distinct bytes (distinct vLLM ``mm_hash``), and
(b) the dominant color is a one-word semantic probe the model can answer.

Audio follows the same two rules, but its answer space had to be measured
rather than assumed, and several plausible probes turned out to be unusable.
Beep COUNTING, pitch height and pitch DIRECTION are all answered wrongly
and, worse, wrongly in a way that COLLAPSES distinct items onto the same
answer -- precisely the failure that blinds a cross-item detector, since
item A returning item B's cached answer is invisible when both answer
alike. Naming two clips in order, tried as a way to widen the space, failed
outright (0/9 correct), so audio probes stay single-clip.

What does work is the coarse KIND of sound. Five kinds -- tone, static
noise, repeated beeping, low rumble, warbling tone -- were each measured on
the certification target as correctly named, stable across two passes, and
distinct from every other. Silence is deliberately excluded: that model
calls it "tone", stably, colliding with the real tone. Note the boundary
this draws: beeping as a KIND is reliable while the NUMBER of beeps is not.

Every test case uses a unique ``salt`` as the first words of its system
message, so the very first token chunk already differs between cases and
cases cannot hit each other's cache entries.
"""

# Standard
from dataclasses import dataclass
from functools import lru_cache
import base64
import io
import math
import os
import tempfile
import wave

# Third Party
from PIL import Image

IMAGE_SIZE = 448
# Videos use a smaller frame so the placeholder span stays a few hundred
# tokens (8 frames x 224x224 with 2x temporal merge on Qwen2-VL ~= 256).
VIDEO_SIZE = 224
VIDEO_FRAMES = 8
VIDEO_FPS = 2

_PALETTE: list[tuple[str, tuple[int, int, int]]] = [
    ("red", (220, 20, 20)),
    ("green", (20, 180, 20)),
    ("blue", (20, 40, 220)),
    ("yellow", (235, 235, 20)),
    ("purple", (150, 20, 200)),
    ("orange", (240, 140, 10)),
]


def image_color_name(index: int) -> str:
    """Return the dominant color name of the image at ``index``."""
    return _PALETTE[index % len(_PALETTE)][0]


@lru_cache(maxsize=4096)
def image_data_uri(index: int) -> str:
    """Build the deterministic test image at ``index`` as a PNG data URI.

    The image is a solid square of ``image_color_name(index)`` with a small
    deterministic index-dependent block pattern in one corner, keeping the
    dominant color intact while making every index's bytes (and therefore
    vLLM content hash) unique.

    Args:
        index: Image index; any non-negative integer.

    Returns:
        A ``data:image/png;base64,...`` URI.
    """
    _, rgb = _PALETTE[index % len(_PALETTE)]
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), rgb)
    # Deterministic per-index pattern: encode the index in a 8x8 grid of
    # slightly darker blocks in the top-left 96x96 corner.
    px = img.load()
    dark = tuple(max(0, c - 40) for c in rgb)
    for bit in range(24):
        if (index >> bit) & 1:
            bx, by = (bit % 8) * 12, (bit // 8) * 12
            for x in range(bx, bx + 12):
                for y in range(by, by + 12):
                    px[x, y] = dark
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@lru_cache(maxsize=64)
def video_data_uri(index: int) -> str:
    """Build the deterministic test video at ``index`` as an MP4 data URI.

    Every frame is a solid square of ``image_color_name(index)`` with the
    same index-encoding corner pattern as ``image_data_uri`` plus a small
    moving marker, so the clip is a real multi-frame video whose bytes (and
    therefore vLLM content hash) are unique per index while its dominant
    color stays a one-word semantic probe.

    Args:
        index: Video index; any non-negative integer.

    Returns:
        A ``data:video/mp4;base64,...`` URI.
    """
    # Third Party
    import cv2
    import numpy as np

    _, rgb = _PALETTE[index % len(_PALETTE)]
    dark = tuple(max(0, c - 40) for c in rgb)
    frames = []
    for t in range(VIDEO_FRAMES):
        frame = np.full((VIDEO_SIZE, VIDEO_SIZE, 3), rgb, dtype=np.uint8)
        for bit in range(24):
            if (index >> bit) & 1:
                bx, by = (bit % 8) * 8, (bit // 8) * 8
                frame[by : by + 8, bx : bx + 8] = dark
        # A moving marker so consecutive frames differ (a genuine video).
        mx = 8 * t
        frame[VIDEO_SIZE - 16 :, mx : mx + 16] = dark
        frames.append(frame[:, :, ::-1])  # RGB -> BGR for OpenCV
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            VIDEO_FPS,
            (VIDEO_SIZE, VIDEO_SIZE),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open an mp4 writer")
        for frame in frames:
            writer.write(frame)
        writer.release()
        with open(path, "rb") as f:
            payload = f.read()
    finally:
        os.unlink(path)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


AUDIO_SAMPLE_RATE = 16000
# Chosen from a measurement, not for convenience. Qwen3-Omni expands audio
# at a very steady 13.1-13.3 placeholder tokens per second (measured 20 /
# 53 / 79 / 105 / 131 tokens at 1.5 / 4 / 6 / 8 / 10 s). The isolation
# cases prove a hit did NOT reach the media by requiring a separation of
# ``Harness.image_span_margin`` = 4 chunks = 64 tokens, so the span has to
# be comfortably WIDER than that: at the 1.5 s used while designing the
# stimuli the span is only 20 tokens, narrower than the margin, and the
# assertion could never have been satisfied however correct the cache was.
# 8 s gives 105 tokens, about 6.5 chunks, leaving room for the span not
# being chunk-aligned.
AUDIO_SECONDS = 8.0
AUDIO_LEAD_SILENCE = 0.1
AUDIO_TONE_HZ = 440.0
AUDIO_RUMBLE_HZ = 60.0
AUDIO_WARBLE_CENTER_HZ = 600.0
AUDIO_WARBLE_DEPTH_HZ = 300.0
AUDIO_WARBLE_RATE_HZ = 3.0
AUDIO_BEEP_SECONDS = 0.28
AUDIO_BEEP_GAP_SECONDS = 0.18
AUDIO_EDGE_SECONDS = 0.006

# Five kinds, every one of them measured on the certification target
# (Qwen3-Omni-30B) as correctly named, stable across two passes, and
# answered differently from all the others. Two things are deliberately
# NOT in this list:
#
# silence   named "tone" by the 30B, stably -- which COLLIDES with the real
#           tone. A collision is the one failure a cross-item detector
#           cannot see through, since item A returning item B's cached
#           answer is invisible when both answer alike. (The 3B does name
#           silence correctly; the palette follows the model being
#           certified, not the most convenient one.)
# counting  "how many beeps" and both pitch-height and pitch-direction
#           labels were measured to collapse onto shared answers. Note the
#           distinction: BEEPING as a coarse kind is reliable, while the
#           NUMBER of beeps is not, so this list names the kind and never
#           the count.
_AUDIO_KINDS = ("tone", "noise", "beeping", "rumble", "warble")


def audio_kind_name(index: int) -> str:
    """Return the expected one-word kind of the audio clip at ``index``."""
    return _AUDIO_KINDS[index % len(_AUDIO_KINDS)]


def _lcg(seed: int):
    """Yield a deterministic pseudorandom stream, seeded by ``seed``.

    A local generator rather than ``random`` so a clip's bytes depend only
    on its index, never on global interpreter state or call order.
    """
    state = (seed * 2 + 1) & 0x7FFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state


def _fade_edges(values: list[float], edge: int) -> list[float]:
    """Apply raised-cosine fades in place and return the same list.

    A hard start or stop is a click, and a click is a different sound from
    the one being probed.
    """
    count = len(values)
    for i in range(min(edge, count // 2)):
        weight = 0.5 * (1.0 - math.cos(math.pi * i / edge))
        values[i] *= weight
        values[count - 1 - i] *= weight
    return values


def _audio_body(kind: str, count: int, rand) -> list[float]:
    """Build ``count`` samples of the named kind, in [-1.0, 1.0].

    Args:
        kind: One of ``_AUDIO_KINDS``.
        count: Number of samples to produce.
        rand: Pseudorandom stream from ``_lcg`` (used by ``noise``).

    Returns:
        The sample values, already faded at both ends.

    Raises:
        ValueError: If ``kind`` is not a known audio kind.
    """
    edge = max(1, int(AUDIO_EDGE_SECONDS * AUDIO_SAMPLE_RATE))
    step = 2.0 * math.pi / AUDIO_SAMPLE_RATE
    if kind == "tone":
        return _fade_edges(
            [0.6 * math.sin(step * AUDIO_TONE_HZ * i) for i in range(count)], edge
        )
    if kind == "rumble":
        return _fade_edges(
            [0.75 * math.sin(step * AUDIO_RUMBLE_HZ * i) for i in range(count)], edge
        )
    if kind == "noise":
        return _fade_edges(
            [0.35 * (next(rand) / 0x3FFFFFFF - 1.0) for i in range(count)], edge
        )
    if kind == "warble":
        # Frequency swept continuously; phase is accumulated so the sweep
        # stays continuous and the clip has no discontinuity to click at.
        values: list[float] = []
        phase = 0.0
        for i in range(count):
            freq = AUDIO_WARBLE_CENTER_HZ + AUDIO_WARBLE_DEPTH_HZ * math.sin(
                step * AUDIO_WARBLE_RATE_HZ * i
            )
            phase += step * freq
            values.append(0.6 * math.sin(phase))
        return _fade_edges(values, edge)
    if kind == "beeping":
        # Beeps repeat for the WHOLE clip rather than a fixed count: with a
        # fixed three, lengthening the clip to widen the placeholder span
        # would leave most of it trailing silence, and what the model calls
        # the clip could change under us. The count is never probed (beep
        # COUNTING was measured unreliable), only the kind, so filling the
        # duration is both safe and duration-independent.
        beep = int(AUDIO_BEEP_SECONDS * AUDIO_SAMPLE_RATE)
        gap = int(AUDIO_BEEP_GAP_SECONDS * AUDIO_SAMPLE_RATE)
        values = []
        while len(values) < count:
            values.extend(
                _fade_edges(
                    [0.6 * math.sin(step * AUDIO_TONE_HZ * i) for i in range(beep)],
                    edge,
                )
            )
            values.extend([0.0] * gap)
        return values[:count]
    raise ValueError(f"unknown audio kind: {kind}")


@lru_cache(maxsize=256)
def audio_data_uri(index: int) -> str:
    """Build the deterministic test clip at ``index`` as a WAV data URI.

    The clip's KIND cycles with ``index % 5`` and is what the semantic probe
    checks. Independently, every index gets unique BYTES (and therefore a
    unique vLLM ``mm_hash``) from a per-index dither of one least
    significant bit applied to every sample: at roughly -90 dBFS it cannot
    change which kind a listener reports, but it does mean two clips of the
    same kind are still two distinct cache entries. That matters because
    the isolation cases need same-answer/different-content pairs -- without
    the dither, two "tone" indices would be byte-identical and a false hit
    between them would be undetectable in principle.

    Duration is identical for every index on purpose: a per-index length
    would shift the placeholder span and quietly move the chunk boundaries
    that the boundary-phase cases are built to control.

    Args:
        index: Clip index; any non-negative integer.

    Returns:
        A ``data:audio/wav;base64,...`` URI of 16-bit mono PCM.
    """
    n_lead = int(AUDIO_SAMPLE_RATE * AUDIO_LEAD_SILENCE)
    n_body = int(AUDIO_SAMPLE_RATE * AUDIO_SECONDS)
    rand = _lcg(index + 1)
    body = _audio_body(audio_kind_name(index), n_body, rand)
    pcm = bytearray()
    for value in [0.0] * n_lead + body:
        sample = int(max(-1.0, min(1.0, value)) * 32767.0)
        # Bit 16, not bit 0: in a power-of-two-modulus LCG the low bit
        # alternates in a pattern that does NOT depend on the seed, so
        # ``& 1`` here gave every index an identical dither and left whole
        # kinds byte-identical across indices -- the exact collision this
        # dither exists to prevent.
        sample += 1 if (next(rand) >> 16) & 1 else -1
        pcm += max(-32768, min(32767, sample)).to_bytes(2, "little", signed=True)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(AUDIO_SAMPLE_RATE)
        out.writeframes(bytes(pcm))
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


COLOR_QUESTION = (
    "What is the dominant color of this image? Answer with exactly one word."
)
VIDEO_COLOR_QUESTION = (
    "What is the dominant color in this video? Answer with exactly one word."
)
MULTI_COLOR_QUESTION = (
    "List the dominant color of each image in the order given, "
    "separated by a comma. Answer with color words only."
)
# Every option is listed explicitly so the model picks from a fixed
# vocabulary; the wording is the one the palette was measured with.
AUDIO_KIND_QUESTION = (
    "Which of these best describes the audio: a steady musical tone, "
    "static noise, repeated beeping, a low rumble, or a warbling tone? "
    "Reply with one word: tone, noise, beeping, rumble, or warble."
)
TEXT_ONLY_QUESTION = "What is the capital of France? Answer with exactly one word."


@dataclass(frozen=True)
class MMRequest:
    """One request in the acceptance matrix.

    Attributes:
        key: Unique id, used to map to the baseline output.
        salt: Case-unique first words of the system message (cache isolation
            between cases).
        question: The user question text.
        image_indices: Indices of images attached, in order. Empty for
            text-only requests.
        expected_probe: Lowercase words that must ALL appear in the answer
            for the semantic probe to pass; empty tuple disables the probe.
        max_tokens: Generation budget.
        needs_baseline: Whether the baseline engine must run this request.
        video_indices: Indices of videos attached, in order (after images).
        ignore_eos: Force the full ``max_tokens`` decode (the preemption
            scenario needs guaranteed KV growth during decode).
        audio_indices: Indices of audio clips attached, in order (after
            images and videos). Audio probes are single-clip: naming two
            clips in order was measured at 0/9 correct, so a case with more
            than one clip has no usable semantic probe.
    """

    key: str
    salt: str
    question: str
    image_indices: tuple[int, ...]
    expected_probe: tuple[str, ...]
    max_tokens: int = 8
    needs_baseline: bool = True
    video_indices: tuple[int, ...] = ()
    ignore_eos: bool = False
    audio_indices: tuple[int, ...] = ()

    def messages(self) -> list[dict]:
        """Build the OpenAI-style chat messages for this request.

        The multimodal items come first, then the question. When the pad
        knobs are set (hybrid models, see ``pre_pad_words``), filler text
        surrounds the items so the prompt spans several whole KV blocks.
        """
        items: list[dict] = [
            {"type": "image_url", "image_url": {"url": image_data_uri(i)}}
            for i in self.image_indices
        ]
        items.extend(
            {"type": "video_url", "video_url": {"url": video_data_uri(i)}}
            for i in self.video_indices
        )
        items.extend(
            {"type": "audio_url", "audio_url": {"url": audio_data_uri(i)}}
            for i in self.audio_indices
        )
        mid_pad = mid_pad_words()
        content: list[dict] = []
        for position, item in enumerate(items):
            if position and mid_pad:
                content.append({"type": "text", "text": _filler(mid_pad)})
            content.append(item)
        post_pad = post_pad_words()
        question = (
            f"{_filler(post_pad)}\n{self.question}" if post_pad else self.question
        )
        content.append({"type": "text", "text": question})
        pre_pad = pre_pad_words()
        preamble = f" {_filler(pre_pad)}" if pre_pad else ""
        return [
            {
                "role": "system",
                "content": (
                    f"Session {self.salt}.{preamble} You are a concise assistant."
                ),
            },
            {"role": "user", "content": content},
        ]


def color_request(
    key: str, salt: str, image_index: int, question: str = COLOR_QUESTION
) -> MMRequest:
    """Build a single-image request probing the image's dominant color."""
    return MMRequest(
        key=key,
        salt=salt,
        question=question,
        image_indices=(image_index,),
        expected_probe=(image_color_name(image_index),),
    )


def multi_color_request(
    key: str, salt: str, image_indices: tuple[int, ...]
) -> MMRequest:
    """Build a multi-image request probing all dominant colors in order."""
    return MMRequest(
        key=key,
        salt=salt,
        question=MULTI_COLOR_QUESTION,
        image_indices=image_indices,
        expected_probe=tuple(image_color_name(i) for i in image_indices),
        max_tokens=16,
    )


def video_color_request(key: str, salt: str, video_index: int) -> MMRequest:
    """Build a single-video request probing the video's dominant color."""
    return MMRequest(
        key=key,
        salt=salt,
        question=VIDEO_COLOR_QUESTION,
        image_indices=(),
        expected_probe=(image_color_name(video_index),),
        video_indices=(video_index,),
    )


def audio_kind_request(key: str, salt: str, audio_index: int) -> MMRequest:
    """Build a single-clip request probing the clip's kind of sound."""
    return MMRequest(
        key=key,
        salt=salt,
        question=AUDIO_KIND_QUESTION,
        image_indices=(),
        expected_probe=(audio_kind_name(audio_index),),
        audio_indices=(audio_index,),
    )


def text_request(key: str, salt: str) -> MMRequest:
    """Build a text-only request with a fixed-answer probe."""
    return MMRequest(
        key=key,
        salt=salt,
        question=TEXT_ONLY_QUESTION,
        image_indices=(),
        expected_probe=("paris",),
    )


# Image index allocation: probe-critical cases use dedicated indices with
# distinct colors WITHIN each case; the pressure test uses indices >= 100,
# the preemption scenario indices >= 300, the capacity-eviction scenario
# indices >= 400.
PRESSURE_INDEX_BASE = 100
PREEMPTION_INDEX_BASE = 300
EVICTION_INDEX_BASE = 400

# Chunk-boundary phases: pad the salt with k extra words to shift where the
# image span falls relative to LMCache chunk boundaries (chunk_size=16).
BOUNDARY_PHASES = 16


def pre_pad_words() -> int:
    """Filler words between the case salt and the multimodal items.

    Set via ``LMCACHE_MM_E2E_PRE_PAD_WORDS`` by the conftest for
    Mamba/GDN hybrid models, whose hit granularity is the unified block
    size ``N`` (hundreds of tokens) rather than a 16-token chunk. This pad
    puts SHARED, cacheable blocks in front of the image so two requests
    that differ only in image content still have a common prefix to hit —
    without it, the first block already differs and every hit count
    collapses to zero. It follows the case salt, so case isolation (the
    first chunk differs per case) is unchanged.
    """
    return int(os.environ.get("LMCACHE_MM_E2E_PRE_PAD_WORDS", "0"))


def post_pad_words() -> int:
    """Filler words between the multimodal items and the question.

    Set via ``LMCACHE_MM_E2E_POST_PAD_WORDS`` for hybrid models. This is
    the load-bearing half of the hybrid prompt shape: it puts several
    whole blocks AFTER the image span, so that a request differing only in
    image content also differs in every one of those blocks. With the
    image in the last (partial) block instead, block-granular hit counts
    would be IDENTICAL for different images and the suite's primary
    cross-image detector would be blind.
    """
    return int(os.environ.get("LMCACHE_MM_E2E_POST_PAD_WORDS", "0"))


def mid_pad_words() -> int:
    """Filler words inserted between consecutive multimodal items.

    Set via ``LMCACHE_MM_E2E_MID_PAD_WORDS`` for hybrid models: it
    completes whole blocks after the FIRST image of a multi-image request,
    so the partial-sharing case (T2.2) can hit past that image instead of
    stopping at the shared pre-pad.
    """
    return int(os.environ.get("LMCACHE_MM_E2E_MID_PAD_WORDS", "0"))


def _filler(words: int) -> str:
    """A deterministic filler phrase of ``words`` ignorable pad words."""
    return "Ignore the following filler text. " + " ".join(["pad"] * words)


def boundary_step_words() -> int:
    """Words per T0.4 boundary phase (default 1 ~= 1 token per phase).

    With ``BOUNDARY_PHASES`` phases the sweep covers ``phases * step``
    tokens of alignment space; the conftest sets
    ``LMCACHE_MM_E2E_BOUNDARY_STEP`` to ``N // BOUNDARY_PHASES`` for a
    hybrid model so the sweep covers one full ``N``-token block period.
    """
    return int(os.environ.get("LMCACHE_MM_E2E_BOUNDARY_STEP", "1"))


def boundary_salt(phase: int) -> str:
    """Salt for chunk-boundary phase ``phase``, shifted by padding words."""
    return f"t04 phase {phase} " + " ".join(["pad"] * (phase * boundary_step_words()))


def catalog() -> dict[str, MMRequest]:
    """Enumerate every request the suite may send, keyed by request key.

    The baseline runner executes all requests with ``needs_baseline=True``
    once; tests replay subsets in their own order on the LMCache engine.
    """
    requests: list[MMRequest] = [
        # T0.1 / T0.3 / T1: two same-shape, different-color images.
        color_request("t01-A", "t01", 0),
        color_request("t01-B", "t01", 2),  # 0=red, 2=blue
        # T1.2: same image as t12-A, different follow-up question.
        color_request("t12-A", "t12", 1),
        MMRequest(
            key="t12-A-q2",
            salt="t12",
            question="Is this image mostly dark or mostly bright? One word.",
            image_indices=(1,),
            expected_probe=(),
        ),
        # T0.5: mixed text-only and multimodal traffic.
        text_request("t05-text", "t05"),
        color_request("t05-A", "t05", 4),
        color_request("t05-B", "t05", 5),
        # T2.1: two images, both orders.
        multi_color_request("t21-AB", "t21", (0, 2)),
        multi_color_request("t21-BA", "t21", (2, 0)),
        # T2.2: single image, then the same image plus a new one.
        color_request("t22-A", "t22", 1),
        multi_color_request("t22-AC", "t22", (1, 3)),
        # T0.8: concurrent batch traffic (entries also replayed singly).
        color_request("t08-A", "t08", 0),
        color_request("t08-B", "t08", 2),
        text_request("t08-text", "t08"),
    ]
    # T0.4: per-phase image pair (green=1, yellow=3).
    for phase in range(BOUNDARY_PHASES):
        salt = boundary_salt(phase)
        requests.append(color_request(f"t04-p{phase}-A", salt, 1))
        requests.append(color_request(f"t04-p{phase}-B", salt, 3))
    result = {r.key: r for r in requests}
    if len(result) != len(requests):
        raise ValueError("Duplicate request keys in catalog")
    return result


def long_prefix_color_request(
    key: str, salt: str, pad_words: int, image_index: int
) -> MMRequest:
    """Single-image color request with a long padded text prefix.

    Used by the chunked-prefill scenario: the pad pushes the image span deep
    enough into the prompt that a small ``max_num_batched_tokens`` budget
    places a scheduler-step boundary INSIDE the span. Verified against a
    config-matched plain-vLLM baseline: small models can misname colors
    behind long pad prefixes even WITHOUT LMCache, so a bare semantic probe
    would misattribute model weakness to the cache.

    Args:
        key: Unique request key.
        salt: Case-unique salt prefix (pad words are appended to it).
        pad_words: Number of filler words appended to the salt.
        image_index: Index of the attached image.

    Returns:
        The built request.
    """
    padded = f"{salt} " + " ".join(["pad"] * pad_words)
    return MMRequest(
        key=key,
        salt=padded,
        question=COLOR_QUESTION,
        image_indices=(image_index,),
        expected_probe=(image_color_name(image_index),),
        needs_baseline=True,
    )


def eviction_requests(n: int) -> list[MMRequest]:
    """Build the capacity-eviction scenario requests: ``n`` distinct images.

    Verified against a config-matched plain-vLLM baseline (with probe
    rescue) plus re-run-vs-first-run equivalence.

    Args:
        n: Number of distinct images; sized to overflow the tiny cache.

    Returns:
        The built requests.
    """
    return [
        MMRequest(
            key=f"t10-{i}",
            salt="t10",
            question=COLOR_QUESTION,
            image_indices=(EVICTION_INDEX_BASE + i,),
            expected_probe=(image_color_name(EVICTION_INDEX_BASE + i),),
            needs_baseline=True,
        )
        for i in range(n)
    ]


def video_requests() -> dict[str, MMRequest]:
    """T2.3 video requests, keyed by request key.

    Kept OUT of ``catalog()`` because they are only valid (and only get
    baselines) for models whose spec declares the ``video`` modality; an
    image-only model's baseline engine would reject the video input.
    """
    requests = [
        # Same-shape, different-color videos behind an identical prompt.
        video_color_request("t23-A", "t23", 0),  # red
        video_color_request("t23-B", "t23", 2),  # blue
    ]
    return {r.key: r for r in requests}


def audio_requests() -> dict[str, MMRequest]:
    """T2.4 audio requests, keyed by request key.

    Kept OUT of ``catalog()`` for the same reason as ``video_requests``:
    they are only valid, and only get baselines, for a spec that declares
    the ``audio`` modality; another model's baseline engine would reject
    the input.

    The two clips are different KINDS, not merely different bytes, so the
    semantic probe can tell them apart -- a false hit shows up as clip B
    answering with clip A's kind. Same-kind pairs would still have distinct
    bytes (see ``audio_data_uri``) but no observable answer difference, and
    a detector that cannot see the collision it is looking for proves
    nothing.
    """
    requests = [
        # tone (index 0) and beeping (index 2) behind an identical prompt.
        audio_kind_request("t24-A", "t24", 0),
        audio_kind_request("t24-B", "t24", 2),
    ]
    return {r.key: r for r in requests}


def preemption_requests(n: int, max_tokens: int) -> list[MMRequest]:
    """Build the preemption scenario requests: ``n`` distinct images.

    Long decodes (``max_tokens``) grow the KV of every running request until
    the deliberately tiny GPU block pool overflows and the scheduler
    preempts; distinct colors expose any cross-request contamination on the
    recompute path.

    Args:
        n: Number of distinct-image requests (scheduled as one batch).
        max_tokens: Decode budget per request; must be large enough that
            decode growth overflows the block pool.

    Returns:
        The built requests.
    """
    return [
        MMRequest(
            key=f"t11-{i}",
            salt="t11",
            question=COLOR_QUESTION,
            image_indices=(PREEMPTION_INDEX_BASE + i,),
            expected_probe=(image_color_name(PREEMPTION_INDEX_BASE + i),),
            max_tokens=max_tokens,
            needs_baseline=True,
            ignore_eos=True,
        )
        for i in range(n)
    ]


def pressure_requests(n: int) -> list[MMRequest]:
    """Build the T0.2 collision-pressure requests: ``n`` distinct images.

    These are exempt from baseline comparison (pass-2 outputs are compared
    against pass-1 outputs instead), so ``needs_baseline`` is False. The
    color probe backs the pass-2 replay check like everywhere else: the
    miss and hit passes are different numeric regimes, and a model that
    trails its answer with degenerate repetition diverges byte-wise while
    still naming the right color; a false hit names the OTHER image's
    color and fails the probe hard.
    """
    return [
        MMRequest(
            key=f"t02-{i}",
            salt="t02",
            question=COLOR_QUESTION,
            image_indices=(PRESSURE_INDEX_BASE + i,),
            expected_probe=(image_color_name(PRESSURE_INDEX_BASE + i),),
            needs_baseline=False,
        )
        for i in range(n)
    ]
