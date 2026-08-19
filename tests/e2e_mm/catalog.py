# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic images and request builders for the MM suite.

Images are solid-color squares with a deterministic per-index pattern, so
that (a) every index yields distinct bytes (distinct vLLM ``mm_hash``), and
(b) the dominant color is a one-word semantic probe the model can answer.

Every test case uses a unique ``salt`` as the first words of its system
message, so the very first token chunk already differs between cases and
cases cannot hit each other's cache entries.
"""

# Standard
from dataclasses import dataclass
from functools import lru_cache
import base64
import io

# Third Party
from PIL import Image

IMAGE_SIZE = 448

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


COLOR_QUESTION = (
    "What is the dominant color of this image? Answer with exactly one word."
)
MULTI_COLOR_QUESTION = (
    "List the dominant color of each image in the order given, "
    "separated by a comma. Answer with color words only."
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
    """

    key: str
    salt: str
    question: str
    image_indices: tuple[int, ...]
    expected_probe: tuple[str, ...]
    max_tokens: int = 8
    needs_baseline: bool = True

    def messages(self) -> list[dict]:
        """Build the OpenAI-style chat messages for this request."""
        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": image_data_uri(i)}}
            for i in self.image_indices
        ]
        content.append({"type": "text", "text": self.question})
        return [
            {
                "role": "system",
                "content": f"Session {self.salt}. You are a concise assistant.",
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
# the capacity-eviction scenario indices >= 400.
PRESSURE_INDEX_BASE = 100
EVICTION_INDEX_BASE = 400

# Chunk-boundary phases: pad the salt with k extra words to shift where the
# image span falls relative to LMCache chunk boundaries (chunk_size=16).
BOUNDARY_PHASES = 16


def boundary_salt(phase: int) -> str:
    """Salt for chunk-boundary phase ``phase``, padded with ``phase`` words."""
    return f"t04 phase {phase} " + " ".join(["pad"] * phase)


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


def pressure_requests(n: int) -> list[MMRequest]:
    """Build the T0.2 collision-pressure requests: ``n`` distinct images.

    These are exempt from baseline comparison (pass-2 outputs are compared
    against pass-1 outputs instead), so ``needs_baseline`` is False.
    """
    return [
        MMRequest(
            key=f"t02-{i}",
            salt="t02",
            question=COLOR_QUESTION,
            image_indices=(PRESSURE_INDEX_BASE + i,),
            expected_probe=(),
            needs_baseline=False,
        )
        for i in range(n)
    ]
