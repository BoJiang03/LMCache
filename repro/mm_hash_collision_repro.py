# SPDX-License-Identifier: Apache-2.0
"""Reproduction for LMCache issue #3301: 16-bit mm_hash truncation causes
silent cross-image KV cache hits (串图).

Background
----------
LMCache (pre-fix) substituted multimodal placeholder token IDs with
``int(mm_hash_hex, 16) & 0xFFFF`` -- 16 bits of identity per image. By the
birthday bound, ~300 distinct same-shape images give ~50% probability that
two of them share a truncated value, and colliding images share ALL their
KV cache keys: the second image silently serves the first image's KV.

What this script does
---------------------
1. Launches a vLLM engine with the LMCache connector (single process) and
   sends N distinct solid-color images through it, recording the real
   multimodal identifiers the connector sees (via a read-only wrapper around
   ``apply_mm_hashes_to_token_ids``).
2. Finds a pair of images whose identifiers collide under 16-bit truncation
   but whose dominant colors differ.
3. Asks the engine for the color of each image in the pair. On a buggy
   build, the second answer names the FIRST image's color (false hit); on a
   fixed build both answers are correct.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python repro/mm_hash_collision_repro.py \
        [--model Qwen/Qwen2.5-VL-3B-Instruct] [--num-images 800]

Requires a GPU and the model weights. Exit code 1 = false hit reproduced,
0 = no false hit observed.
"""

# Standard
import argparse
import base64
import io
import os
import pathlib
import sys

# Pin THIS repo's lmcache package: a stray editable install could otherwise
# resolve `import lmcache` to a different source tree.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RECORDED_IDENTIFIERS: list[str] = []

PALETTE = [
    ("red", (220, 20, 20)),
    ("green", (20, 180, 20)),
    ("blue", (20, 40, 220)),
    ("yellow", (235, 235, 20)),
    ("purple", (150, 20, 200)),
    ("orange", (240, 140, 10)),
]

COLOR_QUESTION = (
    "What is the dominant color of this image? Answer with exactly one word."
)


def image_color_name(index: int) -> str:
    return PALETTE[index % len(PALETTE)][0]


def image_data_uri(index: int) -> str:
    """Deterministic 448x448 solid-color image with an index pattern."""
    # Third Party
    from PIL import Image

    _, rgb = PALETTE[index % len(PALETTE)]
    img = Image.new("RGB", (448, 448), rgb)
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
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def messages_for(index: int, question: str) -> list[dict]:
    return [
        {"role": "system", "content": "Repro session. You are a concise assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri(index)}},
                {"type": "text", "text": question},
            ],
        },
    ]


def truncate_16bit(identifier: str) -> int:
    """The historical key derivation: 16 bits of the identifier."""
    # Standard
    import hashlib
    import string

    s = identifier.strip()
    hex_part = s[2:] if s.lower().startswith("0x") else s
    if hex_part and all(c in string.hexdigits for c in hex_part):
        return int(hex_part, 16) & 0xFFFF
    return int.from_bytes(
        hashlib.sha256(s.encode("utf-8")).digest()[:2], byteorder="big"
    )


def install_identifier_recorder() -> None:
    """Wrap apply_mm_hashes_to_token_ids to record identifiers (read-only).

    Must run BEFORE the engine (and thus the LMCache connector adapter) is
    imported, so the adapter binds the wrapped function.
    """
    # First Party
    import lmcache.integration.vllm.utils as lmc_utils

    original = lmc_utils.apply_mm_hashes_to_token_ids

    def recording(token_ids, mm_hashes, mm_positions):
        RECORDED_IDENTIFIERS.extend(mm_hashes)
        return original(token_ids, mm_hashes, mm_positions)

    lmc_utils.apply_mm_hashes_to_token_ids = recording


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--num-images",
        type=int,
        default=800,
        help="images to record; 800 gives >99%% collision probability "
        "in a 16-bit space",
    )
    args = parser.parse_args()

    # Single process so the recorder wrapper sees the connector's calls.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("LMCACHE_CHUNK_SIZE", "16")
    os.environ.setdefault("LMCACHE_LOCAL_CPU", "True")
    os.environ.setdefault("LMCACHE_MAX_LOCAL_CPU_SIZE", "40")

    install_identifier_recorder()

    # Third Party
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    llm = LLM(
        model=args.model,
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheConnectorV1", kv_role="kv_both"
        ),
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    sp_record = SamplingParams(temperature=0.0, max_tokens=1)
    sp_probe = SamplingParams(temperature=0.0, max_tokens=8)

    # Stage 1: record identifiers image by image (order = image index).
    print(f"[stage 1] recording identifiers for {args.num_images} images ...")
    identifier_of: dict[int, str] = {}
    for i in range(args.num_images):
        before = len(RECORDED_IDENTIFIERS)
        llm.chat(
            messages_for(i, "Describe."), sampling_params=sp_record, use_tqdm=False
        )
        new = RECORDED_IDENTIFIERS[before:]
        if new:
            identifier_of[i] = new[0]
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.num_images} recorded")

    if not identifier_of:
        print("ERROR: no identifiers recorded -- the connector path changed?")
        return 2

    # Stage 2: find a 16-bit collision between images of different colors.
    by_bucket: dict[int, int] = {}
    pair: tuple[int, int] | None = None
    for i, ident in identifier_of.items():
        bucket = truncate_16bit(ident)
        j = by_bucket.get(bucket)
        if j is not None and image_color_name(i) != image_color_name(j):
            pair = (j, i)
            break
        by_bucket.setdefault(bucket, i)

    if pair is None:
        print(
            f"[stage 2] no different-color 16-bit collision among "
            f"{len(identifier_of)} images; rerun with a larger --num-images."
        )
        return 0

    x, y = pair
    print(
        f"[stage 2] collision: image {x} ({image_color_name(x)}, "
        f"id={identifier_of[x]}) vs image {y} ({image_color_name(y)}, "
        f"id={identifier_of[y]}) -- both truncate to "
        f"0x{truncate_16bit(identifier_of[x]):04x}"
    )

    # Stage 3: probe both images. Their KV was stored in stage 1; on a buggy
    # build both images share cache keys, so image y is answered from image
    # x's KV.
    ans_x = (
        llm.chat(
            messages_for(x, COLOR_QUESTION), sampling_params=sp_probe, use_tqdm=False
        )[0]
        .outputs[0]
        .text
    )
    ans_y = (
        llm.chat(
            messages_for(y, COLOR_QUESTION), sampling_params=sp_probe, use_tqdm=False
        )[0]
        .outputs[0]
        .text
    )
    color_x, color_y = image_color_name(x), image_color_name(y)
    print(f"[stage 3] image {x} is {color_x}, model says: {ans_x!r}")
    print(f"[stage 3] image {y} is {color_y}, model says: {ans_y!r}")

    if color_y in ans_y.lower():
        print("VERDICT: no false hit -- each image answered from its own KV.")
        return 0
    if color_x in ans_y.lower():
        print(
            "VERDICT: FALSE HIT REPRODUCED -- the second image was answered "
            "from the FIRST image's KV cache (issue #3301)."
        )
        return 1
    print("VERDICT: inconclusive -- neither color named; inspect manually.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
