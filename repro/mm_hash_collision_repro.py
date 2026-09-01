# SPDX-License-Identifier: Apache-2.0
"""Reproduction for LMCache issue #3301: 16-bit mm_hash truncation causes
silent cross-image KV cache hits.

Background
----------
LMCache (pre-fix) substituted multimodal placeholder token IDs with
``int(mm_hash_hex, 16) & 0xFFFF`` -- 16 bits of identity per image, written
across the whole placeholder span. By the birthday bound, ~300 distinct
same-shape images give ~50% probability that two of them share the
truncated value, and colliding images then share ALL of their KV cache
keys: the second image is silently served the first image's KV.

Deployment path
---------------
This runs the multi-process path: a real ``lmcache.v1.multiprocess``
cache server subprocess plus a vLLM engine driving ``LMCacheMPConnector``.
That is the path LMCache supports, and it is where the connector-side
keying this reproduces actually runs.

What this script does
---------------------
1. Starts an MP cache server and a vLLM engine wired to it, then sends N
   distinct solid-color images through, recording the real multimodal
   identifiers the connector sees (a read-only wrapper around
   ``apply_mm_hashes_to_token_ids``).
2. Finds a pair of images whose identifiers collide under 16-bit
   truncation but whose dominant colors differ.
3. Asks the engine for the color of each image in the pair. On a buggy
   build one of the two is answered with the OTHER image's color, because
   both images share every cache key and whichever stored last owns the
   entry. On a fixed build both answers are correct.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python repro/mm_hash_collision_repro.py \
        [--model Qwen/Qwen2.5-VL-3B-Instruct] [--num-images 800]

Requires a GPU and the model weights. Exit code 1 = false hit reproduced,
0 = no false hit observed, 2 = the run could not conclude.
"""

# Standard
import argparse
import base64
import io
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Pin THIS repo's lmcache package: a stray editable install could otherwise
# resolve `import lmcache` to a different source tree.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RECORDED_IDENTIFIERS: list[str] = []

# The engine and the server must agree on the chunk size; 16 tokens keeps
# every image span many chunks wide, so a collision is unmistakable.
CHUNK_SIZE = 16
# Seconds to let asynchronous stores commit before the probe reads them
# back. Without it the probe can outrun stage 1's in-flight stores and
# report a miss that is only a race.
STORE_COMMIT_GRACE_S = 5.0

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
    """Return the dominant color name of the image at ``index``."""
    return PALETTE[index % len(PALETTE)][0]


def image_data_uri(index: int) -> str:
    """Build the deterministic 448x448 test image at ``index``.

    The image is a solid square of ``image_color_name(index)`` with the
    index encoded as a pattern of darker blocks in one corner, so the
    dominant color stays intact while every index's bytes -- and therefore
    vLLM's content hash -- are unique.

    Args:
        index: Image index; any non-negative integer.

    Returns:
        A ``data:image/png;base64,...`` URI.
    """
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
    """Build a one-image conversation for ``index``."""
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
    """Wrap ``apply_mm_hashes_to_token_ids`` to record identifiers.

    Read-only: the wrapper appends to ``RECORDED_IDENTIFIERS`` and defers
    to the original. It must run BEFORE the MP connector's modules are
    imported, because they bind the function by name at import time.
    """
    # First Party
    import lmcache.integration.vllm.utils as lmc_utils

    original = lmc_utils.apply_mm_hashes_to_token_ids

    def recording(token_ids, mm_hashes, mm_positions):
        RECORDED_IDENTIFIERS.extend(mm_hashes)
        return original(token_ids, mm_hashes, mm_positions)

    lmc_utils.apply_mm_hashes_to_token_ids = recording


def start_cache_server(
    zmq_port: int, http_port: int, l1_size_gb: float, log_path: pathlib.Path
) -> subprocess.Popen:
    """Launch an LMCache MP cache server and wait until it is healthy.

    Args:
        zmq_port: ZMQ port the connector will talk to.
        http_port: HTTP port carrying the health endpoint.
        l1_size_gb: Host memory pool size. Must be large enough to hold
            every image's KV, or eviction hides the collision behind an
            ordinary miss.
        log_path: File capturing the server's stdout and stderr.

    Returns:
        The running server process.

    Raises:
        RuntimeError: If the server does not become healthy in 120s.
    """
    # `-m` resolves through the child's own sys.path, where an editable
    # install would otherwise win over this repo.
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_REPO_ROOT}:{existing}" if existing else str(_REPO_ROOT)
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.multiprocess.http_server",
        "--port",
        str(zmq_port),
        "--http-port",
        str(http_port),
        "--chunk-size",
        str(CHUNK_SIZE),
        "--l1-size-gb",
        str(l1_size_gb),
        "--eviction-policy",
        "LRU",
    ]
    log_file = open(log_path, "w")
    proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    url = f"http://localhost:{http_port}/healthcheck"
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return proc
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    proc.terminate()
    raise RuntimeError(
        f"MP cache server did not become healthy; log tail:\n"
        f"{log_path.read_text()[-2000:]}"
    )


def build_engine(model: str, zmq_port: int):
    """Build a vLLM engine wired to the MP cache server on ``zmq_port``."""
    # Third Party
    from vllm import LLM
    from vllm.config import KVTransferConfig

    return LLM(
        model=model,
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_connector_module_path="lmcache.integration.vllm.lmcache_mp_connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": zmq_port,
            },
        ),
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        enable_prefix_caching=False,
    )


def find_colliding_pair(identifier_of: dict[int, str]) -> tuple[int, int] | None:
    """Return two image indices that collide under 16-bit truncation.

    Args:
        identifier_of: Image index to the identifier the connector saw.

    Returns:
        ``(first, second)`` in submission order, both of a different
        dominant color so the answer itself reveals a false hit; ``None``
        if no such pair exists in this sample.
    """
    by_bucket: dict[int, int] = {}
    for index in sorted(identifier_of):
        bucket = truncate_16bit(identifier_of[index])
        earlier = by_bucket.get(bucket)
        if earlier is not None and image_color_name(index) != image_color_name(earlier):
            return earlier, index
        by_bucket.setdefault(bucket, index)
    return None


def main() -> int:
    """Run the reproduction; see the module docstring for exit codes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--num-images",
        type=int,
        default=800,
        help="images to record; 800 gives >99%% collision probability "
        "in a 16-bit space",
    )
    parser.add_argument(
        "--l1-size-gb",
        type=float,
        default=16.0,
        help="cache server host memory pool; must hold every image's KV",
    )
    parser.add_argument("--zmq-port", type=int, default=0)
    parser.add_argument("--server-log", default="mm_hash_collision_repro_server.log")
    args = parser.parse_args()

    zmq_port = args.zmq_port or 25000 + (os.getpid() % 5000)
    http_port = zmq_port + 5000

    # The connector's scheduler side must run in this process so the
    # recorder sees its calls.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    install_identifier_recorder()

    log_path = pathlib.Path(args.server_log)
    print(
        f"[setup] starting MP cache server on zmq {zmq_port}, http {http_port}",
        flush=True,
    )
    server = start_cache_server(zmq_port, http_port, args.l1_size_gb, log_path)
    try:
        # Third Party
        from vllm import SamplingParams

        llm = build_engine(args.model, zmq_port)
        sp_record = SamplingParams(temperature=0.0, max_tokens=1)
        sp_probe = SamplingParams(temperature=0.0, max_tokens=8)

        # Stage 1: record identifiers image by image (order = image index).
        print(
            f"[stage 1] recording identifiers for {args.num_images} images ...",
            flush=True,
        )
        identifier_of: dict[int, str] = {}
        for i in range(args.num_images):
            before = len(RECORDED_IDENTIFIERS)
            llm.chat(
                messages_for(i, "Describe."),
                sampling_params=sp_record,
                use_tqdm=False,
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
        pair = find_colliding_pair(identifier_of)
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

        # Stage 3: probe both images. Their KV was stored in stage 1; on a
        # buggy build the two share every cache key, so whichever stored
        # last owns the entry and the other one is answered from it.
        time.sleep(STORE_COMMIT_GRACE_S)
        answers = {}
        for index in (x, y):
            answers[index] = (
                llm.chat(
                    messages_for(index, COLOR_QUESTION),
                    sampling_params=sp_probe,
                    use_tqdm=False,
                )[0]
                .outputs[0]
                .text
            )
            print(
                f"[stage 3] image {index} is {image_color_name(index)}, "
                f"model says: {answers[index]!r}"
            )

        for index, other in ((x, y), (y, x)):
            said = answers[index].lower()
            if image_color_name(index) in said:
                continue
            if image_color_name(other) in said:
                print(
                    f"VERDICT: FALSE HIT REPRODUCED -- image {index} was "
                    f"answered from image {other}'s KV cache (issue #3301)."
                )
                return 1
            print(
                f"VERDICT: inconclusive -- image {index} named neither color; "
                f"inspect manually."
            )
            return 2
        print("VERDICT: no false hit -- each image answered from its own KV.")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    sys.exit(main())
