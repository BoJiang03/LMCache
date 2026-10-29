"""Send A, base, B, A on one prompt; report LMCache hits and output match.

Usage: client.py <vllm_port> <baseline_port> <base_model> <results.json>
"""

import json
import random
import sys
import time
import urllib.request

PORT, BASELINE_PORT = int(sys.argv[1]), int(sys.argv[2])
BASE, OUT = sys.argv[3], sys.argv[4]
PROMPT_LEN = 4096  # multiple of chunk_size 256, so every token is cacheable
rng = random.Random(0)
PROMPT = [rng.randrange(1000, 30000) for _ in range(PROMPT_LEN)]
trigger_id = 0


def post(port, path, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def complete(port, model, prompt):
    body = {"model": model, "prompt": prompt, "max_tokens": 32, "temperature": 0}
    return json.loads(post(port, "/v1/completions", body))["choices"][0]["text"]


def counter(name):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics") as r:
        text = r.read().decode()
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith(name + "{") or line.startswith(name + " ")
    )


def trigger_drain():
    """Lazy offload stores only on a later token-producing step, so send a
    short unrelated request and give the store time to land."""
    global trigger_id
    trigger_id += 1
    complete(PORT, BASE, [50 + trigger_id] * 16)
    time.sleep(3)


results = []
for label, model in (
    ("A", "lora-a"),
    ("base", BASE),
    ("B", "lora-b"),
    ("A-again", "lora-a"),
):
    # vLLM's own prefix cache would serve A-again before LMCache is asked.
    post(PORT, "/reset_prefix_cache")
    ext0 = counter("vllm:external_prefix_cache_hits_total")
    loc0 = counter("vllm:prefix_cache_hits_total")
    text = complete(PORT, model, PROMPT)
    ext = counter("vllm:external_prefix_cache_hits_total") - ext0
    loc = counter("vllm:prefix_cache_hits_total") - loc0
    ref = complete(BASELINE_PORT, model, PROMPT)
    row = {
        "req": label,
        "model": model if model != BASE else "base",
        "lmcache_hit_tokens": int(ext),
        "vllm_local_hit_tokens": int(loc),
        "output_matches_reference": text == ref,
        "text": text[:60],
        "ref": ref[:60],
    }
    results.append(row)
    print(json.dumps(row), flush=True)
    trigger_drain()

json.dump(results, open(OUT, "w"), indent=1)
