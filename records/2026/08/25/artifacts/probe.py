"""Fire N distinct long prompts at the server, purely to generate store traffic.

The question is only how many of the resulting KV store batches the MP server
completes, so this deliberately avoids aiperf: no trajectory replay, no warmup,
no think time -- just enough distinct long prefixes to make the connector store.
"""
import json, sys, time, urllib.request, random

PORT, N, TOK = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
random.seed(7)
WORDS = [f"tok{i:05d}" for i in range(4000)]

def post(path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

ok = 0
for i in range(N):
    prompt = f"session-{i} " + " ".join(random.choice(WORDS) for _ in range(TOK))
    t = time.time()
    try:
        r = post("/v1/completions", {
            "model": "agentx", "prompt": prompt, "max_tokens": 8, "temperature": 0.0,
        })
        ok += 1
        print(f"  req {i}: {r['usage']['prompt_tokens']} prompt tok, {time.time()-t:.1f}s", flush=True)
    except Exception as exc:
        print(f"  req {i}: FAILED {exc}", flush=True)
print(f"{ok}/{N} completed")
