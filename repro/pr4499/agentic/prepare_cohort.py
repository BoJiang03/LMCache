#!/usr/bin/env python
"""Stage B of cohort preparation: pick the fixed agentic session cohort.

Selection is deterministic -- file order, one trajectory per issue, fixed
token window -- so the same JSONL always yields the same cohort, and both
policies under test replay exactly the same requests.

What the replay models. One session is one real SWE-agent run. Step `k`
sends the whole conversation so far and asks for the next action:

    step 0:  [system, issue]
    step 1:  [system, issue, action_0, observation_0]
    step k:  [system, issue, action_0, observation_0, ..., observation_k-1]

so every step's prompt is a *prefix extension* of the step before it, which
is the property that makes agent serving a prefix-cache workload. The
recorded action is replayed rather than the model's own output: the prompt
sequence then depends only on the dataset, not on sampling, so a policy A/B
compares two identical request streams.

Guards enforced here, because a violated one silently destroys the effect
being measured:

- role pattern is exactly system, user, (assistant, user)*;
- at least `steps` assistant turns, and no truncation inside a step;
- the step-`steps` prompt lands inside the configured token window, which
  is what fixes the per-session KV working set;
- token-level prefix stability: the tokens of step k, minus the generation
  prompt the template appends, are a prefix of the tokens of step k+1.

Environment:
    AGENTIC_RAW         input JSONL from extract_trajectories.py
    AGENTIC_COHORT_OUT  output cohort JSON
    AGENTIC_STEPS       replayed steps per session (default 12)
    AGENTIC_MIN_TOKENS  minimum final-step prompt tokens (default 8000)
    AGENTIC_MAX_TOKENS  maximum final-step prompt tokens (default 22000)
    AGENTIC_SESSIONS    cohort size to select (default 48)
    SMOKE_MODEL         tokenizer/model id (default Qwen/Qwen3-8B)
    HF_HUB_CACHE        model cache directory
"""

import hashlib
import json
import os
import statistics
import sys

from transformers import AutoTokenizer

RAW = os.environ.get("AGENTIC_RAW", "")
OUT = os.environ.get("AGENTIC_COHORT_OUT", "")
STEPS = int(os.environ.get("AGENTIC_STEPS", "12"))
MIN_TOKENS = int(os.environ.get("AGENTIC_MIN_TOKENS", "8000"))
MAX_TOKENS = int(os.environ.get("AGENTIC_MAX_TOKENS", "22000"))
SESSIONS = int(os.environ.get("AGENTIC_SESSIONS", "48"))
MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-8B")

#: Tokens of slack allowed between the end of step k's cached prefix and the
#: start of step k+1's divergence. The chat template ends a prompt with the
#: generation prompt (`<|im_start|>assistant\n` plus, for this model with
#: thinking disabled, an empty think block); those tokens are not part of the
#: next step's prompt, and they are the only difference tolerated here.
PREFIX_SLACK = 16


def _render(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """Tokenize one step's prompt exactly as the server will.

    Args:
        tokenizer: The serving tokenizer.
        messages: The chat messages of this step.

    Returns:
        The prompt's token ids.
    """
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def _common_prefix(left: list[int], right: list[int]) -> int:
    """Length of the longest common prefix of two token lists."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _shaped(messages: list[dict[str, str]]) -> bool:
    """Whether a trajectory has the system, user, (assistant, user)* shape."""
    if len(messages) < 2 * STEPS:
        return False
    if messages[0]["role"] != "system" or messages[1]["role"] != "user":
        return False
    for index in range(2, 2 * STEPS):
        expected = "assistant" if index % 2 == 0 else "user"
        if messages[index]["role"] != expected:
            return False
    return all(message["content"].strip() for message in messages[: 2 * STEPS])


def main() -> int:
    """Select the cohort and write it with its token profile.

    Returns:
        Process exit code; 1 if the cohort could not be filled.
    """
    if not RAW or not OUT:
        print(__doc__)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=os.environ.get("HF_HUB_CACHE"))
    seen: set[str] = set()
    sessions: list[dict] = []
    scanned = 0
    with open(RAW) as fh:
        for line in fh:
            if len(sessions) >= SESSIONS:
                break
            scanned += 1
            row = json.loads(line)
            if row["instance_id"] in seen:
                continue
            messages = row["messages"]
            if not _shaped(messages):
                continue
            kept = messages[: 2 * STEPS]
            final = _render(tokenizer, kept)
            if not MIN_TOKENS <= len(final) <= MAX_TOKENS:
                continue
            seen.add(row["instance_id"])
            step_tokens = []
            previous: list[int] = []
            broken = 0
            for step in range(STEPS):
                current = _render(tokenizer, kept[: 2 + 2 * step])
                step_tokens.append(len(current))
                if previous:
                    shared = _common_prefix(previous, current)
                    if shared < len(previous) - PREFIX_SLACK:
                        broken += 1
                previous = current
            if broken:
                print(
                    f"[cohort] skip {row['instance_id']}: {broken} unstable steps",
                    file=sys.stderr,
                )
                continue
            sessions.append(
                {
                    "instance_id": row["instance_id"],
                    "source_model": row["model"],
                    "exit_status": row["exit_status"],
                    "step_prompt_tokens": step_tokens,
                    "messages": kept,
                }
            )
    if len(sessions) < SESSIONS:
        print(f"[cohort] only {len(sessions)} of {SESSIONS} sessions", file=sys.stderr)
        return 1
    finals = [session["step_prompt_tokens"][-1] for session in sessions]
    firsts = [session["step_prompt_tokens"][0] for session in sessions]
    digest = hashlib.sha256(
        json.dumps([s["messages"] for s in sessions], sort_keys=True).encode()
    ).hexdigest()
    document = {
        "source_jsonl": RAW,
        "model": MODEL,
        "steps": STEPS,
        "sessions": len(sessions),
        "scanned_rows": scanned,
        "cohort_sha256": digest,
        "final_prompt_tokens": {
            "min": min(finals),
            "p50": int(statistics.median(finals)),
            "max": max(finals),
            "sum": sum(finals),
        },
        "first_prompt_tokens": {
            "min": min(firsts),
            "p50": int(statistics.median(firsts)),
            "max": max(firsts),
        },
        "cohort": sessions,
    }
    with open(OUT, "w") as fh:
        json.dump(document, fh)
    print(
        f"[cohort] {len(sessions)} sessions, {STEPS} steps, "
        f"final prompt tokens min={min(finals)} p50={int(statistics.median(finals))} "
        f"max={max(finals)} sum={sum(finals)}"
    )
    print(f"[cohort] sha256 {digest}")
    print(f"[cohort] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
