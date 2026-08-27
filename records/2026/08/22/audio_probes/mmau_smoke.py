# SPDX-License-Identifier: Apache-2.0
"""MMAU prompt-shape and parse smoke on plain vLLM (no LMCache).

The MME parity gate rests on answers that PARSE: if the verdict never lands
inside the decode budget, every pass parses to '' and the flip/score gates
pass while measuring nothing (that is what mme_min_parse_ratio exists to
catch). MMAU is four-way multiple choice rather than yes/no, so before
wiring it into benchmark_parity the prompt shape has to be shown to produce
a parseable choice, and the model's baseline accuracy has to be known --
a gate calibrated against an unknown baseline is not a gate.

Reports per item: the parsed letter, whether it matches ground truth, and
whether a second identical pass gives the same text (determinism is what
makes the byte-equality oracle usable at all).

usage: python mmau_smoke.py <hf_id> <n_items> <out_json>
"""

import base64
import glob
import json
import os
import pathlib
import re
import sys

hf_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-Omni-30B-A3B-Instruct"
n_items = int(sys.argv[2]) if len(sys.argv) > 2 else 40
out_json = sys.argv[3] if len(sys.argv) > 3 else "mmau_smoke.json"

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HUB_CACHE", "/raid/data/hub")
os.environ.setdefault("PYTHONHASHSEED", "0")
venv_include = pathlib.Path(sys.prefix) / "include"
if (venv_include / "Python.h").exists():
    os.environ["CPATH"] = f"{venv_include}:{os.environ.get('CPATH', '')}"

# MMAU test-mini holds 970 four-choice rows, 20 five-choice and 10 two-choice.
# Five letters cover all of them, so no row is dropped for its option count.
LETTERS = "ABCDE"

_SHARD_GLOBS = (
    "/raid/data/hub/datasets--TwinkStart--MMAU/snapshots/*/data/test_mini-*.parquet",
    "/home/bo/.cache/huggingface/hub/datasets--TwinkStart--MMAU/"
    "snapshots/*/data/test_mini-*.parquet",
)
_META_COLUMNS = ["id", "question", "choices", "answer", "task", "difficulty"]


def _shards() -> list[str]:
    for pattern in _SHARD_GLOBS:
        found = sorted(glob.glob(pattern))
        if found:
            return found
    return []


def load_items(limit: int) -> list[dict]:
    """Read MMAU test-mini rows, audio inlined as a data URI.

    A truncated read in shard order would sample ONE task: the split holds
    333 sound / 333 speech / 334 music, but stored in long same-task runs
    (96, 333, 301, 48, 33, 189), so the first 40 rows are all 'sound'. That
    was measured, not assumed -- an earlier version of this loader claimed
    the shards were pre-mixed and a 40-item smoke reported
    ``accuracy_by_task`` with a single key.

    Rows are therefore round-robined across tasks in shard order: still
    fully deterministic and unshuffled, but a prefix of any length covers
    all three tasks in near-equal proportion. Audio is read in a second
    pass, for the selected rows only, so a small smoke does not decode the
    whole 2.84 GB of WAV bytes.

    Args:
        limit: Maximum rows to return; 0 or negative returns every row.

    Returns:
        Item dicts with qid, question, choices, answer_letter, answer_text,
        task, difficulty and audio_uri.
    """
    import pyarrow.parquet as pq

    shards = _shards()
    by_task: dict[str, list[tuple[str, int, dict]]] = {}
    for shard in shards:
        cols = pq.read_table(shard, columns=_META_COLUMNS).to_pydict()
        for i in range(len(cols["id"])):
            choices = list(cols["choices"][i])
            answer = cols["answer"][i]
            if answer not in choices or not 2 <= len(choices) <= len(LETTERS):
                # Genuinely malformed: no ground truth to score against.
                continue
            meta = {
                "qid": cols["id"][i],
                "question": cols["question"][i],
                "choices": choices,
                "answer_letter": LETTERS[choices.index(answer)],
                "answer_text": answer,
                "task": cols["task"][i],
                "difficulty": cols["difficulty"][i],
            }
            by_task.setdefault(cols["task"][i], []).append((shard, i, meta))

    # Round-robin over tasks, tasks themselves in a fixed alphabetical order
    # so the selection does not depend on dict insertion order.
    queues = [by_task[task] for task in sorted(by_task)]
    selected: list[tuple[str, int, dict]] = []
    for rank in range(max((len(q) for q in queues), default=0)):
        for queue in queues:
            if rank < len(queue):
                selected.append(queue[rank])
    if limit > 0:
        selected = selected[:limit]

    # Second pass: fetch audio only for the rows that survived selection.
    wanted: dict[str, set[int]] = {}
    for shard, i, _ in selected:
        wanted.setdefault(shard, set()).add(i)
    audio: dict[tuple[str, int], bytes] = {}
    for shard, rows in wanted.items():
        col = pq.read_table(shard, columns=["audio"]).to_pydict()["audio"]
        for i in rows:
            audio[(shard, i)] = col[i]["bytes"]

    items = []
    for shard, i, meta in selected:
        uri = base64.b64encode(audio[(shard, i)]).decode("ascii")
        items.append({**meta, "audio_uri": "data:audio/wav;base64," + uri})
    return items


def prompt_text(item: dict) -> str:
    lines = [item["question"], ""]
    lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(item["choices"])]
    lines.append("")
    lines.append("Answer with the letter of the correct option only.")
    return "\n".join(lines)


def conversations(items: list[dict]) -> list[list[dict]]:
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": item["audio_uri"]}},
                    {"type": "text", "text": prompt_text(item)},
                ],
            }
        ]
        for item in items
    ]


def parse_letter(text: str, choices: list[str]) -> str:
    """The chosen option letter, '' if the answer names none.

    Two shapes are accepted, in order: a leading letter (``B``, ``B.``,
    ``(B)``, ``Answer: B``) which is what the instruction asks for, and
    failing that an exact choice TEXT anywhere in the answer, since a model
    that ignores the instruction usually restates the option verbatim.
    Substring letter search is avoided on purpose -- 'A' appears inside
    ordinary words and would manufacture verdicts out of prose.

    A letter past the end of THIS item's choice list does not count: most
    rows have four options but some have two, and accepting 'C' on a
    two-choice question would score a hallucinated option as a real answer.
    """
    stripped = text.strip()
    pattern = rf"^\W*(?:answer\s*[:\-]?\s*)?\(?([A-{LETTERS[-1]}])\)?(?:[.,:)\s]|$)"
    m = re.match(pattern, stripped, re.I)
    if m:
        letter = m.group(1).upper()
        return letter if LETTERS.index(letter) < len(choices) else ""
    lowered = stripped.lower()
    hits = [LETTERS[i] for i, c in enumerate(choices) if c.strip().lower() in lowered]
    return hits[0] if len(hits) == 1 else ""


def main() -> int:
    from vllm import LLM, SamplingParams

    items = load_items(n_items)
    if not items:
        print("no MMAU items found", file=sys.stderr)
        return 2
    llm = LLM(
        model=hf_id,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1, "image": 1, "video": 1},
        mm_encoder_attn_backend="TORCH_SDPA",
        disable_log_stats=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=8, seed=0)
    convs = conversations(items)
    first = llm.chat(convs, params, use_tqdm=False)
    again = llm.chat(convs, params, use_tqdm=False)

    rows = []
    for item, o1, o2 in zip(items, first, again, strict=True):
        t1 = o1.outputs[0].text
        letter = parse_letter(t1, item["choices"])
        rows.append(
            {
                "qid": item["qid"],
                "task": item["task"],
                "difficulty": item["difficulty"],
                "raw": t1.strip()[:60],
                "parsed": letter,
                "expected": item["answer_letter"],
                "correct": letter == item["answer_letter"],
                "stable": t1 == o2.outputs[0].text,
                "prompt_tokens": len(o1.prompt_token_ids),
            }
        )
    parsed = [r for r in rows if r["parsed"]]
    by_task: dict[str, list[bool]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["correct"])
    n_choices: dict[int, int] = {}
    for item in items:
        k = len(item["choices"])
        n_choices[k] = n_choices.get(k, 0) + 1
    report = {
        "model": hf_id,
        "n": len(rows),
        # Proof that the sample actually spans the split rather than one
        # task run -- the failure the round-robin in load_items exists for.
        "n_by_task": {k: len(v) for k, v in sorted(by_task.items())},
        "n_by_choice_count": dict(sorted(n_choices.items())),
        "parse_ratio": round(len(parsed) / len(rows), 4),
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
        "stable_ratio": round(sum(r["stable"] for r in rows) / len(rows), 4),
        "accuracy_by_task": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(by_task.items())
        },
        "prompt_tokens": {
            "min": min(r["prompt_tokens"] for r in rows),
            "max": max(r["prompt_tokens"] for r in rows),
            "mean": round(sum(r["prompt_tokens"] for r in rows) / len(rows), 1),
        },
        "rows": rows,
    }
    pathlib.Path(out_json).write_text(json.dumps(report, indent=2))
    print(
        json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2)
    )
    unparsed = [r for r in rows if not r["parsed"]]
    if unparsed:
        print("\nunparsed samples:")
        for r in unparsed[:8]:
            print("  ", r["raw"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
