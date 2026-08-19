# SPDX-License-Identifier: Apache-2.0
"""MME benchmark score-parity check for LMCache multimodal support.

The synthetic acceptance suite (test_mm_acceptance.py) proves cache-key
isolation; this script proves the cache HIT path does not degrade real
model quality. It scores the MME benchmark three ways:

1. ``baseline``  -- plain vLLM, no LMCache (run in a subprocess);
2. ``pass1``     -- LMCache engine, cold cache (miss path, fills the cache);
3. ``pass2``     -- same engine, same questions again (hit path: the KV for
   every prompt is restored from LMCache instead of being computed).

and reports standard MME Perception/Cognition scores per run plus per-item
answer flips. Any KV corruption on the hit path shows up directly as
pass2-vs-pass1 flips and a score drop.

Usage (from tests/e2e_mm):

    CUDA_VISIBLE_DEVICES=0 python benchmark_parity.py \
        [--model Qwen/Qwen2.5-VL-3B-Instruct] [--limit 0] \
        [--out mme_parity_report.json]

``--limit 0`` (default) runs the full 2374-question benchmark. Requires GPU,
model weights, and the ``lmms-lab/MME`` dataset (downloaded via HF).

Exit code 0 = parity holds (see THRESHOLDS below), 1 = parity violated.
"""

# Standard
import argparse
import base64
import io
import json
import os
import pathlib
import subprocess
import sys

# Pin THIS repo's lmcache package (see tests/e2e_mm/conftest.py).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

PERCEPTION_CATEGORIES = [
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
]
COGNITION_CATEGORIES = [
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
]

# Bound the visual token count so the full benchmark fits the LMCache CPU
# cache and the context window (Qwen smart-resize: tokens <= max_pixels/28^2).
MAX_PIXELS = 768 * 28 * 28

# Parity thresholds: batched GPU inference is not bit-deterministic, so a
# tiny number of borderline yes/no flips between passes is tolerated.
MAX_FLIP_FRACTION = 0.005  # pass2 vs pass1 per-item answer flips
MAX_SCORE_DELTA = 10.0  # |pass2 - pass1| on the 2800-point MME total
MIN_HIT_RATIO = 0.8  # pass2 lookup hit ratio (else parity is vacuous)


def load_items(limit: int) -> list[dict]:
    """Load MME questions: [{qid, image_uri, question, answer, category}]."""
    # Third Party
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/MME", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    items = []
    uri_cache: dict[str, str] = {}
    for row in ds:
        qid = row["question_id"]
        if qid not in uri_cache:
            buf = io.BytesIO()
            row["image"].convert("RGB").save(buf, format="PNG")
            uri_cache[qid] = "data:image/png;base64," + base64.b64encode(
                buf.getvalue()
            ).decode("ascii")
        items.append(
            {
                "qid": qid,
                "image_uri": uri_cache[qid],
                "question": row["question"],
                "answer": row["answer"].strip().lower(),
                "category": row["category"],
            }
        )
    return items


def conversations(items: list[dict]) -> list[list[dict]]:
    """Build one single-turn conversation per benchmark question."""
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": item["image_uri"]}},
                    {"type": "text", "text": item["question"]},
                ],
            }
        ]
        for item in items
    ]


def parse_yes_no(text: str) -> str:
    """Extract the yes/no verdict from a model answer ('' if neither)."""
    lowered = text.strip().lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return ""


def mme_scores(items: list[dict], answers: list[str]) -> dict:
    """Standard MME scoring: per-category acc*100 + acc+*100, summed.

    acc  = per-question accuracy;
    acc+ = fraction of images whose BOTH questions are answered correctly.
    Perception sums 10 categories (max 2000), Cognition 4 (max 800).
    """
    by_cat: dict[str, dict[str, list]] = {}
    for item, answer in zip(items, answers, strict=True):
        cat = by_cat.setdefault(item["category"], {"correct": [], "by_image": {}})
        ok = parse_yes_no(answer) == item["answer"]
        cat["correct"].append(ok)
        cat["by_image"].setdefault(item["qid"], []).append(ok)

    per_category = {}
    for name, cat in by_cat.items():
        acc = sum(cat["correct"]) / len(cat["correct"])
        plus_flags = [all(v) for v in cat["by_image"].values()]
        acc_plus = sum(plus_flags) / len(plus_flags)
        per_category[name] = round(acc * 100 + acc_plus * 100, 2)

    perception = round(sum(per_category.get(c, 0.0) for c in PERCEPTION_CATEGORIES), 2)
    cognition = round(sum(per_category.get(c, 0.0) for c in COGNITION_CATEGORIES), 2)
    return {
        "perception": perception,
        "cognition": cognition,
        "total": round(perception + cognition, 2),
        "per_category": per_category,
    }


def run_batch(llm, items: list[dict]) -> list[str]:
    """Run all questions in one batched chat call, order-aligned."""
    # Third Party
    from vllm import SamplingParams

    outputs = llm.chat(
        conversations(items),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=8, seed=0),
        use_tqdm=True,
    )
    return [out.outputs[0].text for out in outputs]


def engine_kwargs(model: str) -> dict:
    return dict(
        model=model,
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": MAX_PIXELS},
    )


def run_baseline(model: str, limit: int, out_path: str) -> None:
    """Subprocess role: plain vLLM answers for every question."""
    # Third Party
    from vllm import LLM

    items = load_items(limit)
    llm = LLM(**engine_kwargs(model))
    answers = run_batch(llm, items)
    with open(out_path, "w") as f:
        json.dump(answers, f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--limit", type=int, default=0, help="0 = full benchmark")
    parser.add_argument("--out", default="mme_parity_report.json")
    parser.add_argument("--role", choices=["main", "baseline"], default="main")
    parser.add_argument("--baseline-out", default="")
    args = parser.parse_args()

    # First Party (test-local)
    from harness import configure_environment, cumulative_lookup_stats

    configure_environment()

    if args.role == "baseline":
        run_baseline(args.model, args.limit, args.baseline_out)
        return 0

    items = load_items(args.limit)
    print(f"[parity] {len(items)} MME questions loaded")

    baseline_path = pathlib.Path(args.out).with_suffix(".baseline.json")
    proc = subprocess.run(
        [
            sys.executable,
            __file__,
            "--role",
            "baseline",
            "--model",
            args.model,
            "--limit",
            str(args.limit),
            "--baseline-out",
            str(baseline_path),
        ],
        timeout=7200,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"baseline subprocess failed with {proc.returncode}")
    answers_base = json.loads(baseline_path.read_text())

    # Third Party
    from vllm import LLM
    from vllm.config import KVTransferConfig

    # First Party
    from lmcache.observability import LMCStatsMonitor

    llm = LLM(
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheConnectorV1", kv_role="kv_both"
        ),
        **engine_kwargs(args.model),
    )
    monitor = LMCStatsMonitor.GetOrCreate()

    answers_p1 = run_batch(llm, items)
    t0, h0 = cumulative_lookup_stats(monitor)
    answers_p2 = run_batch(llm, items)
    t1, h1 = cumulative_lookup_stats(monitor)
    hit_ratio = (h1 - h0) / max(1, t1 - t0)

    scores = {
        "baseline": mme_scores(items, answers_base),
        "pass1_miss": mme_scores(items, answers_p1),
        "pass2_hit": mme_scores(items, answers_p2),
    }
    flips_p1_base = sum(
        parse_yes_no(a) != parse_yes_no(b)
        for a, b in zip(answers_p1, answers_base, strict=True)
    )
    flips_p2_p1 = sum(
        parse_yes_no(a) != parse_yes_no(b)
        for a, b in zip(answers_p2, answers_p1, strict=True)
    )
    report = {
        "model": args.model,
        "num_questions": len(items),
        "scores": scores,
        "flips_pass1_vs_baseline": flips_p1_base,
        "flips_pass2_vs_pass1": flips_p2_p1,
        "pass2_lookup_hit_ratio": round(hit_ratio, 4),
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    score_delta = abs(scores["pass2_hit"]["total"] - scores["pass1_miss"]["total"])
    ok = (
        flips_p2_p1 <= MAX_FLIP_FRACTION * len(items)
        and score_delta <= MAX_SCORE_DELTA
        and hit_ratio >= MIN_HIT_RATIO
    )
    print(
        f"[parity] hit_ratio={hit_ratio:.3f} "
        f"score_delta(pass2-pass1)={score_delta:.2f} "
        f"flips(pass2 vs pass1)={flips_p2_p1}/{len(items)} "
        f"=> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    # vLLM/NCCL teardown can hang for a long time while holding all GPU
    # memory; the report is already on disk, so exit hard.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
