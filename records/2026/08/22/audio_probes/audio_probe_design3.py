# SPDX-License-Identifier: Apache-2.0
"""Round 3: re-verify the audio oracle on the CERTIFICATION target, and ask
whether ordered pairs widen it beyond three values.

Rounds 1 and 2 were both run on Qwen2.5-Omni-3B, which is not the model
being certified. Round 2 settled on ``sound_kind`` (tone / noise / silence)
as the only family that is correct AND stable AND pairwise distinct -- but
an oracle that holds on the 3B says nothing about the 30B, so it is
re-measured here on whichever model is passed in.

The second question is capacity. ``sound_kind`` yields only THREE distinct
answers, and the image side of the suite has six colors. Three is enough
for cross-item isolation (which needs two), but not for a multi-item case
that has to name each item in order. Ordered PAIRS of kinds would give
nine, so this round measures whether the model reports both members of a
two-clip prompt in the right order. That is a hypothesis, not an
assumption: round 2 already showed this model family miscounting beeps and
collapsing pitch labels onto each other, and a pair oracle that silently
collapses would blind the cross-item detector exactly like those did.

usage: python audio_probe_design3.py <hf_id> <out_json> [mm_encoder_backend]
"""

import base64
import io
import itertools
import json
import os
import pathlib
import sys
import wave

import numpy as np

hf_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-Omni-3B"
out_json = sys.argv[2] if len(sys.argv) > 2 else "audio_probe_design3.json"
mm_backend = sys.argv[3] if len(sys.argv) > 3 else ""

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HUB_CACHE", "/raid/data/hub")
os.environ.setdefault("PYTHONHASHSEED", "0")
venv_include = pathlib.Path(sys.prefix) / "include"
if (venv_include / "Python.h").exists():
    os.environ["CPATH"] = f"{venv_include}:{os.environ.get('CPATH', '')}"

SAMPLE_RATE = 16000
RNG_SEED = 0
KINDS = ("tone", "noise", "silence")


def _wav_uri(samples: np.ndarray) -> str:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def _env(n: int) -> np.ndarray:
    e = np.ones(n)
    edge = max(1, int(0.006 * SAMPLE_RATE))
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
    e[:edge] = ramp
    e[-edge:] = ramp[::-1]
    return e


def _tone(freq: float, seconds: float, amp: float = 0.6) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return amp * _env(len(t)) * np.sin(2 * np.pi * freq * t)


def _noise(seconds: float, amp: float = 0.35) -> np.ndarray:
    n = int(SAMPLE_RATE * seconds)
    rng = np.random.default_rng(RNG_SEED)
    return amp * _env(n) * rng.standard_normal(n)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds))


def kind_samples(which: str, seconds: float = 1.5) -> np.ndarray:
    """One clip of the named kind, with a short leading silence."""
    body = {
        "tone": _tone(440.0, seconds),
        "noise": _noise(seconds),
        "silence": _silence(seconds),
    }[which]
    return np.concatenate([_silence(0.1), body])


SINGLE_QUESTION = (
    "Is this audio a musical tone, static noise, or silence? "
    "Reply with one word: tone, noise, or silence."
)
PAIR_QUESTION = (
    "You are given two audio clips in order. For each one, say whether it "
    "is a musical tone, static noise, or silence. Reply with exactly two "
    "words separated by a space, in the order the clips were given."
)


def _messages(uris: list[str], question: str) -> list[dict]:
    content: list[dict] = [
        {"type": "audio_url", "audio_url": {"url": u}} for u in uris
    ]
    content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": content},
    ]


def main() -> int:
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=hf_id,
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 2},
        disable_log_stats=True,
    )
    if mm_backend:
        kwargs["mm_encoder_attn_backend"] = mm_backend
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=12, seed=0)
    report: dict[str, object] = {"model": hf_id, "mm_backend": mm_backend}

    # --- single-clip oracle, re-measured on THIS model ---
    single_rows = []
    for k in KINDS:
        msgs = _messages([_wav_uri(kind_samples(k))], SINGLE_QUESTION)
        o1 = llm.chat(msgs, params, use_tqdm=False)[0]
        o2 = llm.chat(msgs, params, use_tqdm=False)[0]
        a1 = o1.outputs[0].text.strip()
        single_rows.append(
            {
                "expected": k,
                "answer": a1[:40],
                "correct": k in a1.lower(),
                "stable": a1 == o2.outputs[0].text.strip(),
                "prompt_tokens": len(o1.prompt_token_ids),
            }
        )
    answers = [r["answer"].lower() for r in single_rows]
    report["sound_kind"] = {
        "rows": single_rows,
        "all_correct": all(r["correct"] for r in single_rows),
        "all_stable": all(r["stable"] for r in single_rows),
        "all_distinct": len(set(answers)) == len(answers),
    }
    s = report["sound_kind"]
    print(
        f"sound_kind      correct={s['all_correct']} stable={s['all_stable']} "
        f"distinct={s['all_distinct']} answers={answers}"
    )

    # --- ordered pairs: does the answer space widen to nine? ---
    pair_rows = []
    for a, b in itertools.product(KINDS, repeat=2):
        uris = [_wav_uri(kind_samples(a)), _wav_uri(kind_samples(b))]
        msgs = _messages(uris, PAIR_QUESTION)
        o1 = llm.chat(msgs, params, use_tqdm=False)[0]
        o2 = llm.chat(msgs, params, use_tqdm=False)[0]
        a1 = o1.outputs[0].text.strip()
        words = [w.strip(".,").lower() for w in a1.split()][:2]
        pair_rows.append(
            {
                "expected": f"{a} {b}",
                "answer": a1[:40],
                # Correct only if BOTH members are named in the right order.
                "correct": words == [a, b],
                "stable": a1 == o2.outputs[0].text.strip(),
                "prompt_tokens": len(o1.prompt_token_ids),
            }
        )
    pair_answers = [r["answer"].lower() for r in pair_rows]
    report["kind_pairs"] = {
        "rows": pair_rows,
        "all_correct": all(r["correct"] for r in pair_rows),
        "all_stable": all(r["stable"] for r in pair_rows),
        "all_distinct": len(set(pair_answers)) == len(pair_answers),
        "n_distinct": len(set(pair_answers)),
        "n_total": len(pair_rows),
        # The number that decides usability: ordered pairs are worth having
        # only if the DISTINCT count is meaningfully above the three that
        # single clips already give.
        "n_correct": sum(r["correct"] for r in pair_rows),
    }
    p = report["kind_pairs"]
    print(
        f"kind_pairs      correct={p['n_correct']}/{p['n_total']} "
        f"stable={p['all_stable']} distinct={p['n_distinct']}/{p['n_total']}"
    )
    for r in pair_rows:
        print(f"   {r['expected']:18s} -> {r['answer']!r}")
    pathlib.Path(out_json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
