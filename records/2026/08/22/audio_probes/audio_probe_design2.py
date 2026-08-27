# SPDX-License-Identifier: Apache-2.0
"""Wider sweep for an audio probe oracle: correct, stable, and distinct.

Round 1 (audio_probe_design.py) found Qwen2.5-Omni-3B fully deterministic
on synthetic audio but unable to describe it: beeps miscounted above two
(3 -> "4", 4 -> "3") and 1760 Hz called "low". Both failures were STABLE,
so what the suite's cross-item detector needs -- item A and item B giving
different answers -- may survive even where correctness does not. This
round measures all three properties over more stimulus families, so the
choice is made on evidence rather than on which question sounds natural:

  correct   answer matches the synthesised ground truth
  stable    identical answer on a second pass
  distinct  no two items in the family share an answer

A family that is stable and distinct is usable as a detector; one that is
also correct is usable without explaining itself.

usage: python audio_probe_design2.py <hf_id> <out_json>
"""

import base64
import io
import json
import os
import pathlib
import sys
import wave

import numpy as np

hf_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-Omni-3B"
out_json = sys.argv[2] if len(sys.argv) > 2 else "audio_probe_design2.json"

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("PYTHONHASHSEED", "0")
venv_include = pathlib.Path(sys.prefix) / "include"
if (venv_include / "Python.h").exists():
    os.environ["CPATH"] = f"{venv_include}:{os.environ.get('CPATH', '')}"

SAMPLE_RATE = 16000
RNG_SEED = 0


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


def beeps(n: int, gap: float) -> np.ndarray:
    parts = [_silence(0.25)]
    for _ in range(n):
        parts += [_tone(440.0, 0.3), _silence(gap)]
    return np.concatenate(parts)


def steady(freq: float) -> np.ndarray:
    return np.concatenate([_silence(0.1), _tone(freq, 1.6)])


def kind(which: str) -> np.ndarray:
    body = {
        "tone": _tone(440.0, 1.5),
        "noise": _noise(1.5),
        "silence": _silence(1.5),
    }[which]
    return np.concatenate([_silence(0.1), body])


def rising_or_falling(direction: str) -> np.ndarray:
    lo, hi = 220.0, 1760.0
    a, b = (lo, hi) if direction == "rising" else (hi, lo)
    return np.concatenate([_silence(0.1), _tone(a, 0.7), _silence(0.15), _tone(b, 0.7)])


# (family, question, [(expected, samples)])
FAMILIES = [
    (
        "beeps_gap_600ms",
        "How many separate beeps are in this audio? Reply with just the digit.",
        [(str(n), beeps(n, 0.6)) for n in (1, 2, 3, 4, 5)],
    ),
    (
        "sound_kind",
        "Is this audio a musical tone, static noise, or silence? "
        "Reply with one word: tone, noise, or silence.",
        [(k, kind(k)) for k in ("tone", "noise", "silence")],
    ),
    (
        "pitch_wide",
        "Is the pitch of this tone very low, medium, or very high? "
        "Reply with one word: low, medium, or high.",
        [
            ("low", steady(110.0)),
            ("medium", steady(660.0)),
            ("high", steady(3520.0)),
        ],
    ),
    (
        "pitch_direction",
        "Does the pitch go up or down between the two tones? "
        "Reply with one word: up or down.",
        [("up", rising_or_falling("rising")), ("down", rising_or_falling("falling"))],
    ),
]


def _messages(uri: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": uri}},
                {"type": "text", "text": question},
            ],
        },
    ]


def main() -> int:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=hf_id,
        max_model_len=4096,
        gpu_memory_utilization=0.55,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1, "image": 1},
        disable_log_stats=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=12, seed=0)
    report: dict[str, object] = {"model": hf_id, "families": {}}
    for name, question, items in FAMILIES:
        rows = []
        for expected, samples in items:
            msgs = _messages(_wav_uri(samples), question)
            o1 = llm.chat(msgs, params, use_tqdm=False)[0]
            o2 = llm.chat(msgs, params, use_tqdm=False)[0]
            a1 = o1.outputs[0].text.strip()
            rows.append(
                {
                    "expected": expected,
                    "answer": a1[:50],
                    "correct": expected.lower() in a1.lower(),
                    "stable": a1 == o2.outputs[0].text.strip(),
                    "prompt_tokens": len(o1.prompt_token_ids),
                    "seconds": round(len(samples) / SAMPLE_RATE, 2),
                }
            )
        answers = [r["answer"].lower() for r in rows]
        report["families"][name] = {
            "question": question,
            "rows": rows,
            "all_correct": all(r["correct"] for r in rows),
            "all_stable": all(r["stable"] for r in rows),
            "all_distinct": len(set(answers)) == len(answers),
        }
        f = report["families"][name]
        print(
            f"{name:18s} correct={f['all_correct']} stable={f['all_stable']} "
            f"distinct={f['all_distinct']} answers={answers}"
        )
    pathlib.Path(out_json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
