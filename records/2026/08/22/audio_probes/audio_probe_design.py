# SPDX-License-Identifier: Apache-2.0
"""Find an audio probe the model answers correctly AND deterministically.

The image suite rests on one property: a synthetic item whose content the
model reports reliably ("what colour is this?" -> "Red"), so that a false
cache hit makes it name the OTHER item's colour. Audio needs the same
thing, and nothing in the suite has it yet. This script measures candidate
probes on plain vLLM (no LMCache) and reports, per candidate:

  correct     - answer matches the synthesised ground truth
  stable      - the same clip answered identically on a second pass

A candidate that is not both is unusable as an oracle, whatever else it is.

usage: python audio_probe_design.py <hf_id> <out_json>
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
out_json = sys.argv[2] if len(sys.argv) > 2 else "audio_probe_design.json"

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("PYTHONHASHSEED", "0")
venv_include = pathlib.Path(sys.prefix) / "include"
if (venv_include / "Python.h").exists():
    os.environ["CPATH"] = f"{venv_include}:{os.environ.get('CPATH', '')}"

SAMPLE_RATE = 16000


def _wav_uri(samples: np.ndarray) -> str:
    """16 kHz mono PCM16 WAV as a base64 data URI."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def _tone(freq: float, seconds: float) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    # Raised-cosine edges so a beep has no click transient to count instead.
    env = np.ones_like(t)
    edge = max(1, int(0.005 * SAMPLE_RATE))
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
    env[:edge] = ramp
    env[-edge:] = ramp[::-1]
    return 0.6 * env * np.sin(2 * np.pi * freq * t)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds))


def beeps(n: int) -> np.ndarray:
    """``n`` 440 Hz beeps of 0.25 s separated by 0.35 s of silence."""
    parts = [_silence(0.2)]
    for _ in range(n):
        parts.append(_tone(440.0, 0.25))
        parts.append(_silence(0.35))
    return np.concatenate(parts)


def pitch(kind: str) -> np.ndarray:
    """A 1.5 s steady tone, low or high."""
    return np.concatenate(
        [_silence(0.1), _tone({"low": 220.0, "high": 1760.0}[kind], 1.5)]
    )


# Each candidate: (name, question, [(expected_answer, samples), ...]).
CANDIDATES = [
    (
        "beep_count",
        "How many separate beeps are in this audio? Reply with just the digit.",
        [(str(n), beeps(n)) for n in (1, 2, 3, 4)],
    ),
    (
        "pitch_binary",
        "Is the tone in this audio low-pitched or high-pitched? "
        "Reply with one word: low or high.",
        [(k, pitch(k)) for k in ("low", "high")],
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
    params = SamplingParams(temperature=0.0, max_tokens=16, seed=0)
    report: dict[str, object] = {"model": hf_id, "candidates": {}}
    for name, question, items in CANDIDATES:
        rows = []
        for expected, samples in items:
            uri = _wav_uri(samples)
            msgs = _messages(uri, question)
            first = llm.chat(msgs, params, use_tqdm=False)[0].outputs[0].text
            again = llm.chat(msgs, params, use_tqdm=False)[0].outputs[0].text
            rows.append(
                {
                    "expected": expected,
                    "answer": first.strip()[:60],
                    "correct": expected.lower() in first.strip().lower(),
                    "stable": first == again,
                    "seconds": round(len(samples) / SAMPLE_RATE, 2),
                }
            )
        report["candidates"][name] = {
            "question": question,
            "rows": rows,
            "all_correct": all(r["correct"] for r in rows),
            "all_stable": all(r["stable"] for r in rows),
        }
        print(name, json.dumps(report["candidates"][name], indent=2))
    pathlib.Path(out_json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
