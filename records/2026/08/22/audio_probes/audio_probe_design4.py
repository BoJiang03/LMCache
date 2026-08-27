# SPDX-License-Identifier: Apache-2.0
"""Round 4: how many distinct audio answers does the CERTIFICATION target
actually give?

Round 3 killed the three-value palette on Qwen3-Omni-30B: 'silence' comes
back as "tone", stably, which collides with the real tone and is therefore
the blinding kind of failure -- item A returning item B's cached answer is
invisible when both answer alike. That leaves only tone and noise, and two
values is a thin basis for an isolation matrix whose image counterpart has
six colors.

So before settling for two, this asks whether a WIDER question buys more.
The candidates are chosen to be acoustically far from both a steady 440 Hz
tone and white noise, and far from the properties already measured to fail
(counting, pitch height, pitch direction):

  beeping   an interrupted tone -- the model miscounts beeps, but "is it
            beeping" is a coarser judgement than "how many"
  rumble    a 60 Hz sine, far below the 440 Hz reference
  warble    a tone swept continuously between two pitches

Each is asked with a single closed question listing every option, so the
model picks from a fixed vocabulary and the answers stay comparable. As in
round 2 the three properties measured are correct, stable and DISTINCT --
distinctness being the one that actually decides usability.

usage: python audio_probe_design4.py <hf_id> <out_json> [mm_encoder_backend]
"""

import base64
import io
import json
import os
import pathlib
import sys
import wave

import numpy as np

hf_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-Omni-30B-A3B-Instruct"
out_json = sys.argv[2] if len(sys.argv) > 2 else "audio_probe_design4.json"
mm_backend = sys.argv[3] if len(sys.argv) > 3 else "TORCH_SDPA"

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HUB_CACHE", "/raid/data/hub")
os.environ.setdefault("PYTHONHASHSEED", "0")
venv_include = pathlib.Path(sys.prefix) / "include"
if (venv_include / "Python.h").exists():
    os.environ["CPATH"] = f"{venv_include}:{os.environ.get('CPATH', '')}"

SAMPLE_RATE = 16000
SECONDS = 1.5
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


def _lead(body: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(int(0.1 * SAMPLE_RATE)), body])


def steady_tone() -> np.ndarray:
    return _lead(_tone(440.0, SECONDS))


def static_noise() -> np.ndarray:
    n = int(SAMPLE_RATE * SECONDS)
    rng = np.random.default_rng(RNG_SEED)
    return _lead(0.35 * _env(n) * rng.standard_normal(n))


def silence() -> np.ndarray:
    return _lead(np.zeros(int(SAMPLE_RATE * SECONDS)))


def beeping() -> np.ndarray:
    """Three short 440 Hz beeps separated by gaps."""
    gap = np.zeros(int(0.18 * SAMPLE_RATE))
    parts = []
    for _ in range(3):
        parts += [_tone(440.0, 0.28), gap]
    return _lead(np.concatenate(parts))


def rumble() -> np.ndarray:
    """A 60 Hz sine: far below the 440 Hz reference tone."""
    return _lead(_tone(60.0, SECONDS, amp=0.75))


def warble() -> np.ndarray:
    """A tone swept continuously between 300 and 900 Hz."""
    n = int(SAMPLE_RATE * SECONDS)
    t = np.arange(n) / SAMPLE_RATE
    freq = 600.0 + 300.0 * np.sin(2 * np.pi * 3.0 * t)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    return _lead(0.6 * _env(n) * np.sin(phase))


CANDIDATES = [
    ("tone", steady_tone),
    ("noise", static_noise),
    ("silence", silence),
    ("beeping", beeping),
    ("rumble", rumble),
    ("warble", warble),
]

QUESTION = (
    "Which of these best describes the audio: a steady musical tone, "
    "static noise, silence, repeated beeping, a low rumble, or a warbling "
    "tone? Reply with one word: tone, noise, silence, beeping, rumble, or "
    "warble."
)


def _messages(uri: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": uri}},
                {"type": "text", "text": QUESTION},
            ],
        },
    ]


def main() -> int:
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=hf_id,
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1},
        disable_log_stats=True,
    )
    if mm_backend:
        kwargs["mm_encoder_attn_backend"] = mm_backend
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=8, seed=0)

    rows = []
    for expected, build in CANDIDATES:
        msgs = _messages(_wav_uri(build()))
        o1 = llm.chat(msgs, params, use_tqdm=False)[0]
        o2 = llm.chat(msgs, params, use_tqdm=False)[0]
        a1 = o1.outputs[0].text.strip()
        rows.append(
            {
                "expected": expected,
                "answer": a1[:40],
                "normalized": a1.strip(".,!").lower().split()[0] if a1 else "",
                "correct": expected in a1.lower(),
                "stable": a1 == o2.outputs[0].text.strip(),
                "prompt_tokens": len(o1.prompt_token_ids),
            }
        )
        r = rows[-1]
        print(
            f"{expected:9s} -> {r['answer']!r:22s} correct={r['correct']} "
            f"stable={r['stable']}"
        )

    # The usable palette: kinds that are correct, stable, and whose answer
    # is claimed by no other kind.
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["normalized"]] = counts.get(r["normalized"], 0) + 1
    usable = [
        r["expected"]
        for r in rows
        if r["correct"] and r["stable"] and counts[r["normalized"]] == 1
    ]
    report = {
        "model": hf_id,
        "mm_backend": mm_backend,
        "question": QUESTION,
        "rows": rows,
        "answer_counts": counts,
        "usable_palette": usable,
        "palette_size": len(usable),
    }
    pathlib.Path(out_json).write_text(json.dumps(report, indent=2))
    print(f"\nusable palette ({len(usable)}): {usable}")
    collisions = {a: c for a, c in counts.items() if c > 1}
    if collisions:
        print(f"collisions (blinding failure mode): {collisions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
