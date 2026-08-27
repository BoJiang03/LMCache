# SPDX-License-Identifier: Apache-2.0
"""Measure whether a combined image+audio probe is usable on the target.

The audio work established a per-model rule: a semantic probe belongs to the
(model, stimulus) pair and must be measured on the model being certified.
This round asks the question the cross-modal cases need answered:

  correct    does the model name BOTH the image color and the sound kind,
             from the fixed vocabulary the question offers
  stable     same answer on a second pass (no cache involved here)
  distinct   different (image, audio) content -> different answers, which is
             what a cross-item hit detector needs to be able to see
  order      does the answer survive putting the CLIP first, i.e. does
             "first the color" still bind to the image

Everything is built through catalog.cross_modal_request, so what is measured
is what the suite ships.

usage: python crossmodal_probe.py <hf_id> <out_json> [mm_encoder_backend]
"""

# Standard
import json
import os
import pathlib
import sys

sys.path.insert(0, "/home/bo/LMCache-worktrees/multi_modal/tests/e2e_mm")

hf_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-Omni-30B-A3B-Instruct"
out_json = sys.argv[2] if len(sys.argv) > 2 else "crossmodal_probe.json"
mm_backend = sys.argv[3] if len(sys.argv) > 3 else "TORCH_SDPA"

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HUB_CACHE", "/raid/data/hub")
os.environ.setdefault("PYTHONHASHSEED", "0")
venv_include = pathlib.Path(sys.prefix) / "include"
if (venv_include / "Python.h").exists():
    os.environ["CPATH"] = f"{venv_include}:{os.environ.get('CPATH', '')}"

# (image_index, audio_index): images 0/1/2 are red/green/blue, audio 0/2/4
# are tone/beeping/warble. Both coordinates vary independently so a probe
# that only ever reads one of the two items shows up as a collision.
COMBOS = [(0, 0), (0, 2), (2, 0), (2, 4), (1, 2)]
ORDERS = [("image", "audio"), ("audio", "image")]


def main() -> int:
    # First Party (test-local)
    import catalog

    # Third Party
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=hf_id,
        max_model_len=8192,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 2, "audio": 1},
        disable_log_stats=True,
    )
    if mm_backend:
        kwargs["mm_encoder_attn_backend"] = mm_backend
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=16, seed=0)

    cases = []
    for order in ORDERS:
        for image_index, audio_index in COMBOS:
            tag = f"{'IA' if order[0] == 'image' else 'AI'}-i{image_index}a{audio_index}"
            cases.append(
                (
                    tag,
                    order,
                    image_index,
                    audio_index,
                    catalog.cross_modal_request(
                        tag, f"probe-{tag}", image_index, audio_index, order
                    ),
                )
            )

    convs = [case[4].messages() for case in cases]
    first = llm.chat(convs, params, use_tqdm=False)
    again = llm.chat(convs, params, use_tqdm=False)

    rows = []
    for case, o1, o2 in zip(cases, first, again, strict=True):
        tag, order, image_index, audio_index, req = case
        text = o1.outputs[0].text.strip()
        text2 = o2.outputs[0].text.strip()
        lowered = text.lower()
        color = catalog.image_color_name(image_index)
        kind = catalog.audio_kind_name(audio_index)
        rows.append(
            {
                "tag": tag,
                "order": list(order),
                "image_index": image_index,
                "audio_index": audio_index,
                "expected": [color, kind],
                "text": text,
                "text_again": text2,
                "color_ok": color in lowered,
                "kind_ok": kind in lowered,
                "correct": color in lowered and kind in lowered,
                "stable": text == text2,
                "probe_passes": all(w in lowered for w in req.expected_probe),
            }
        )

    # Distinctness is only required between different CONTENT. The same
    # content in a different order SHOULD answer alike, so group by order.
    distinct = {}
    for order in ORDERS:
        name = "".join(m[0].upper() for m in order)
        texts = [r["text"] for r in rows if r["order"] == list(order)]
        distinct[name] = {
            "n": len(texts),
            "unique": len(set(texts)),
            "all_distinct": len(set(texts)) == len(texts),
        }

    # And the cross-order agreement: same (image, audio) either way round.
    by_key = {(r["order"][0], r["image_index"], r["audio_index"]): r for r in rows}
    agreement = []
    for image_index, audio_index in COMBOS:
        ia = by_key[("image", image_index, audio_index)]
        ai = by_key[("audio", image_index, audio_index)]
        agreement.append(
            {
                "combo": [image_index, audio_index],
                "ia_text": ia["text"],
                "ai_text": ai["text"],
                "same": ia["text"] == ai["text"],
                "both_correct": ia["correct"] and ai["correct"],
            }
        )

    report = {
        "model": hf_id,
        "mm_encoder_attn_backend": mm_backend,
        "rows": rows,
        "distinct_within_order": distinct,
        "cross_order": agreement,
        "summary": {
            "n": len(rows),
            "correct": sum(1 for r in rows if r["correct"]),
            "stable": sum(1 for r in rows if r["stable"]),
            "color_ok": sum(1 for r in rows if r["color_ok"]),
            "kind_ok": sum(1 for r in rows if r["kind_ok"]),
            "usable": all(r["correct"] for r in rows)
            and all(d["all_distinct"] for d in distinct.values()),
        },
    }
    pathlib.Path(out_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["distinct_within_order"], indent=2))
    for row in rows:
        print(
            f"{row['tag']:>18}  expect={row['expected']}  "
            f"correct={row['correct']}  stable={row['stable']}  {row['text']!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
