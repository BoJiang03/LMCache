# SPDX-License-Identifier: Apache-2.0
"""Validate the SUITE's own audio generator against the model.

Rounds 1-4 measured hand-written stimuli. This one imports catalog.py and
probes exactly the clips and exactly the question the suite will ship, for
the reason recorded after an earlier false certification: a hand-rolled
runner can certify an engine, prompt or stimulus that the suite never
actually builds. Anything measured here is measured on the shipping code.

Two properties are checked, and they are different claims:

  correct     each index is named by its own kind, so ``expected_probe``
              in catalog.audio_kind_request is right
  same-kind   two DIFFERENT indices of the same kind get the SAME answer

The second matters because catalog dithers every clip to give it unique
bytes. The dither is meant to be inaudible; if it is not, two indices of
one kind could answer differently and the semantic probe would fail on
whichever index the case happened to pick.

usage: python audio_probe_catalog.py <hf_id> <out_json> [mm_encoder_backend]
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, "/home/bo/LMCache-worktrees/multi_modal/tests/e2e_mm")

hf_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-Omni-30B-A3B-Instruct"
out_json = sys.argv[2] if len(sys.argv) > 2 else "audio_probe_catalog.json"
mm_backend = sys.argv[3] if len(sys.argv) > 3 else "TORCH_SDPA"

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HUB_CACHE", "/raid/data/hub")
os.environ.setdefault("PYTHONHASHSEED", "0")
venv_include = pathlib.Path(sys.prefix) / "include"
if (venv_include / "Python.h").exists():
    os.environ["CPATH"] = f"{venv_include}:{os.environ.get('CPATH', '')}"

# Two full cycles of the palette, so every kind appears at two different
# indices with two different dithers.
INDICES = list(range(10))


def main() -> int:
    import catalog
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

    # Build the requests through the suite's own builder, and send exactly
    # the messages it produces.
    requests = [
        catalog.audio_kind_request(f"audio_probe_{i}", f"probe{i}", i) for i in INDICES
    ]
    convs = [r.messages() for r in requests]
    first = llm.chat(convs, params, use_tqdm=False)
    again = llm.chat(convs, params, use_tqdm=False)

    rows = []
    for req, index, o1, o2 in zip(requests, INDICES, first, again, strict=True):
        text = o1.outputs[0].text.strip()
        expected = catalog.audio_kind_name(index)
        rows.append(
            {
                "index": index,
                "expected": expected,
                "answer": text[:40],
                "normalized": text.strip(".,!").lower().split()[0] if text else "",
                # The suite's own probe rule: every expected word present.
                "probe_passes": all(w in text.lower() for w in req.expected_probe),
                "correct": expected in text.lower(),
                "stable": text == o2.outputs[0].text.strip(),
                "prompt_tokens": len(o1.prompt_token_ids),
            }
        )
        r = rows[-1]
        print(
            f"  idx{index:<3d} {expected:8s} -> {r['answer']!r:14s} "
            f"probe={r['probe_passes']} stable={r['stable']}"
        )

    by_kind: dict[str, set[str]] = {}
    for r in rows:
        by_kind.setdefault(r["expected"], set()).add(r["normalized"])
    same_kind_agrees = {k: len(v) == 1 for k, v in sorted(by_kind.items())}
    answers_per_kind = {k: sorted(v) for k, v in sorted(by_kind.items())}
    # Distinctness across kinds: no answer claimed by two different kinds.
    claimed: dict[str, set[str]] = {}
    for kind, answers in by_kind.items():
        for a in answers:
            claimed.setdefault(a, set()).add(kind)
    collisions = {a: sorted(k) for a, k in claimed.items() if len(k) > 1}

    report = {
        "model": hf_id,
        "mm_backend": mm_backend,
        "question": catalog.AUDIO_KIND_QUESTION,
        "palette": list(catalog._AUDIO_KINDS),
        "rows": rows,
        "all_probe_pass": all(r["probe_passes"] for r in rows),
        "all_stable": all(r["stable"] for r in rows),
        "same_kind_agrees": same_kind_agrees,
        "answers_per_kind": answers_per_kind,
        "cross_kind_collisions": collisions,
        "prompt_tokens": sorted({r["prompt_tokens"] for r in rows}),
    }
    pathlib.Path(out_json).write_text(json.dumps(report, indent=2))
    print(f"\nall probes pass : {report['all_probe_pass']}")
    print(f"all stable      : {report['all_stable']}")
    print(f"same-kind agrees: {same_kind_agrees}")
    print(f"collisions      : {collisions or 'none'}")
    print(f"prompt tokens   : {report['prompt_tokens']}")
    return 0 if report["all_probe_pass"] and not collisions else 1


if __name__ == "__main__":
    sys.exit(main())
