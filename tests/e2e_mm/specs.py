# SPDX-License-Identifier: Apache-2.0
"""Model registry for the multimodal acceptance suite.

Adding support certification for a new model means adding one ``ModelSpec``
entry here. The test logic is model-agnostic; only this declarative spec (and
optional ``extra_suites`` flags for special architectures) is per-model.
"""

# Standard
from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of one model under certification.

    Attributes:
        key: Short identifier used in ``LMCACHE_MM_E2E_MODELS``.
        hf_id: HuggingFace model id (smallest usable variant, to bound cost).
        modalities: Modalities to certify ("image", "video", "audio").
        max_model_len: Engine context length for the test run.
        gpu_memory_utilization: Fraction of GPU memory for the engine.
        extra_suites: Special-architecture add-on suites this model needs
            (e.g. "deepstack", "bidirectional_image_attn", "modality_lora").
        chat_template_kwargs: Extra kwargs passed to every ``llm.chat`` call
            (test engine, baseline runner, and MME parity runs alike), e.g.
            ``{"enable_thinking": False}`` for hybrid-thinking models whose
            template supports disabling the reasoning preamble. Must be
            JSON-serializable. The suite's oracles (baseline exact match,
            semantic probes, MME yes/no parsing) assume the answer lands
            within each request's small ``max_tokens`` budget, so thinking
            MUST be disabled here for models that would otherwise emit a
            reasoning preamble.
        mme_mm_processor_kwargs: ``mm_processor_kwargs`` for the MME parity
            engines only (real photos of arbitrary size, unlike the suite's
            small synthetic images). Must cap the per-image token count so a
            question fits the 8192-token parity context; the kwarg names are
            model-processor-specific (Qwen: ``max_pixels``; GLM:
            ``size.longest_edge`` in total pixels). Must be
            JSON-serializable.
        min_decode_tokens: Floor applied to every request's ``max_tokens``
            (suite, baselines, and MME parity alike). Models that lead with a
            short preamble before the answer even with thinking disabled
            (e.g. GLM's ``<|begin_of_box|>``-boxed answers) need enough
            budget for the answer to land inside the generated text, or the
            semantic probes and MME yes/no parsing read only preamble. 0
            keeps each request's own budget.
        mme_max_tokens: Decode budget for the MME parity runs specifically.
            Real MME photos draw a much longer preamble than the suite's
            synthetic images (reasoning over OCR'd code, artwork, posters),
            so a budget that suffices for the suite can still truncate MME
            answers and fail the parse-ratio gate. 0 falls back to
            ``min_decode_tokens`` (and the parity runner's own default when
            that is also 0).
        mme_max_flip_fraction: Per-item answer-flip budget for the MME
            parity gate, as a fraction of the question count. 0 keeps the
            gate's default (0.5%, calibrated on short-answer models). Long
            chain-of-thought models need a higher floor: with vs. without
            the LMCache connector are two different (each fully
            deterministic) numeric regimes, and a 200+-token reasoning
            chain amplifies the regime difference into ~1% of borderline
            answers landing on the other side or a repetition loop
            (parse ''), with no corruption signature. Real KV corruption
            is still caught by the byte-identical replay oracles, the
            hit-ratio gate, and the score-delta gates.
        mme_max_local_cpu_gb: LMCache local-CPU capacity (GB) for the MME
            parity run. 0 keeps the runner's 40 GB default, which holds the
            full benchmark's KV for the certified GQA-2 models (28-36
            KB/token). A wider-KV model overflows it -- the pass-2 replay
            revisits requests in store order, the LRU scan evicts every
            entry before its revisit, and the hit-ratio gate fails at ~0
            with zero flips (pure recompute, not corruption). Size it to
            hold the whole benchmark: questions x prompt tokens x KV bytes
            per token.
        answer_extract_pattern: Regex whose LAST match's group(1) is the
            model's final answer inside a generated text ('' = the whole
            text is the answer). For models that phrase a preamble before a
            marked answer (GLM: ``<|begin_of_box|>...<|end_of_box|>``), the
            hit-path replay oracle compares extracted answers when the full
            texts differ: miss pass (KV computed) and hit pass (KV loaded)
            are different numeric regimes, and a verbose preamble gives the
            regime noise many tokens to flip, while a real KV corruption
            flips the marked answer itself.
    """

    key: str
    hf_id: str
    modalities: frozenset = frozenset({"image"})
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.6
    extra_suites: frozenset = field(default_factory=frozenset)
    chat_template_kwargs: dict[str, object] = field(default_factory=dict)
    mme_mm_processor_kwargs: dict[str, object] = field(default_factory=dict)
    min_decode_tokens: int = 0
    mme_max_tokens: int = 0
    mme_max_flip_fraction: float = 0.0
    mme_max_local_cpu_gb: float = 0.0
    answer_extract_pattern: str = ""


# MME photos are arbitrarily large; cap them at ~768 image tokens per photo
# (Qwen smart-resize: tokens <= max_pixels / 28^2; GLM: 4 pixels per 14x14
# patch merge, same 602112-pixel budget) so a question fits the parity
# engine's 8192-token context.
_MME_PIXEL_BUDGET = 768 * 28 * 28

MODEL_SPECS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec(
            key="qwen2.5-vl-3b",
            hf_id="Qwen/Qwen2.5-VL-3B-Instruct",
            modalities=frozenset({"image", "video"}),
            mme_mm_processor_kwargs={"max_pixels": _MME_PIXEL_BUDGET},
        ),
        ModelSpec(
            key="qwen2-vl-2b",
            hf_id="Qwen/Qwen2-VL-2B-Instruct",
            modalities=frozenset({"image", "video"}),
            mme_mm_processor_kwargs={"max_pixels": _MME_PIXEL_BUDGET},
        ),
        ModelSpec(
            key="internvl3.5-2b",
            # The transformers-native export (InternVLForConditionalGeneration).
            # The OpenGVLab-format repo (InternVLChatModel) ships custom config
            # code and would need trust_remote_code, which the suite's engines
            # do not pass.
            hf_id="OpenGVLab/InternVL3_5-2B-HF",
            modalities=frozenset({"image", "video"}),
            # InternVL tiles photos into 448x448 crops worth 256 tokens each,
            # plus a 256-token global thumbnail whenever a photo is tiled
            # (HF processor: crop_to_patches); max_patches=2 caps a photo at
            # 768 image tokens, the same budget the Qwen/GLM pixel caps encode.
            mme_mm_processor_kwargs={"max_patches": 2},
            # The Qwen3-1.7B backbone is GQA-8: 28 layers x 8 KV heads x
            # 128 dims = 112 KB/token, 3-4x the certified GQA-2 models.
            # Full MME (~2374 x <=1000-token prompts) is ~250 GB of KV;
            # at the 40 GB default the pass-2 LRU scan hit ~0 (measured
            # 2026-08-20: hit_ratio 0.013 with 0 flips and 0.00 score
            # delta -- pure recompute). 280 GB holds the whole run.
            mme_max_local_cpu_gb=280.0,
        ),
        ModelSpec(
            key="qwen3-vl-2b",
            hf_id="Qwen/Qwen3-VL-2B-Instruct",
            modalities=frozenset({"image", "video"}),
            # DeepStack: the vision tower emits multiscale features from ViT
            # layers 5/11/17 that vLLM injects into LLM layers 0-2 via a
            # per-step side buffer OUTSIDE the paged KV. The add-on suite
            # verifies the LMCache resume path against KV recomputed with
            # that injection (see test_deepstack.py).
            extra_suites=frozenset({"deepstack"}),
            # New-style processor size cap (Qwen2VLImageProcessorFast):
            # 16x16 patches with 2x2 merge = 1024 pixels per token, so
            # 786432 total pixels cap a photo at the same ~768-token budget
            # the other specs encode.
            mme_mm_processor_kwargs={
                "size": {"shortest_edge": 65536, "longest_edge": 786432}
            },
            # Qwen3-1.7B-class backbone, GQA-8: 28 layers x 8 KV heads x
            # 128 dims = 112 KB/token, same as InternVL3.5-2B; the full MME
            # run needs the same 280 GB local-CPU capacity (see that spec).
            mme_max_local_cpu_gb=280.0,
        ),
        ModelSpec(
            key="glm-4.6v-flash",
            hf_id="zai-org/GLM-4.6V-Flash",
            modalities=frozenset({"image", "video"}),
            chat_template_kwargs={"enable_thinking": False},
            mme_mm_processor_kwargs={
                "size": {"shortest_edge": 12544, "longest_edge": _MME_PIXEL_BUDGET}
            },
            min_decode_tokens=64,
            # Real MME photos draw far longer reasoning than the suite's
            # synthetic swatches: at 64 tokens, 42% of baseline answers
            # truncated before the boxed verdict (parse-ratio gate FAIL).
            mme_max_tokens=256,
            # Measured (2026-08-20): flips are exactly reproducible across
            # runs (27 p1-vs-base, 17 p2-vs-p1 of 2374; baseline reruns are
            # byte-identical, self-flips 0) and per-question inspection
            # showed only benign reasoning drift / repetition-loop
            # truncation on borderline items -- deterministic numeric-regime
            # divergence, not corruption. Control: with NO LMCache at all,
            # changing only max_num_seqs (default -> 64) flips 5.10%
            # (121/2374, score -0.67pt) -- vLLM kernels are not
            # batch-invariant, and the connector's 1.14% sits well below
            # the engine's own batch-sensitivity floor. 1.5% covers it.
            mme_max_flip_fraction=0.015,
            # Tempered: the preamble may OPEN a spurious unclosed box marker,
            # so the answer group must not span another begin marker.
            answer_extract_pattern=(
                r"<\|begin_of_box\|>((?:(?!<\|begin_of_box\|>).)*?)<\|end_of_box\|>"
            ),
        ),
    ]
}


def selected_model_keys() -> list[str]:
    """Return the model keys selected via ``LMCACHE_MM_E2E_MODELS``.

    Returns:
        The comma-separated keys from the environment, defaulting to
        ``["qwen2.5-vl-3b"]``.

    Raises:
        KeyError: If a selected key is not registered in ``MODEL_SPECS``.
    """
    raw = os.environ.get("LMCACHE_MM_E2E_MODELS", "qwen2.5-vl-3b")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    for k in keys:
        if k not in MODEL_SPECS:
            raise KeyError(
                f"Unknown model key {k!r}; registered keys: {sorted(MODEL_SPECS)}"
            )
    return keys
