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
    """

    key: str
    hf_id: str
    modalities: frozenset = frozenset({"image"})
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.6
    extra_suites: frozenset = field(default_factory=frozenset)


MODEL_SPECS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec(
            key="qwen2.5-vl-3b",
            hf_id="Qwen/Qwen2.5-VL-3B-Instruct",
        ),
        ModelSpec(
            key="qwen2-vl-2b",
            hf_id="Qwen/Qwen2-VL-2B-Instruct",
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
