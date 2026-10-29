"""Write two random-weight LoRA adapters for Qwen2.5-1.5B.

peft's default init zeroes lora_B, which makes an adapter equal to the base
model, so both matrices are random here. k_proj and v_proj are targeted
because only a LoRA that changes K/V makes a cache collision corrupt output.

Usage: make_adapters.py <out_dir> <base_model_path>
"""

import json
import pathlib
import sys

import torch
from safetensors.torch import save_file

OUT, BASE = pathlib.Path(sys.argv[1]), sys.argv[2]
RANK, ALPHA, STD = 8, 16, 0.03
LAYERS, HIDDEN, KV_DIM = 28, 1536, 256
SHAPES = {  # module -> (in, out)
    "q_proj": (HIDDEN, HIDDEN),
    "k_proj": (HIDDEN, KV_DIM),
    "v_proj": (HIDDEN, KV_DIM),
    "o_proj": (HIDDEN, HIDDEN),
}

for name, seed in (("lora-a", 1), ("lora-b", 2)):
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in range(LAYERS):
        for module, (fan_in, fan_out) in SHAPES.items():
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            tensors[f"{prefix}.lora_A.weight"] = (
                torch.randn(RANK, fan_in, generator=gen) * STD
            ).to(torch.bfloat16)
            tensors[f"{prefix}.lora_B.weight"] = (
                torch.randn(fan_out, RANK, generator=gen) * STD
            ).to(torch.bfloat16)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": BASE,
                "r": RANK,
                "lora_alpha": ALPHA,
                "lora_dropout": 0.0,
                "target_modules": list(SHAPES),
                "bias": "none",
                "fan_in_fan_out": False,
                "modules_to_save": None,
            }
        )
    )
    print("wrote", d)
