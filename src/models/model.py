"""LoRA-wrapped decision LLM (Qwen3.5-2B, vision tower kept for multimodal)."""
from __future__ import annotations

from typing import Any, Dict


def lora_config(cfg: Dict[str, Any]):
    from peft import LoraConfig

    return LoraConfig(
        r=int(cfg.get("lora_r", 32)),
        lora_alpha=int(cfg.get("lora_alpha", 64)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=cfg.get("lora_target_modules", "all-linear"),
        bias="none",
    )


def build_model(cfg: Dict[str, Any]):
    """Load the base causal LM and wrap it with LoRA. Returns (peft_model, processor)."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoProcessor

    name = cfg["model_name"]
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    model = get_peft_model(model, lora_config(cfg))
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(name)
    return model, processor
