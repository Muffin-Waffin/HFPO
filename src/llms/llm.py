from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import torch

from configs.config import MODEL_REGISTRY, DEFAULT_MODEL


def get_model_config(model_key: str = None) -> dict:
    key = model_key or DEFAULT_MODEL
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {key}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]


def load_llm(model_key: str = None):
    config = get_model_config(model_key)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_quant_type=config["quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, config["compute_dtype"]),
        bnb_4bit_use_double_quant=config["use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=quant_config,
        device_map=config["device_map"],
        dtype=getattr(torch, config["dtype"]),
    )

    model.eval()

    return model, tokenizer


def get_generation_config(model_key: str = None) -> dict:
    config = get_model_config(model_key)
    return {
        "max_new_tokens": config["max_new_tokens"],
        "temperature": config["temperature"],
    }