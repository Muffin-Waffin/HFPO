"""
src/evaluation/scorer.py

Central evaluation component for FedGAPrompt.

Pipeline:
Dataset Sample
    ↓
Prompt Builder
    ↓
Qwen3
    ↓
Parser
    ↓
Metrics

This module provides the fitness function that will later be used
by the Genetic Algorithm.
"""

import re
from typing import Any

import torch
from datasets import Dataset

from configs.config import MAX_NEW_TOKENS
from src.evaluation.metrics import accuracy
from src.evaluation.parser import (
    parse_multiple_choice,
    parse_yes_no_maybe,
)
from src.prompts.builder import build_prompt


def _build_messages(system_prompt: str, user_prompt: str):
    """
    Build chat messages for Qwen.
    """
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def _strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks from Qwen output.
    """
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def _select_parser(sample: dict[str, Any]):
    """
    Select the correct parser based on dataset type.

    PubMedQA:
        yes / no / maybe

    MedQA / MedMCQA:
        A / B / C / D
    """

    if len(sample["choices"]) == 3:
        return parse_yes_no_maybe

    return parse_multiple_choice


def _generate(model, tokenizer, inputs):
    """
    Generate model output using deterministic decoding.
    """

    with torch.inference_mode():

        output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=None,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


def score_sample(
    model,
    tokenizer,
    system_prompt: str,
    sample: dict[str, Any],
):
    """
    Evaluate a single dataset sample.

    Returns
    -------
    (
        score,
        prediction,
        clean_response,
    )
    """

    # Build the user prompt.
    user_prompt = build_prompt(sample)

    # Build chat conversation.
    messages = _build_messages(
        system_prompt,
        user_prompt,
    )

    # Apply Qwen chat template.
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",

        enable_thinking=False,   # add this
    ).to(model.device)

    # Generate response.
    response = _generate(
        model,
        tokenizer,
        inputs,
    )

    # Remove reasoning block.
    clean_response = _strip_think_tags(response)

    # Parse answer.
    parser = _select_parser(sample)

    prediction = parser(clean_response)

    # Compute score.
    score = accuracy(
        prediction,
        sample["answer"],
    )

    return (
        score,
        prediction,
        clean_response,
    )


def score_prompt(
    model,
    tokenizer,
    system_prompt: str,
    dataset: Dataset,
):
    """
    Evaluate a system prompt across an entire dataset.

    Returns
    -------
    Average accuracy in [0, 1].
    """

    if len(dataset) == 0:
        return 0.0

    total_score = 0

    for sample in dataset:

        score, _, _ = score_sample(
            model,
            tokenizer,
            system_prompt,
            sample,
        )

        total_score += score

    return total_score / len(dataset)





