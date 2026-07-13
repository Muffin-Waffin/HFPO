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

from configs.config import MAX_NEW_TOKENS, MAX_NEW_TOKENS_COT
from src.evaluation.metrics import accuracy
from src.evaluation.parser import (
    parse_multiple_choice,
    parse_yes_no_maybe,
    LETTER_TO_INDEX,
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
    Remove  blocks from Qwen output.
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


def _generate(model, tokenizer, inputs, max_new_tokens: int = MAX_NEW_TOKENS):
    """
    Generate model output using deterministic decoding.
    """

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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
    cot: bool = False,
    max_new_tokens: int | None = None,
    debug: bool = False,
):
    """
    Evaluate a single dataset sample.

    Returns
    -------
    (
        score,
        prediction,
        clean_response,
        raw_response,
        parse_path,
    )
    """

    # Build the user prompt.
    user_prompt = build_prompt(sample, cot=cot)

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

    # Determine token budget.
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS_COT if cot else MAX_NEW_TOKENS

    # Generate response.
    raw_response = _generate(
        model,
        tokenizer,
        inputs,
        max_new_tokens=max_new_tokens,
    )

    # Remove reasoning block.
    clean_response = _strip_think_tags(raw_response)

    # Parse answer with debug info.
    parser = _select_parser(sample)
    
    prediction, parse_path = _parse_with_debug(parser, clean_response, cot)

    # Compute score.
    score = accuracy(
        prediction,
        sample["answer"],
    )

    if debug:
        return (
            score,
            prediction,
            clean_response,
            raw_response,
            parse_path,
        )
    
    return (
        score,
        prediction,
        clean_response,
    )


def _parse_with_debug(parser, clean_response: str, cot: bool):
    """
    Parse with debug info about which path was taken.
    Returns (prediction, parse_path)
    """
    if parser.__name__ == "parse_multiple_choice":
        response = clean_response.strip()
        # Check for Answer: anchor
        answer_matches = re.findall(r"Answer:\s*\(?([ABCD])\)?", response, re.IGNORECASE)
        if answer_matches:
            return LETTER_TO_INDEX[answer_matches[-1].upper()], "answer_anchor"
        # Fallback
        match = re.search(r"\b([ABCD])\b", response.upper())
        if match:
            return LETTER_TO_INDEX[match.group(1)], "fallback_standalone"
        return None, "no_match"
    else:  # parse_yes_no_maybe
        response = clean_response.strip().lower()
        answer_matches = re.findall(r"Answer:\s*(yes|no|maybe)", response, re.IGNORECASE)
        if answer_matches:
            return answer_matches[-1].lower(), "answer_anchor"
        if "yes" in response:
            return "yes", "fallback_substring"
        if "no" in response:
            return "no", "fallback_substring"
        if "maybe" in response:
            return "maybe", "fallback_substring"
        return None, "no_match"


def score_prompt(
    model,
    tokenizer,
    system_prompt: str,
    dataset: Dataset,
    cot: bool = False,
    max_new_tokens: int | None = None,
    debug: bool = False,
):
    """
    Evaluate a system prompt across an entire dataset.

    Returns
    -------
    Average accuracy in [0, 1].
    If debug=True, returns (accuracy, debug_logs) where debug_logs is a list of dicts.
    """

    if len(dataset) == 0:
        return 0.0

    total_score = 0
    debug_logs = []

    for idx, sample in enumerate(dataset):

        if debug:
            score, prediction, clean_response, raw_response, parse_path = score_sample(
                model,
                tokenizer,
                system_prompt,
                sample,
                cot=cot,
                max_new_tokens=max_new_tokens,
                debug=True,
            )
            debug_logs.append({
                "sample_idx": idx,
                "raw_response": raw_response,
                "clean_response": clean_response,
                "prediction": prediction,
                "ground_truth": sample["answer"],
                "parse_path": parse_path,
                "correct": score == 1.0,
            })
        else:
            score, _, _ = score_sample(
                model,
                tokenizer,
                system_prompt,
                sample,
                cot=cot,
                max_new_tokens=max_new_tokens,
            )

        total_score += score

    accuracy_result = total_score / len(dataset)

    if debug:
        return accuracy_result, debug_logs
    
    return accuracy_result





