"""
src/prompts/builder.py

Builds the user prompt from a standardized dataset sample.

This module is intentionally independent of the system prompt.
The system prompt is supplied separately through the model's
chat template during evaluation.
"""

OPTION_LABELS = ["A", "B", "C", "D", "E", "F"]


def _format_choices(choices: list[str]) -> str:
    """
    Format answer choices into:

    A. ...
    B. ...
    C. ...
    """

    lines = []

    for label, choice in zip(OPTION_LABELS, choices):
        lines.append(f"{label}. {choice}")

    return "\n".join(lines)


def build_prompt(sample: dict) -> str:
    """
    Build the user prompt from a standardized dataset sample.

    Parameters
    ----------
    sample : dict
        {
            "question": ...,
            "context": ...,
            "choices": ...
        }

    Returns
    -------
    str
        Prompt ready to be inserted as the user message
        in the chat template.
    """

    prompt = ""

    if sample["context"]:
        prompt += f"Context:\n{sample['context']}\n\n"

    prompt += f"Question:\n{sample['question']}\n\n"

    prompt += "Options:\n"
    prompt += _format_choices(sample["choices"])

    prompt += "\n\nAnswer (respond with ONLY one uppercase letter: A, B, C, or D):"

    return prompt