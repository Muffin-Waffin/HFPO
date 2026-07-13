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


def build_prompt(sample: dict, cot: bool = False) -> str:
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
    cot : bool, default False
        If True, append a CoT-friendly instruction asking the model to
        reason step by step and end with "Answer: X". If False (default),
        use the original instruction requesting only the answer letter/word.

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

    if len(sample["choices"]) == 3:
        if cot:
            prompt += (
                "\n\nThink step by step, then end your response with "
                "'Answer: yes', 'Answer: no', or 'Answer: maybe'."
            )
        else:
            prompt += (
                "\n\nAnswer (respond with ONLY one word: "
                "yes, no, or maybe):"
            )
    else:
        if cot:
            prompt += (
                "\n\nThink step by step, then end your response with "
                "'Answer: X' where X is the correct letter."
            )
        else:
            prompt += (
                "\n\nAnswer (respond with ONLY one uppercase letter: "
                "A, B, C, or D):"
            )

    return prompt