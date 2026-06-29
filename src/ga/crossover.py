"""
src/ga/crossover.py

One-point crossover for the FedGAPrompt genetic algorithm.

Splits two parent prompts at the sentence level and swaps
their second halves to produce two children.

Example
-------
Parent A: "S1. S2. S3."
Parent B: "SA. SB. SC."

Child A:  "S1. SB. SC."
Child B:  "SA. S2. S3."
"""

import random

from src.ga.population import PromptIndividual


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split a prompt into sentences on period boundaries."""
    return [s.strip() for s in text.split(".") if s.strip()]


def _join_sentences(sentences: list[str]) -> str:
    """Rejoin sentences into a single prompt string."""
    return ". ".join(sentences) + "."


# ── public API ────────────────────────────────────────────────────────────────

def one_point_crossover(
    parent_a: PromptIndividual,
    parent_b: PromptIndividual,
) -> tuple[PromptIndividual, PromptIndividual]:
    """
    Produce two children by swapping sentence-level halves of two parents.

    The crossover point is chosen randomly within the shorter parent's
    sentence count so both children are always non-empty.

    If either parent has only one sentence, children are returned as
    copies of the parents (crossover has no effect).

    Parameters
    ----------
    parent_a : First parent PromptIndividual.
    parent_b : Second parent PromptIndividual.

    Returns
    -------
    Tuple of two new PromptIndividuals with fitness reset to 0.0.
    """
    sentences_a = _split_sentences(parent_a.prompt)
    sentences_b = _split_sentences(parent_b.prompt)

    # Need at least 2 sentences in each parent for a meaningful split.
    if len(sentences_a) < 2 or len(sentences_b) < 2:
        return (
            PromptIndividual(prompt=parent_a.prompt),
            PromptIndividual(prompt=parent_b.prompt),
        )

    # Crossover point is chosen within the shorter parent's range.
    max_point = min(len(sentences_a), len(sentences_b)) - 1
    point = random.randint(1, max_point)

    child_a_sentences = sentences_a[:point] + sentences_b[point:]
    child_b_sentences = sentences_b[:point] + sentences_a[point:]

    return (
        PromptIndividual(prompt=_join_sentences(child_a_sentences)),
        PromptIndividual(prompt=_join_sentences(child_b_sentences)),
    )