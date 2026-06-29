"""
src/ga/mutation.py

Lightweight prompt mutations for the FedGAPrompt genetic algorithm.

Four mutation types are supported:
    - replace_adjective  : Swaps one adjective for another.
    - insert_instruction : Inserts one additional instruction sentence.
    - remove_sentence    : Drops one sentence at random.
    - shuffle_sentences  : Swaps two sentences at random positions.

No LLM is called. All mutations use Python's random module.
Each mutation returns a new string; the original is never modified.
"""

import random

# ── word banks ────────────────────────────────────────────────────────────────

_ADJECTIVES = [
    "accurate", "careful", "concise", "correct", "detailed",
    "expert", "precise", "rigorous", "systematic", "thorough",
]

_INSTRUCTIONS = [
    "Think step by step before answering.",
    "Only respond with the answer letter.",
    "Consider all options before choosing.",
    "Eliminate incorrect options first.",
    "Base your answer on clinical evidence.",
    "Respond with a single uppercase letter.",
    "Do not explain your reasoning.",
    "Focus on the most likely diagnosis.",
    "Apply standard medical guidelines.",
    "Choose the best answer from the options given.",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split a prompt into sentences on period boundaries."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return sentences


def _join_sentences(sentences: list[str]) -> str:
    """Rejoin sentences into a single prompt string."""
    return ". ".join(sentences) + "."


# ── mutation operators ────────────────────────────────────────────────────────

def replace_adjective(prompt: str) -> str:
    """
    Replace the first recognized adjective in the prompt with a
    randomly chosen alternative from the word bank.

    Falls back to appending a new adjective instruction if none found.
    """
    words = prompt.split()
    for i, word in enumerate(words):
        if word.lower().rstrip(".,;") in _ADJECTIVES:
            replacement = random.choice(
                [a for a in _ADJECTIVES if a != word.lower().rstrip(".,;")]
            )
            words[i] = replacement
            return " ".join(words)

    # Fallback: prepend a random adjective instruction.
    adjective = random.choice(_ADJECTIVES)
    return f"Be {adjective}. {prompt}"


def insert_instruction(prompt: str) -> str:
    """
    Insert one randomly chosen instruction sentence into the prompt.

    The instruction is inserted at a random sentence boundary.
    """
    sentences = _split_sentences(prompt)
    instruction = random.choice(_INSTRUCTIONS)
    position = random.randint(0, len(sentences))
    sentences.insert(position, instruction)
    return _join_sentences(sentences)


def remove_sentence(prompt: str) -> str:
    """
    Remove one sentence at random.

    If the prompt has only one sentence, returns it unchanged
    to avoid producing an empty prompt.
    """
    sentences = _split_sentences(prompt)
    if len(sentences) <= 1:
        return prompt
    index = random.randrange(len(sentences))
    sentences.pop(index)
    return _join_sentences(sentences)


def shuffle_sentences(prompt: str) -> str:
    """
    Swap two randomly chosen sentences.

    If the prompt has fewer than two sentences, returns it unchanged.
    """
    sentences = _split_sentences(prompt)
    if len(sentences) < 2:
        return prompt
    i, j = random.sample(range(len(sentences)), 2)
    sentences[i], sentences[j] = sentences[j], sentences[i]
    return _join_sentences(sentences)


# ── public API ────────────────────────────────────────────────────────────────

_MUTATIONS = [
    replace_adjective,
    insert_instruction,
    remove_sentence,
    shuffle_sentences,
]


def mutate(prompt: str) -> str:
    """
    Apply one randomly chosen mutation to a prompt string.

    Parameters
    ----------
    prompt : The system prompt to mutate.

    Returns
    -------
    A new mutated prompt string.
    """
    mutation_fn = random.choice(_MUTATIONS)
    return mutation_fn(prompt)