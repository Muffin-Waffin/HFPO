from difflib import SequenceMatcher


def too_similar(
    candidate: str,
    existing_prompts: set[str],
    threshold: float = 0.90,
) -> bool:
    """Return True if candidate is too similar to an existing prompt."""

    candidate = candidate.strip()

    for prompt in existing_prompts:
        score = SequenceMatcher(
            None,
            candidate,
            prompt.strip(),
        ).ratio()

        if score >= threshold:
            return True

    return False