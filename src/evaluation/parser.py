import re


LETTER_TO_INDEX = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}


def parse_multiple_choice(response: str):
    """
    Extract A/B/C/D from the model response and
    convert it to an integer index.

    First tries to find an explicit "Answer: X" pattern
    (case-insensitive, near the end of the response).
    Falls back to finding the first standalone A/B/C/D
    for backward compatibility with non-CoT responses.
    """

    response = response.strip()

    # First, try to find "Answer: X" pattern (case-insensitive)
    # Use findall to get all matches and take the last one (final answer)
    answer_matches = re.findall(r"Answer:\s*\(?([ABCD])\)?", response, re.IGNORECASE)
    if answer_matches:
        return LETTER_TO_INDEX[answer_matches[-1].upper()]

    # Fallback: first standalone A/B/C/D for backward compatibility
    match = re.search(r"\b([ABCD])\b", response.upper())
    if match:
        return LETTER_TO_INDEX[match.group(1)]

    return None


def parse_yes_no_maybe(response: str):
    """
    Extract yes/no/maybe from the model response.

    First tries to find an explicit "Answer: yes/no/maybe" pattern
    (case-insensitive, near the end of the response).
    Falls back to substring search for backward compatibility.
    """

    response = response.strip().lower()

    # First, try to find "Answer: yes/no/maybe" pattern
    answer_matches = re.findall(r"Answer:\s*(yes|no|maybe)", response, re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1].lower()

    # Fallback: substring search (original behavior)
    if "yes" in response:
        return "yes"

    if "no" in response:
        return "no"

    if "maybe" in response:
        return "maybe"

    return None