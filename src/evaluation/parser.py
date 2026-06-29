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
    """

    response = response.upper()

    match = re.search(r"\b([ABCD])\b", response)

    if match:
        return LETTER_TO_INDEX[match.group(1)]

    return None


def parse_yes_no_maybe(response: str):
    """
    Extract yes/no/maybe from the model response.
    """

    response = response.lower()

    if "yes" in response:
        return "yes"

    if "no" in response:
        return "no"

    if "maybe" in response:
        return "maybe"

    return None