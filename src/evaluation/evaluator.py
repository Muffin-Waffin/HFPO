from __future__ import annotations

from typing import Any, Protocol


class Evaluator(Protocol):
    """Protocol implemented by all HFPO evaluation backends."""

    def score_sample(self, prompt_text: str, sample: Any) -> int:
        """Return 1 if the prediction is correct, else 0."""
        ...