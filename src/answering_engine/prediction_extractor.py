"""Prediction extraction for the Answering Engine.

This module defines :class:`PredictionExtractor`, responsible for
extracting the final predicted answer from a model's raw response text.
It wraps the existing parser functions from ``src.evaluation.parser``
(``parse_multiple_choice`` and ``parse_yes_no_maybe``) without
modifying them, adding automatic parser selection based on the dataset
sample's answer format.

Examples:
    ``"The answer is B"`` → ``1`` (index for B)
    ``"Based on the evidence, yes"`` → ``"yes"``
"""

from __future__ import annotations

from typing import Any

from src.evaluation.parser import parse_multiple_choice, parse_yes_no_maybe


class PredictionExtractor:
    """Extracts predicted answers from model response text.

    ``PredictionExtractor`` is a stateless utility that selects the
    appropriate parser based on the number of answer choices in a
    dataset sample and applies it to extract the predicted answer.

    The extraction logic is fully delegated to the existing parser
    functions in ``src.evaluation.parser``:

    - **3 choices** (PubMedQA): Uses ``parse_yes_no_maybe()``, which
      returns ``"yes"``, ``"no"``, ``"maybe"``, or ``None``.
    - **4+ choices** (MedQA, MedMCQA): Uses ``parse_multiple_choice()``,
      which returns an integer index (0–3) or ``None``.

    A return value of ``None`` indicates that the model response could
    not be parsed into a valid answer.
    """

    @staticmethod
    def extract(response: str, sample: dict[str, Any]) -> object:
        """Extract the predicted answer from a model response.

        Automatically selects the correct parser based on the number
        of choices in the sample.

        Args:
            response: The clean model response text (after stripping
                any reasoning blocks such as ``<think>`` tags).
            sample: The dataset sample in the standardized format,
                containing at least a ``"choices"`` key.

        Returns:
            The extracted prediction: an ``int`` (index for A/B/C/D),
            a ``str`` (``"yes"``/``"no"``/``"maybe"``), or ``None``
            if the response could not be parsed.

        Raises:
            KeyError: If ``sample`` does not contain a ``"choices"``
                key.
        """
        parser = PredictionExtractor._select_parser(sample)
        return parser(response)

    @staticmethod
    def _select_parser(sample: dict[str, Any]):
        """Select the correct parser based on the sample's answer format.

        This mirrors the ``_select_parser`` logic in
        ``src.evaluation.scorer`` without importing or modifying it.

        Args:
            sample: A dataset sample with a ``"choices"`` key.

        Returns:
            The appropriate parser function.
        """
        if len(sample["choices"]) == 3:
            return parse_yes_no_maybe
        return parse_multiple_choice
