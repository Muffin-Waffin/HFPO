"""Interface definitions for the HFPO Prompt Generator subsystem.

This module defines the abstract interfaces used by the Prompt Generator.
By depending on Protocols rather than concrete implementations, HFPO
remains independent of any specific LLM provider or inference backend.
Any object implementing these interfaces can be used by the
PromptGenerator without modification.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ReasoningLLM(Protocol):
    """Protocol for reasoning language models used by HFPO.

    A ReasoningLLM is responsible for generating improved prompt text from
    a structured instruction supplied by the PromptGenerator. The model may
    be local or remote, open-source or proprietary, provided it satisfies
    this interface.

    Implementations may include models such as BioMistral, Qwen, Llama,
    GPT-OSS, Claude, GPT, or any future reasoning model.

    Notes:
        - The returned string should contain only the generated prompt
          text.
        - The PromptGenerator is responsible for cleaning and validating
          the returned text before it becomes a PromptCandidate.
    """

    def generate(
        self,
        prompt: str,
        temperature: float,
    ) -> str:
        """Generate prompt text from an instruction.

        Args:
            prompt: The fully formatted instruction presented to the
                reasoning model.
            temperature: Sampling temperature controlling generation
                diversity.

        Returns:
            The generated prompt text.
        """
        ...


@runtime_checkable
class PromptCleaner(Protocol):
    """Protocol for prompt cleaning implementations."""

    def clean(self, text: str) -> str:
        """Normalize raw LLM output.

        Args:
            text: Raw text returned by the reasoning model.

        Returns:
            Cleaned prompt text.
        """
        ...


@runtime_checkable
class PromptValidator(Protocol):
    """Protocol for prompt validation implementations."""

    def validate(
        self,
        prompt_text: str,
        existing_prompts: set[str],
    ) -> None:
        """Validate a generated prompt.

        Implementations should raise an exception if the prompt is
        invalid.

        Args:
            prompt_text: Prompt text to validate.
            existing_prompts: Existing prompt texts used to detect
                duplicates.

        Raises:
            ValueError: If the prompt is invalid.
        """
        ...


__all__ = [
    "ReasoningLLM",
    "PromptCleaner",
    "PromptValidator",
]