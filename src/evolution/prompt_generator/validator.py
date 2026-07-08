"""Validation utility for the Prompt Generator subsystem.

This module defines PromptValidator, which is responsible only for
determining whether a cleaned prompt is acceptable for inclusion in
the evolutionary population. It performs no cleaning, no LLM
interaction, no PromptCandidate creation, no template generation, and
no retries.
"""

from .models import PromptGenerationRequest


class PromptValidator:
    """Validates cleaned prompt text against population acceptance
    rules.

    PromptValidator checks a single cleaned prompt against a fixed,
    ordered set of validation rules: non-emptiness, length bounds,
    difference from both parent prompts, absence from the set of
    already-existing prompt texts, and a minimum word count. It is a
    stateless class that performs no cleaning, no LLM interaction, no
    PromptCandidate construction, no template generation, no retries,
    no logging, and no other side effects. Validation either succeeds
    silently or fails by raising ValueError with an informative
    message.
    """

    __slots__ = ()
    def __call__(
        self,
        prompt_text: str,
        request: PromptGenerationRequest,
    ) -> None:
        """Validate cleaned prompt text.

        Args:
            prompt_text: The cleaned prompt text to validate.
            request: The generation request associated with the prompt.

        Raises:
            TypeError: If the supplied arguments have incorrect types.
            ValueError: If the prompt fails validation.
        """
        self.validate(prompt_text, request)

    def validate(
        self,
        prompt_text: str,
        request: PromptGenerationRequest,
    ) -> None:
        """Validates a cleaned prompt against all acceptance rules.

        Runs the full validation sequence in order: non-emptiness,
        length bounds, difference from both parent prompts, absence
        from existing prompt texts, and minimum word count.

        Args:
            prompt_text: The cleaned prompt text to validate.
            request: The generation request that produced
                prompt_text, providing the parent candidates and the
                set of already-existing prompt texts to validate
                against.

        Returns:
            None, if validation succeeds.

        Raises:
            ValueError: If prompt_text fails any validation rule.
        """
        if not isinstance(prompt_text, str):
            raise TypeError(
                f"prompt_text must be a string, got {type(prompt_text).__name__}."
            )

        if not isinstance(request, PromptGenerationRequest):
            raise TypeError(
                "request must be a PromptGenerationRequest."
            )

        self._validate_not_empty(prompt_text)
        self._validate_length(prompt_text)
        self._validate_parent_difference(prompt_text, request)
        self._validate_duplicate(prompt_text, request)
        self._validate_word_count(prompt_text)

    def _validate_not_empty(self, prompt_text: str) -> None:
        """Validates that the prompt is not empty or whitespace-only.

        Args:
            prompt_text: The cleaned prompt text to validate.

        Raises:
            ValueError: If prompt_text is an empty string or consists
                only of whitespace.
        """
        if not prompt_text:
            raise ValueError("Prompt must not be empty.")
        if not prompt_text.strip():
            raise ValueError("Prompt must not contain only whitespace.")

    def _validate_length(self, prompt_text: str) -> None:
        """Validates that the prompt length is within accepted bounds.

        Args:
            prompt_text: The cleaned prompt text to validate.

        Raises:
            ValueError: If prompt_text is shorter than 20 characters
                or longer than 4000 characters.
        """
        if len(prompt_text) < 20:
            raise ValueError(
                "Prompt length must be at least 20 characters, got "
                f"{len(prompt_text)}."
            )
        if len(prompt_text) > 4000:
            raise ValueError(
                "Prompt length must not exceed 4000 characters, got "
                f"{len(prompt_text)}."
            )

    def _validate_parent_difference(
        self,
        prompt_text: str,
        request: PromptGenerationRequest,
    ) -> None:
        """Validates that the prompt differs from both parent prompts.

        Args:
            prompt_text: The cleaned prompt text to validate.
            request: The generation request providing parent_a and,
                if present, parent_b.

        Raises:
            ValueError: If prompt_text exactly equals parent_a.text,
                or, when parent_b is present, if prompt_text exactly
                equals parent_b.text.
        """
        if prompt_text == request.parent_a.text:
            raise ValueError("Prompt must not exactly equal parent_a.text.")
        if (
            request.parent_b is not None
            and prompt_text == request.parent_b.text
        ):
            raise ValueError("Prompt must not exactly equal parent_b.text.")

    def _validate_duplicate(
        self,
        prompt_text: str,
        request: PromptGenerationRequest,
    ) -> None:
        """Validates that the prompt does not already exist.

        Args:
            prompt_text: The cleaned prompt text to validate.
            request: The generation request providing the set of
                already-existing prompt texts.

        Raises:
            ValueError: If prompt_text is already present in
                request.existing_prompt_texts.
        """
        if prompt_text in request.existing_prompt_texts:
            raise ValueError(
                "Prompt must not already exist inside "
                "existing_prompt_texts."
            )

    def _validate_word_count(self, prompt_text: str) -> None:
        """Validates that the prompt contains a minimum number of
        words.

        Args:
            prompt_text: The cleaned prompt text to validate.

        Raises:
            ValueError: If prompt_text contains fewer than 3 words,
                where words are determined by splitting on whitespace.
        """
        word_count = len(prompt_text.split())
        if word_count < 3:
            raise ValueError(
                f"Prompt must contain at least 3 words, got {word_count}."
            )