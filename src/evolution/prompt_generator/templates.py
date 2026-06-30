"""Prompt template formatting utility for the Prompt Generator
subsystem.

This module defines PromptTemplateBuilder, which is responsible only
for formatting the instruction shown to the reasoning LLM. It does
not call the LLM, clean text, validate prompts, create
PromptCandidates, perform retries, or contain any other business
logic. It only converts a PromptGenerationRequest into a formatted
prompt string using the predefined mutation and crossover templates.
"""

from .models import PromptGenerationRequest
from .prompts.crossover import CROSSOVER_TEMPLATE
from .prompts.mutation import MUTATION_TEMPLATE


class PromptTemplateBuilder:
    """Builds formatted LLM instruction strings from generation
    requests.

    PromptTemplateBuilder selects between the mutation template and
    the crossover template based on whether a PromptGenerationRequest
    represents a crossover, and formats the selected template with
    the request's task description and parent prompt text. It is a
    stateless class that performs no LLM interaction, no cleaning, no
    validation, no PromptCandidate construction, no retries, and no
    other side effects.
    """

    __slots__ = ()

    def build(self, request: PromptGenerationRequest) -> str:
        """Builds the formatted prompt instruction for a request.

        Selects the crossover template if the request represents a
        crossover, or the mutation template otherwise, and returns the
        fully formatted instruction string.

        Args:
            request: The generation request describing the parent(s),
                task, and generation context to format the template
                with.

        Returns:
            The formatted prompt instruction string ready to be sent
            to the reasoning LLM.

        Raises:
            TypeError: If request is not a PromptGenerationRequest.
            RuntimeError: If request represents a crossover but is
                missing parent_b.
        """
        if not isinstance(request, PromptGenerationRequest):
            raise TypeError(
                "request must be a PromptGenerationRequest, got "
                f"{type(request).__name__}."
            )
        if request.parent_b is not None:
            return self._build_crossover(request)
        return self._build_mutation(request)

    def _build_mutation(self, request: PromptGenerationRequest) -> str:
        """Formats the mutation template for a request.

        Args:
            request: The generation request providing the task
                description and the single parent prompt to mutate.

        Returns:
            The MUTATION_TEMPLATE formatted with the request's task
            description and parent_a's prompt text.
        """
        return MUTATION_TEMPLATE.format(
            task_description=request.task_description,
            parent_prompt=request.parent_a.text,
        )

    def _build_crossover(self, request: PromptGenerationRequest) -> str:
        """Formats the crossover template for a request.

        Args:
            request: The generation request providing the task
                description and both parent prompts to combine.

        Returns:
            The CROSSOVER_TEMPLATE formatted with the request's task
            description and both parents' prompt text.

        Raises:
            RuntimeError: If request is missing parent_b.
        """
        if request.parent_b is None:
            raise RuntimeError(
                "Crossover request is missing parent_b."
            )
        return CROSSOVER_TEMPLATE.format(
            task_description=request.task_description,
            parent_a=request.parent_a.text,
            parent_b=request.parent_b.text,
        )