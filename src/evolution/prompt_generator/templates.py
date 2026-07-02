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
import random

from .prompts.mutation.reasoning import MUTATION_TEMPLATE as REASONING_TEMPLATE
from .prompts.mutation.elimination import MUTATION_TEMPLATE as ELIMINATION_TEMPLATE
from .prompts.mutation.differential import MUTATION_TEMPLATE as DIFFERENTIAL_TEMPLATE
from .prompts.mutation.evidence import MUTATION_TEMPLATE as EVIDENCE_TEMPLATE
from .prompts.mutation.probability import MUTATION_TEMPLATE as PROBABILITY_TEMPLATE
from .prompts.mutation.guideline import MUTATION_TEMPLATE as GUIDELINE_TEMPLATE
from .prompts.mutation.role import MUTATION_TEMPLATE as ROLE_TEMPLATE
from .prompts.mutation.aggressive import MUTATION_TEMPLATE as AGGRESSIVE_TEMPLATE



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
    __slots__ = ("_last_mutation_operator",)

    def __init__(self):
        self._last_mutation_operator = None

    def build(self, request: PromptGenerationRequest) -> tuple[str, str]:
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
        """Formats a randomly selected mutation template."""
        MUTATION_TEMPLATES = [
        ("reasoning", REASONING_TEMPLATE),
        ("elimination", ELIMINATION_TEMPLATE),
        ("differential", DIFFERENTIAL_TEMPLATE),
        ("evidence", EVIDENCE_TEMPLATE),
        ("probability", PROBABILITY_TEMPLATE),
        ("guideline", GUIDELINE_TEMPLATE),
        ("role", ROLE_TEMPLATE),
        ("aggressive", AGGRESSIVE_TEMPLATE),
    ]
        operator_name, template = random.choice(MUTATION_TEMPLATES)

        instruction = template.format(
            task_description=request.task_description,
            parent_prompt=request.parent_a.text,
        )

        return operator_name, instruction
    def _build_crossover(self, request: PromptGenerationRequest) -> tuple[str, str]:
        """Formats the crossover template for a request."""
        if request.parent_b is None:
            raise RuntimeError(
                "Crossover request is missing parent_b."
            )
        instruction = CROSSOVER_TEMPLATE.format(
            task_description=request.task_description,
            parent_a=request.parent_a.text,
            parent_b=request.parent_b.text,
        )
        return "crossover", instruction
