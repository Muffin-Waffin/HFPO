"""Prompt template formatting utility for the Prompt Generator
subsystem.

This module defines PromptTemplateBuilder, which is responsible only
for formatting the instruction shown to the reasoning LLM. It does
not call the LLM, clean text, validate prompts, create
PromptCandidates, perform retries, or contain any other business
logic. It only converts a PromptGenerationRequest into a formatted
prompt string using the predefined mutation and crossover templates.
"""

from .models import PromptGenerationRequest, ParentPerformance
from .prompts.prompt_utils.crossover import CROSSOVER_TEMPLATE
import random
from .thinking_directions import THINKING_DIRECTIONS
from .prompts.prompt_utils.mutation import MUTATION_TEMPLATE

# from .prompts.mutation.reasoning import MUTATION_TEMPLATE as REASONING_TEMPLATE
# from .prompts.mutation.elimination import MUTATION_TEMPLATE as ELIMINATION_TEMPLATE
# from .prompts.mutation.differential import MUTATION_TEMPLATE as DIFFERENTIAL_TEMPLATE
# from .prompts.mutation.evidence import MUTATION_TEMPLATE as EVIDENCE_TEMPLATE
# from .prompts.mutation.probability import MUTATION_TEMPLATE as PROBABILITY_TEMPLATE
# from .prompts.mutation.guideline import MUTATION_TEMPLATE as GUIDELINE_TEMPLATE
# from .prompts.mutation.role import MUTATION_TEMPLATE as ROLE_TEMPLATE
# from .prompts.mutation.aggressive import MUTATION_TEMPLATE as AGGRESSIVE_TEMPLATE

# Fixed mutation template that wraps the evolving strategy
FIXED_MUTATION_TEMPLATE: str = """You are assisting in the optimization of prompts for a medical question-answering system through an evolutionary optimization process.

Task Description
----------------
{task_description}

Parent Prompt
-------------
{parent_prompt}

{parent_performance}

Mutation Strategy
-----------------
{strategy}

Output Requirements
-------------------
Return ONLY the new prompt.

Do NOT:
- explain your changes;
- compare the new prompt with the parent;
- include Markdown;
- include code fences;
- number the output;
- surround the prompt with quotation marks;
- mention mutation, evolution, optimization, prompt engineering, or genetic algorithms in the generated prompt.

Produce exactly one new prompt and nothing else."""


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
    __slots__ = ("_last_mutation_operator", "_mutation_manager")

    def __init__(self, mutation_manager=None):
        self._last_mutation_operator = None
        self._mutation_manager = mutation_manager

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

    def _format_performance_summary(self, performance: ParentPerformance | None, operator_name: str | None = None) -> str:
        """Format a performance summary for inclusion in the prompt template."""
        if performance is None:
            return "Parent Performance Summary\n--------------------------\n(No performance data available.)"
        
        lines = [
            "Parent Performance Summary",
            "--------------------------",
            f"Average Fitness: {performance.average_fitness:.3f}",
            "",
            "Per-Objective Fitness:",
        ]
        
        for obj, score in performance.per_objective.items():
            lines.append(f"  {obj}: {score:.3f}")
        
        lines.extend([
            "",
            f"Strongest Objective: {performance.best_objective} ({performance.per_objective[performance.best_objective]:.3f})",
            f"Weakest Objective: {performance.worst_objective} ({performance.per_objective[performance.worst_objective]:.3f})",
        ])
        
        # Add interpretation
        scores = list(performance.per_objective.values())
        if len(scores) > 1:
            score_range = max(scores) - min(scores)
            if score_range > 0.15:
                lines.extend([
                    "",
                    "Interpretation: Performance is inconsistent across objectives.",
                    f"  {performance.best_objective} is substantially higher than {performance.worst_objective}.",
                ])
            else:
                lines.extend([
                    "",
                    "Interpretation: Performance is relatively consistent across objectives.",
                ])
        
        if operator_name:
            lines.extend([
                "",
                f"Current Mutation Operator: {operator_name}",
            ])
        
        return "\n".join(lines)

    def _build_mutation(
        self,
        request: PromptGenerationRequest,
    ) -> tuple[str, str]:
        """Build the mutation instruction."""

        performance_summary = self._format_performance_summary(
            request.parent_a_performance,
            None,
        )

        if self._mutation_manager is not None:
            # Adaptive mutation (future)
            selected = self._mutation_manager.select()
            self._last_mutation_operator = selected.id
            thinking_direction = selected.strategy

        else:
            thinking_direction = "\n".join(
                f"- {direction}"
                for direction in random.sample(THINKING_DIRECTIONS, k=2)
            )

            self._last_mutation_operator = "random_thinking_directions"

        instruction = MUTATION_TEMPLATE.format(
            task_description=request.task_description,
            parent_prompt=request.parent_a.text,
            parent_performance=performance_summary,
            thinking_direction=thinking_direction,
        )

        return self._last_mutation_operator, instruction
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