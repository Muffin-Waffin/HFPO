"""Mutation prompt template for the Prompt Generator subsystem.

This module defines the mutation prompt template shown to the
reasoning LLM whenever the genetic algorithm wants to mutate a parent
prompt. It contains no generation logic, no API calls, no
PromptCandidate creation, no validation, and no cleaning. It only
defines a reusable, format()-style prompt template.
"""

__all__ = ["MUTATION_TEMPLATE"]


# This template is presented to the reasoning LLM whenever HFPO performs
# a mutation operation. The model receives a single parent prompt and is
# instructed to produce exactly one improved variant while preserving
# the parent's purpose. The resulting text should be suitable for direct
# processing by PromptCleaner and PromptValidator.
MUTATION_TEMPLATE: str = """You are assisting in the optimization of prompts for a medical question-answering system through an evolutionary optimization process.

Task Description
----------------
{task_description}

Parent Prompt
-------------
{parent_prompt}

Your objective is to create exactly ONE improved version of the parent prompt.

The new prompt should:
- Preserve the original task and overall intent.
- Retain effective instructions from the parent whenever possible.
- Improve reasoning quality and logical consistency.
- Improve medical accuracy and precision.
- Improve clarity and readability.
- Improve robustness across diverse medical questions.
- Be concise while remaining complete.
- Avoid unnecessary verbosity or repetition.
- Avoid introducing unrelated instructions.
- Avoid changing the underlying task.
- Avoid inventing or hallucinating medical knowledge that is not implied by the parent prompt.

Output Requirements
-------------------
Return ONLY the improved prompt.

Do NOT:
- explain your changes;
- compare the new prompt with the parent;
- include Markdown;
- include code fences;
- number the output;
- surround the prompt with quotation marks;
- mention mutation, evolution, optimization, or genetic algorithms in the generated prompt.

Produce exactly one improved prompt and nothing else.
"""