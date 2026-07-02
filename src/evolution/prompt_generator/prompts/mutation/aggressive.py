"""Mutation prompt template for the Prompt Generator subsystem.

This module defines the mutation prompt template shown to the
reasoning LLM whenever the genetic algorithm wants to produce a
substantially different, high-diversity mutation of a parent prompt.
It contains no generation logic, no API calls, no PromptCandidate
creation, no validation, and no cleaning. It only defines a reusable,
format()-style prompt template.
"""

__all__ = ["MUTATION_TEMPLATE"]


# This template is presented to the reasoning LLM whenever HFPO performs
# an aggressive, high-diversity mutation. The model receives a single
# parent prompt and is instructed to produce exactly one substantially
# different variant, free to change wording, structure, ordering,
# reasoning strategy, and role, while preserving only the parent's
# underlying objective. The resulting text should be suitable for
# direct processing by PromptCleaner and PromptValidator.
MUTATION_TEMPLATE: str = """You are assisting in the optimization of prompts for a medical question-answering system through an evolutionary optimization process.

Task Description
----------------
{task_description}

Parent Prompt
-------------
{parent_prompt}

Your objective is to create exactly ONE substantially different version of the parent prompt.

The new prompt should:
- Preserve only the original objective: answering medical multiple-choice questions.
- Be free to change wording, sentence structure, ordering of instructions, reasoning strategy, and assigned role or persona.
- Explore an approach that is meaningfully different from the parent prompt, rather than a variation on the same theme.
- Remain a coherent, well-formed prompt that a language model could follow directly.
- Avoid introducing instructions unrelated to answering the medical question.
- Avoid changing the underlying objective or the fact that a single answer must be produced.
- Avoid inventing or hallucinating medical knowledge that is not implied by the parent prompt.

Strict Constraint
------------------
Do NOT produce a minor paraphrase of the parent prompt. Small wording changes, synonym substitutions, or reordering of the same sentences are NOT acceptable. The new prompt must differ substantially from the parent in its structure, phrasing, reasoning approach, or role, while still preserving the underlying objective.

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
- mention mutation, evolution, optimization, or genetic algorithms in the generated prompt.

Produce exactly one new prompt and nothing else.

"""