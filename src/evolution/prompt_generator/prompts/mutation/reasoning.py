"""Mutation prompt template for the Prompt Generator subsystem.

This module defines the mutation prompt template shown to the
reasoning LLM whenever the genetic algorithm wants to mutate a parent
prompt by altering its reasoning strategy. It contains no generation
logic, no API calls, no PromptCandidate creation, no validation, and
no cleaning. It only defines a reusable, format()-style prompt
template.
"""

__all__ = ["MUTATION_TEMPLATE"]


# This template is presented to the reasoning LLM whenever HFPO performs
# a reasoning-strategy mutation. The model receives a single parent prompt
# and is instructed to produce exactly one variant that changes how the
# downstream model is asked to reason about medical multiple-choice
# questions while preserving the parent's underlying objective. The
# resulting text should be suitable for direct processing by
# PromptCleaner and PromptValidator.
MUTATION_TEMPLATE: str = """You are assisting in the optimization of prompts for a medical question-answering system through an evolutionary optimization process.

Task Description
----------------
{task_description}

Parent Prompt
-------------
{parent_prompt}

Your objective is to create exactly ONE new version of the parent prompt that changes the reasoning strategy used by the answering model.

The new prompt should:
- Preserve the original task of answering medical multiple-choice questions.
- Preserve the parent's overall objective while substantially changing HOW the answering model is instructed to reason.
- Introduce a clinically meaningful reasoning strategy that is noticeably different from the parent.
- Possible examples include (but are not limited to):
  - differential diagnosis
  - systematic elimination
  - probabilistic reasoning
  - pathophysiology-first reasoning
  - causal reasoning
  - evidence-first reasoning
  - mechanism-based reasoning
  - temporal reasoning
- You may invent other medically meaningful reasoning strategies when appropriate.
- Give the answering model concrete reasoning instructions rather than simply restating the goal.
- Preserve useful formatting or output instructions unless they conflict with the new reasoning strategy.
- Avoid introducing unrelated tasks or objectives.
- Avoid changing the requirement that exactly one answer must be selected.
- Avoid inventing or hallucinating medical knowledge.

Novelty Requirement
-------------------
The generated prompt should be substantially different from the parent.

A human reviewer should immediately recognize that the child prompt follows a different reasoning philosophy rather than being a polished rewrite.

Small wording changes, synonym replacement, sentence reordering, or stylistic improvements are NOT sufficient.

Strict Constraint
-----------------
Do NOT simply reword, paraphrase, merge, or lightly edit the parent prompt.

The reasoning strategy itself must change in a meaningful way.

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

Produce exactly one new prompt and nothing else.
"""