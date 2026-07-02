"""Mutation prompt template for the Prompt Generator subsystem.

This module defines the mutation prompt template shown to the
reasoning LLM whenever the genetic algorithm wants to mutate a parent
prompt toward an evidence-based reasoning strategy. It contains no
generation logic, no API calls, no PromptCandidate creation, no
validation, and no cleaning. It only defines a reusable, format()-style
prompt template.
"""

__all__ = ["MUTATION_TEMPLATE"]


# This template is presented to the reasoning LLM whenever HFPO performs
# an evidence-based mutation. The model receives a single parent prompt
# and is instructed to produce exactly one variant that directs the
# answering model to ground its reasoning in the evidence presented in
# the question while preserving the parent's underlying objective. The
# resulting text should be suitable for direct processing by
# PromptCleaner and PromptValidator.
MUTATION_TEMPLATE: str = """You are assisting in the optimization of prompts for a medical question-answering system through an evolutionary optimization process.

Task Description
----------------
{task_description}

Parent Prompt
-------------
{parent_prompt}

Your objective is to create exactly ONE new version of the parent prompt that changes the reasoning strategy to evidence-based reasoning.

The new prompt should:
- Preserve the original task of answering medical multiple-choice questions.
- Preserve the parent's overall objective while substantially changing HOW the answering model reasons.
- Instruct the answering model to first identify the clinically relevant evidence presented in the question.
- Instruct the answering model to distinguish relevant evidence from distracting or irrelevant details.
- Instruct the answering model to interpret the evidence using established medical knowledge rather than unsupported assumptions.
- Encourage every conclusion to be justified by the available clinical evidence.
- Preserve useful formatting or output instructions unless they conflict with the new reasoning strategy.
- Avoid introducing unrelated objectives.
- Avoid changing the requirement that exactly one answer must be selected.
- Avoid inventing or hallucinating medical knowledge.

Operator Constraint
-------------------
This mutation MUST produce an evidence-based reasoning prompt.

Do NOT replace the evidence-based strategy with:
- differential diagnosis
- systematic elimination
- probabilistic reasoning
- pathophysiology-first reasoning
- causal reasoning

Evidence-based reasoning must remain the dominant reasoning philosophy.

Novelty Requirement
-------------------
The generated prompt should be substantially different from the parent.

A human reviewer should immediately recognize that the child prompt follows an evidence-grounded reasoning philosophy.

Minor wording changes, synonym replacement, sentence reordering, or stylistic improvements are NOT sufficient.

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