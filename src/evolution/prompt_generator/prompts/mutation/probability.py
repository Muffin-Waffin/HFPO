"""Mutation prompt template for the Prompt Generator subsystem.

This module defines the mutation prompt template shown to the
reasoning LLM whenever the genetic algorithm wants to mutate a parent
prompt toward a probabilistic reasoning strategy. It contains no
generation logic, no API calls, no PromptCandidate creation, no
validation, and no cleaning. It only defines a reusable, format()-style
prompt template.
"""

__all__ = ["MUTATION_TEMPLATE"]


# This template is presented to the reasoning LLM whenever HFPO performs
# a probabilistic-reasoning mutation. The model receives a single parent
# prompt and is instructed to produce exactly one variant that directs
# the answering model to estimate and compare the likelihood of candidate
# answers before selecting the final answer. The resulting text should be
# suitable for direct processing by PromptCleaner and PromptValidator.
MUTATION_TEMPLATE: str = """You are assisting in the optimization of prompts for a medical question-answering system through an evolutionary optimization process.

Task Description
----------------
{task_description}

Parent Prompt
-------------
{parent_prompt}

Your objective is to create exactly ONE new version of the parent prompt that changes the reasoning strategy to probabilistic reasoning.

The new prompt should:
- Preserve the original task of answering medical multiple-choice questions.
- Preserve the parent's overall objective while substantially changing HOW the answering model reasons.
- Instruct the answering model to estimate the relative likelihood of every candidate answer.
- Encourage the answering model to compare the probability of competing options before making a decision.
- Instruct the answering model to rank the candidate answers from most likely to least likely based on the available clinical evidence.
- Encourage selecting the final answer only after comparing all plausible alternatives.
- Preserve useful formatting or output instructions unless they conflict with the new reasoning strategy.
- Avoid introducing unrelated objectives.
- Avoid changing the requirement that exactly one answer must be selected.
- Avoid inventing or hallucinating medical knowledge.

Operator Constraint
-------------------
This mutation MUST produce a probabilistic reasoning prompt.

Do NOT replace the probabilistic reasoning strategy with:
- differential diagnosis
- systematic elimination
- evidence-first reasoning
- pathophysiology-first reasoning
- causal reasoning

Probabilistic reasoning must remain the dominant reasoning philosophy.

Novelty Requirement
-------------------
The generated prompt should be substantially different from the parent.

A human reviewer should immediately recognize that the child prompt follows a probabilistic reasoning philosophy.

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