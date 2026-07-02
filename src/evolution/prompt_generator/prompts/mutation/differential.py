"""Mutation prompt template for the Prompt Generator subsystem.

This module defines the mutation prompt template shown to the
reasoning LLM whenever the genetic algorithm wants to mutate a parent
prompt toward a differential-diagnosis strategy. It contains no
generation logic, no API calls, no PromptCandidate creation, no
validation, and no cleaning. It only defines a reusable, format()-style
prompt template.
"""

__all__ = ["MUTATION_TEMPLATE"]


# This template is presented to the reasoning LLM whenever HFPO performs
# a differential-diagnosis mutation. The model receives a single parent
# prompt and is instructed to produce exactly one variant that directs
# the answering model to consider multiple plausible diagnoses before
# selecting the final answer. The resulting text should be suitable for
# direct processing by PromptCleaner and PromptValidator.
MUTATION_TEMPLATE: str = """You are assisting in the optimization of prompts for a medical question-answering system through an evolutionary optimization process.

Task Description
----------------
{task_description}

Parent Prompt
-------------
{parent_prompt}

Your objective is to create exactly ONE new version of the parent prompt that changes the reasoning strategy to differential diagnosis.

The new prompt should:
- Preserve the original task of answering medical multiple-choice questions.
- Preserve the parent's overall objective while substantially changing HOW the answering model reasons.
- Instruct the answering model to first construct a differential diagnosis consisting of multiple plausible explanations supported by the clinical evidence.
- Instruct the answering model to compare the candidate diagnoses using the patient's symptoms, examination findings, laboratory data, imaging, and other relevant clinical evidence.
- Instruct the answering model to eliminate less likely diagnoses before selecting the final answer.
- Ensure the final answer is chosen because it is the strongest remaining diagnosis after comparison.
- Give concrete differential-diagnosis instructions rather than generic reasoning advice.
- Preserve useful formatting or output instructions unless they conflict with the new reasoning strategy.
- Avoid introducing unrelated objectives.
- Avoid changing the requirement that exactly one answer must be selected.
- Avoid inventing or hallucinating medical knowledge.

Operator Constraint
-------------------
This mutation MUST produce a differential-diagnosis prompt.

Do NOT replace the differential-diagnosis strategy with:
- systematic elimination as the primary strategy
- probabilistic reasoning
- evidence-first reasoning
- pathophysiology-first reasoning
- causal reasoning

Differential diagnosis must remain the dominant reasoning philosophy.

Novelty Requirement
-------------------
The generated prompt should be substantially different from the parent.

A human reviewer should immediately recognize that the child prompt follows a differential-diagnosis reasoning philosophy.

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