"""Mutation prompt template for the Prompt Generator subsystem.

This module defines the mutation prompt template shown to the
reasoning LLM whenever the genetic algorithm wants to mutate a parent
prompt by changing only the role or persona assigned to the answering
model. It contains no generation logic, no API calls, no
PromptCandidate creation, no validation, and no cleaning. It only
defines a reusable, format()-style prompt template.
"""

__all__ = ["MUTATION_TEMPLATE"]


# This template is presented to the reasoning LLM whenever HFPO performs
# a role-mutation operation. The model receives a single parent prompt
# and is instructed to produce exactly one variant that changes only the
# assigned professional role while preserving the parent's objective.
# The resulting text should be suitable for direct processing by
# PromptCleaner and PromptValidator.
MUTATION_TEMPLATE: str = """You are assisting in the optimization of prompts for a medical question-answering system through an evolutionary optimization process.

Task Description
----------------
{task_description}

Parent Prompt
-------------
{parent_prompt}

Your objective is to create exactly ONE new version of the parent prompt that changes the professional role or persona assigned to the answering model.

The new prompt should:
- Preserve the original task of answering medical multiple-choice questions.
- Preserve the parent's overall objective.
- Assign the answering model a substantially different professional role.
- Choose a medically meaningful role such as:
  - emergency physician
  - intensivist
  - infectious disease specialist
  - neurologist
  - cardiologist
  - oncologist
  - pediatrician
  - trauma surgeon
  - clinical pharmacologist
  - radiologist
  - pathologist
  - medical board examiner
  - another medically appropriate specialist.
- Keep the task, reasoning strategy, and expected output format functionally equivalent to the parent.
- Preserve useful formatting or output instructions.
- Avoid introducing unrelated objectives.
- Avoid changing the requirement that exactly one answer must be selected.
- Avoid inventing or hallucinating medical knowledge.

Operator Constraint
-------------------
This mutation MUST change only the assigned professional role.

Do NOT:
- introduce a new reasoning strategy;
- introduce systematic elimination;
- introduce differential diagnosis;
- introduce probabilistic reasoning;
- introduce evidence-first reasoning;
- significantly reorganize the prompt.

The primary mutation should be the assigned role.

Novelty Requirement
-------------------
The generated prompt should be substantially different from the parent with respect to the assigned professional identity.

A human reviewer should immediately recognize that the answering model has been placed into a different clinical role.

Changing only adjectives such as "experienced", "expert", or "senior" is NOT sufficient.

Choose a genuinely different medical specialty or professional perspective.

Strict Constraint
-----------------
Do NOT simply reword, paraphrase, merge, or lightly edit the parent prompt.

The role itself must change in a meaningful way.

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