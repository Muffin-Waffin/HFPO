"""Crossover prompt template for the Prompt Generator subsystem.

This module defines the crossover prompt template shown to the
reasoning LLM whenever the genetic algorithm performs a crossover
operation between two parent prompts. It contains no generation
logic, no API calls, no PromptCandidate creation, no validation, and
no cleaning. It only defines a reusable, format()-style prompt
template.
"""

__all__ = ["CROSSOVER_TEMPLATE"]


CROSSOVER_TEMPLATE: str = """You are assisting in the optimization of prompts for a medical question-answering system through an evolutionary optimization process.

Task Description
----------------
{task_description}

Parent Prompt A
---------------
{parent_a}

Parent Prompt B
---------------
{parent_b}

Your objective is to create exactly ONE offspring prompt.

The offspring should preserve the shared objective of both parents while
combining complementary characteristics from each.

Possible characteristics include:
- professional role or persona
- reasoning strategy
- prompt organization
- decision process
- constraints
- output instructions
- formatting

You are encouraged to recombine these characteristics in a novel way.

Examples include:
- use the reasoning strategy from Parent A and the professional role from Parent B;
- use the prompt organization from Parent B and the decision process from Parent A;
- combine complementary constraints from both parents.

Novelty Requirement
-------------------
The offspring should be substantially different from BOTH parents.

A human reviewer should immediately recognize that the offspring is a
new prompt rather than an edited copy or average of either parent.

Behavioral Requirement
----------------------
The offspring should encourage a behavior that neither parent expresses
in exactly the same way.

Small wording changes, synonym replacement, or sentence reordering are
NOT sufficient.

Strict Constraints
------------------
Do NOT:
- concatenate the two prompts;
- average the two prompts;
- copy one parent with minor edits;
- simply paraphrase either parent;
- invent unrelated objectives;
- invent unsupported medical knowledge.

Preserve only the shared task:
answering medical multiple-choice questions accurately.

Output Requirements
-------------------
Return ONLY the offspring prompt.

Do NOT:
- explain your choices;
- compare the parents;
- identify which parent contributed what;
- include Markdown;
- include code fences;
- number the output;
- surround the prompt with quotation marks;
- mention crossover, mutation, evolution, optimization, prompt engineering, or genetic algorithms.

Produce exactly one offspring prompt and nothing else.
"""