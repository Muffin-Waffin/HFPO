"""Crossover prompt template for the Prompt Generator subsystem.

This module defines the crossover prompt template used whenever HFPO
creates one offspring from two parent prompts.

The template intentionally minimizes reasoning instructions and instead
focuses the model on producing a single high-quality instruction prompt.
"""

__all__ = ["CROSSOVER_TEMPLATE"]


CROSSOVER_TEMPLATE: str = """You are improving prompts for a medical multiple-choice question answering system.

Task
----
{task_description}

Parent Prompt A
---------------
{parent_a}

Parent Prompt B
---------------
{parent_b}

Objective
---------
Both parent prompts successfully perform the same underlying task, but
each may contain useful strengths.

Create ONE new prompt that naturally preserves the strongest qualities
of both parents while remaining coherent and internally consistent.

The offspring should read as though it were written by a single author.
It should NOT look like two prompts stitched together.

Guidelines
----------
- Preserve the shared objective of answering medical multiple-choice questions.
- Keep useful instructions from both parents whenever they improve clarity or reasoning.
- Remove redundant, conflicting, or unnecessary instructions.
- Improve clarity, precision, and usefulness whenever possible.
- Prefer a clean, concise prompt over a longer one.
- If one parent contains a clearly better way of expressing an idea, use it.
- It is acceptable to discard weak parts of either parent.

Output Requirements
-------------------
Return ONLY the final prompt.

Do NOT:
- explain your reasoning;
- mention Parent A or Parent B;
- describe which ideas came from which parent;
- compare the parents;
- describe the crossover process;
- include Markdown;
- include code fences;
- include quotation marks;
- include numbered lists;
- mention crossover, mutation, evolution, optimization, prompt engineering, or genetic algorithms.

The output must be a reusable system prompt and nothing else.
"""