"""Crossover prompt template for the Prompt Generator subsystem.

This module defines the crossover prompt template shown to the
reasoning LLM whenever the genetic algorithm performs a crossover
operation between two parent prompts. It contains no generation
logic, no API calls, no PromptCandidate creation, no validation, and
no cleaning. It only defines a reusable, format()-style prompt
template.
"""

__all__ = ["CROSSOVER_TEMPLATE"]


# CROSSOVER_TEMPLATE is presented to the reasoning LLM whenever HFPO
# performs a crossover operation. The model receives two high-quality
# parent prompts and is instructed to synthesize exactly one offspring
# prompt that combines their strongest characteristics. The offspring
# should preserve the underlying task while improving reasoning
# quality, clarity, robustness, and medical accuracy. The resulting
# prompt should be suitable for direct processing by PromptCleaner and
# PromptValidator.
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

Your task is to produce exactly ONE offspring prompt.

Create a new prompt that synthesizes the strongest ideas from both
parents into a single coherent instruction. Do not mechanically merge,
concatenate, or lightly edit the parent prompts.

When producing the offspring prompt, you must:
- Preserve the underlying task shared by both parents.
- Inherit the best reasoning strategies from both parents.
- Retain effective instructions from each parent whenever they improve the overall prompt.
- Combine complementary strengths from both parents.
- Improve reasoning quality and logical consistency.
- Improve medical accuracy and precision.
- Improve clarity and readability.
- Improve robustness across diverse medical questions.
- Be concise while remaining complete.
- Remove unnecessary redundancy.
- Resolve conflicts between the parent prompts when necessary.
- Avoid inventing medical knowledge or unsupported clinical guidance.
- Avoid simply concatenating the two prompts.
- Avoid copying one parent with only trivial edits.
- Produce a single, coherent, unified prompt.

Output Requirements
-------------------
Return ONLY the offspring prompt.

Do NOT:
- explain your reasoning;
- compare the parent prompts;
- mention which parent contributed which ideas;
- include Markdown;
- include code fences;
- number the output;
- surround the prompt with quotation marks;
- mention crossover, mutation, evolution, optimization, or genetic algorithms in the generated prompt.

Produce exactly one offspring prompt and nothing else.
"""