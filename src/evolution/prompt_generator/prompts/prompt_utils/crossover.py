CROSSOVER_TEMPLATE = """You are designing a new prompt for a medical multiple-choice question answering system.

Task
----
{task_description}

Parent Prompt A
---------------
{parent_a}

Parent Prompt B
---------------
{parent_b}

Your objective is to design ONE new prompt inspired by both parents. You can be creative.

Reasoning Process (perform internally)
--------------------------------------
Before writing the new prompt:

1. Identify the strongest ideas or behaviors encouraged by Parent A.
2. Identify the strongest ideas or behaviors encouraged by Parent B.
3. Ignore the wording and sentence structure of both parents.
4. Design a completely new prompt that preserves the strongest ideas from both while expressing them in a fresh way.

Do NOT reveal this reasoning.

Design Principles
-----------------
The offspring should:

- preserve the shared objective of answering medical multiple-choice questions;
- combine useful behaviors from both parents;
- read as though it were written independently;
- have its own organization and wording;
- improve clarity whenever possible;
- avoid unnecessary complexity.

The offspring should NOT:

- copy sentences from either parent;
- concatenate the parents;
- lightly edit one parent;
- explain where ideas came from;
- describe the crossover process;
- mention Parent A or Parent B.

Imagine that someone reading the offspring has never seen either parent.

They should recognize it as a well-written prompt, not as a combination of two prompts.

Output Requirements
-------------------
Return ONLY the new prompt.

Do not include:

- introductions;
- explanations;
- reasoning;
- markdown;
- quotation marks;
- phrases like "Here is the new prompt";
- phrases like "The combined prompt";
- any text before or after the prompt itself.
"""