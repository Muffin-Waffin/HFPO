CROSSOVER_TEMPLATE = """You are designing new prompts for a medical multiple-choice question answering system.

Task
----
{task_description}

Parent Prompt A
---------------
{parent_a}

Parent Prompt B
---------------
{parent_b}

Your objective is to design FIVE new prompts inspired by both parents. You can be creative.

Reasoning Process (perform internally)
--------------------------------------
Before writing each new prompt:

1. Identify the strongest ideas or behaviors encouraged by Parent A.
2. Identify the strongest ideas or behaviors encouraged by Parent B.
3. Ignore the wording and sentence structure of both parents.
4. Design a completely new prompt that preserves the strongest ideas from both while expressing them in a fresh way.

Do NOT reveal this reasoning.

Design Principles
-----------------
Each offspring should:

- preserve the shared objective of answering medical multiple-choice questions;
- combine useful behaviors from both parents;
- read as though it were written independently;
- have its own organization and wording;
- improve clarity whenever possible;
- avoid unnecessary complexity.

Each offspring should NOT:

- copy sentences from either parent;
- concatenate the parents;
- lightly edit one parent;
- explain where ideas came from;
- describe the crossover process;
- mention Parent A or Parent B.

All five prompts should explore DIFFERENT ways of combining ideas from both parents.

Avoid producing five prompts that differ only by:
- synonym replacement;
- adjective changes;
- sentence reordering;
- adding or removing a short phrase.

Instead, vary aspects such as:
- instruction ordering;
- reasoning process;
- role or perspective;
- emphasis;
- decision strategy;
- level of explicitness.

Do NOT:
------
- explain your changes;
- compare candidates;
- rank candidates;
- recommend one candidate;
- mention prompt engineering, mutation, evolution, optimization, or genetic algorithms;
- generate a medical case or patient vignette;
- answer the medical question yourself;
- include code fences;
- surround prompts with quotation marks.

Output
------
Return exactly five candidates using this exact format:

=== Candidate 1 ===
<prompt>

=== Candidate 2 ===
<prompt>

=== Candidate 3 ===
<prompt>

=== Candidate 4 ===
<prompt>

=== Candidate 5 ===
<prompt>

Do not output anything before Candidate 1 or after Candidate 5.

The rewritten prompt must encourage concise reasoning.

Do not instruct the model to repeatedly verify, reconsider,
iterate indefinitely, or continue reasoning until certainty.

Avoid instructions that substantially increase inference length.
"""