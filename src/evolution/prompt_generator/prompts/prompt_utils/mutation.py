MUTATION_TEMPLATE: str = """You are generating diverse prompt variants for a medical multiple-choice question-answering system.

Task Description
----------------
{task_description}

Current Prompt
--------------
{parent_prompt}

Reasoning Method
----------------
{method_name}

Description
-----------
{method_description}

Objective
---------
Generate exactly FIVE evolved prompts.

Each prompt should preserve the original task while naturally encouraging the reasoning methodology described above.

All five prompts should explore DIFFERENT ways of incorporating this reasoning methodology.

Your job is ONLY to generate candidate prompts.
Do NOT choose, rank, compare, or recommend among them.

Requirements
------------
Each candidate prompt must:

- Preserve the original medical multiple-choice question-answering objective.
- Remain a reusable system instruction rather than a concrete medical question.
- Naturally encourage the reasoning methodology described above.
- Introduce meaningful behavioral or structural differences from the other candidates.
- Remain concise and coherent.
- Preserve useful characteristics of the parent prompt where appropriate.

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
"""