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

Each candidate should differ substantially from the others.

Avoid producing candidates that only differ by wording.

Candidates should explore different reasoning behaviour, instruction organization, decision strategy, level of explicitness, and response constraints.

Requirements
------------
Each candidate prompt must:

- Preserve the original medical multiple-choice question-answering objective.
- Remain a reusable system instruction rather than a concrete medical question.
- Naturally encourage the reasoning methodology described above.
- Introduce meaningful behavioral or structural differences from the other candidates.
- Remain concise and coherent.
- Preserve useful characteristics of the parent prompt where appropriate.

A reader should not be able to tell that two candidates came from the same generation request. If candidates share sentence structure, they are too similar.

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