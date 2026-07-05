MUTATION_TEMPLATE = """You are improving prompts for an evolutionary optimization system that searches for high-performing prompts for medical multiple-choice question answering.

Task Description
----------------
{task_description}

Current Prompt
--------------
{parent_prompt}

Thinking Direction
------------------
{thinking_direction}

Objective
---------
Produce ONE improved prompt by applying the thinking direction to the current prompt.

Requirements
------------
The improved prompt must:

- Preserve the original objective.
- Remain a reusable instruction rather than a concrete medical question.
- Keep the prompt concise.
- Improve the prompt instead of rewriting it into a different task.
- Introduce a meaningful change rather than simple paraphrasing.
- Preserve any useful characteristics of the current prompt.

Do NOT:
------
- explain your changes;
- compare the new prompt with the current prompt;
- mention prompt engineering, mutation, evolution, optimization, or genetic algorithms;
- generate a medical case or patient vignette;
- answer the medical question yourself;
- produce multiple prompts;
- include markdown, numbering, or code fences;
- surround the prompt with quotation marks.

Output
------
Return ONLY the improved prompt.
"""