"""Meta-mutation template for evolving mutation strategies."""

__all__ = ["META_MUTATION_TEMPLATE"]

META_MUTATION_TEMPLATE: str = """You are improving a mutation strategy used inside a prompt optimization system for medical question-answering.

Task Description
----------------
{task_description}

Current Mutation Strategy
-------------------------
{current_strategy}

Performance Summary
-------------------
{performance_summary}

Objective
---------
Generate exactly FIVE improved mutation strategies.

Each strategy should generate child prompts with larger average fitness improvements while preserving the characteristics responsible for successful children.

Each candidate strategy must:
- preserve successful behavior from the current strategy;
- eliminate ineffective behavior;
- remain concise;
- describe HOW to mutate, not WHAT medical answer to produce;
- introduce meaningful behavioral differences from the other candidates.

Avoid producing five strategies that differ only by:
- synonym replacement;
- adjective changes;
- sentence reordering;
- adding or removing a short phrase.

Do NOT:
------
- explain your changes;
- compare candidates;
- rank candidates;
- recommend one candidate;
- include code fences;
- surround strategies with quotation marks.

Output
------
Return exactly five candidates using this exact format:

=== Candidate 1 ===
<strategy>

=== Candidate 2 ===
<strategy>

=== Candidate 3 ===
<strategy>

=== Candidate 4 ===
<strategy>

=== Candidate 5 ===
<strategy>

Do not output anything before Candidate 1 or after Candidate 5.
"""