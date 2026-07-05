"""Meta-mutation template for evolving mutation strategies."""

__all__ = ["META_MUTATION_TEMPLATE"]

META_MUTATION_TEMPLATE: str = """You are improving a mutation strategy used inside an evolutionary prompt optimization system for medical question-answering.

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
Improve this mutation strategy so that it generates child prompts with larger average fitness improvements while preserving the characteristics responsible for its successful children.

The new mutation strategy should:
- preserve successful behavior
- eliminate ineffective behavior
- remain concise
- describe HOW to mutate, not WHAT medical answer to produce

Return ONLY the new mutation strategy. Do not include explanations, comparisons, or meta-commentary."""