"""
src/ga/population.py

Defines the PromptIndividual dataclass and population initialization.

A population is simply a list of PromptIndividuals.
Fitness is set to 0.0 at creation and filled in by evaluate_population().
"""

import random
from dataclasses import dataclass, field
from uuid import uuid4




@dataclass
class PromptIndividual:
    """
    A single candidate system prompt in the GA population.

    Attributes
    ----------
    prompt  : The system prompt string being evolved.
    fitness : Accuracy score in [0, 1]. Set after evaluation.
    """

    prompt: str
    fitness: float | None = field(default=None)
    id: str = field(default_factory=lambda: str(uuid4())[:8])

    def __repr__(self) -> str:
        # return f"PromptIndividual(fitness={self.fitness:.3f}, prompt={self.prompt!r})"
        return (
        f"PromptIndividual("
        f"id={self.id}, "
        f"fitness={self.fitness:.3f}, "
        f"prompt={self.prompt!r})"
    )


def initialize_population(
    baseline_prompts: list[str],
    population_size: int,
) -> list[PromptIndividual]:
    """
    Build the initial population from a list of baseline prompts.

    Baseline prompts are used as-is first, then randomly duplicated
    until the requested population_size is reached.
    No LLM is called here.

    Parameters
    ----------
    baseline_prompts : Seed prompts from Milestone 4 experiments.
    population_size  : Total number of individuals to create.

    Returns
    -------
    List of PromptIndividual with fitness=0.0.
    """
    if not baseline_prompts:
        raise ValueError("baseline_prompts must not be empty.")

    if population_size < len(baseline_prompts):
        raise ValueError(
            f"population_size ({population_size}) must be >= "
            f"len(baseline_prompts) ({len(baseline_prompts)})."
        )

    # Start with one individual per baseline prompt.
    population = [PromptIndividual(prompt=p) for p in baseline_prompts]

    # Fill remaining slots by randomly sampling from baselines.
    while len(population) < population_size:
        prompt = random.choice(baseline_prompts)
        population.append(PromptIndividual(prompt=prompt))

    random.shuffle(population)

    return population