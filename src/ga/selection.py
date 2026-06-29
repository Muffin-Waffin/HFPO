"""
src/ga/selection.py
 
Tournament selection for the FedGAPrompt genetic algorithm.
 
Tournament selection randomly picks `tournament_size` individuals
from the population and returns the one with the highest fitness.
This preserves diversity better than pure elitism while still
applying selection pressure toward higher-accuracy prompts.
"""
 
import random
 
from src.ga.population import PromptIndividual
 
 
def tournament_selection(
    population: list[PromptIndividual],
    tournament_size: int = 3,
) -> PromptIndividual:
    """
    Select one individual via tournament selection.
 
    Randomly samples `tournament_size` individuals from the population
    and returns the one with the highest fitness score.
 
    Parameters
    ----------
    population      : Current list of evaluated PromptIndividuals.
    tournament_size : Number of individuals to compete. Higher values
                      increase selection pressure. Default is 3.
 
    Returns
    -------
    The winning PromptIndividual (highest fitness in the tournament).
 
    Raises
    ------
    ValueError : If population is empty or tournament_size exceeds it.
    """
    if not population:
        raise ValueError("Population must not be empty.")
 
    if tournament_size > len(population):
        raise ValueError(
            f"tournament_size ({tournament_size}) exceeds "
            f"population size ({len(population)})."
        )
 
    contestants = random.sample(population, tournament_size)
    return max(contestants, key=lambda ind: ind.fitness)