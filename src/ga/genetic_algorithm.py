"""
src/ga/genetic_algorithm.py

Core Genetic Algorithm engine for FedGAPrompt.

Evolves a population of system prompts over N generations to maximize
score_prompt() accuracy on a medical QA dataset.

Designed to be extended with:
    - Flower federated learning (Milestone 6)
    - Differential privacy      (Milestone 7)
    - Membership inference      (Milestone 8)
"""

import random
from dataclasses import dataclass, field

from src.ga.population import PromptIndividual, initialize_population
from src.ga.selection import tournament_selection
from src.ga.crossover import one_point_crossover
from src.ga.mutation import mutate
from src.evaluation.scorer import score_prompt


# ── generation result ─────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """
    Snapshot of one generation's outcome.

    Attributes
    ----------
    generation   : Generation index (0-based).
    best_fitness : Highest fitness in the population this generation.
    best_prompt  : The prompt that achieved best_fitness.
    avg_fitness  : Mean fitness across the whole population.
    """

    generation: int
    best_fitness: float
    best_prompt: str
    avg_fitness: float = field(default=0.0)


# ── genetic algorithm ─────────────────────────────────────────────────────────

class GeneticAlgorithm:
    """
    Evolves system prompts to maximise medical QA accuracy.

    Parameters
    ----------
    model           : HuggingFace model (Qwen3-8B or BioMistral-7B).
    tokenizer       : Corresponding tokenizer.
    dataset         : List of standardised sample dicts.
    population_size : Number of individuals per generation.
    generations     : Number of generations to run.
    mutation_rate   : Probability [0, 1] that an offspring is mutated.
    crossover_rate  : Probability [0, 1] that two parents are crossed over
                      (otherwise offspring are direct copies of parents).
    tournament_size : Number of contestants in each tournament selection.
    """

    def __init__(
        self,
        model,
        tokenizer,
        dataset: list[dict],
        population_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

        self.population: list[PromptIndividual] = []
        self.history: list[GenerationResult] = []

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate_population(self) -> None:
        """
        Score every individual in the population whose fitness is 0.0.

        Skips already-evaluated individuals to avoid redundant inference.
        fitness is written in-place on each PromptIndividual.
        """
        # for individual in self.population:
        #     if individual.fitness == 0.0:
        #         individual.fitness = score_prompt(
        #             self.model,
        #             self.tokenizer,
        #             individual.prompt,
        #             self.dataset,
        #         )

        for individual in self.population:
            if individual.fitness is None:
                individual.fitness = score_prompt(
                    self.model,
                    self.tokenizer,
                    individual.prompt,
                    self.dataset,
                )

    # ── generation step ───────────────────────────────────────────────────────

    def next_generation(self) -> None:
        """
        Produce a new population from the current one.

        Steps
        -----
        1. Elitism: carry the best individual forward unchanged.
        2. Fill remaining slots via tournament selection + crossover.
        3. Apply mutation probabilistically to each offspring.
        4. Replace current population with the new one.
        """
        new_population: list[PromptIndividual] = []

        # 1. Elitism — always keep the best individual.
        best = self._best_individual()
        new_population.append(PromptIndividual(prompt=best.prompt, fitness=best.fitness))

        # 2. Fill remaining slots.
        while len(new_population) < self.population_size:
            # parent_a = tournament_selection(self.population, self.tournament_size)
            # parent_b = tournament_selection(self.population, self.tournament_size)
            parent_a = tournament_selection(
                self.population,
                self.tournament_size,
            )

            parent_b = tournament_selection(
                self.population,
                self.tournament_size,
            )

            while parent_a.id == parent_b.id:
                parent_b = tournament_selection(
                    self.population,
                    self.tournament_size,
                )
                
            if random.random() < self.crossover_rate:
                child_a, child_b = one_point_crossover(parent_a, parent_b)
            else:
                child_a = PromptIndividual(prompt=parent_a.prompt)
                child_b = PromptIndividual(prompt=parent_b.prompt)

            for child in (child_a, child_b):
                if random.random() < self.mutation_rate:
                    child.prompt = mutate(child.prompt)
                    child.fitness = None # must be re-evaluated after mutation

                if len(new_population) < self.population_size:
                    new_population.append(child)

        self.population = new_population

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        baseline_prompts: list[str],
    ) -> PromptIndividual:
        """
        Run the full GA loop and return the best prompt found.

        Parameters
        ----------
        baseline_prompts : Seed prompts from Milestone 4 experiments.

        Returns
        -------
        The PromptIndividual with the highest fitness seen across all
        generations.
        """
        # Initialise population from baseline prompts.
        self.population = initialize_population(baseline_prompts, self.population_size)

        overall_best: PromptIndividual = self.population[0]

        for gen in range(self.generations):

            # Evaluate any unevaluated individuals.
            self.evaluate_population()

            # Track the best this generation.
            gen_best = self._best_individual()
            avg_fitness = self._average_fitness()

            result = GenerationResult(
                generation=gen,
                best_fitness=gen_best.fitness,
                best_prompt=gen_best.prompt,
                avg_fitness=avg_fitness,
            )
            self.history.append(result)

            self._log_generation(result)

            # Update overall best.
            if gen_best.fitness > overall_best.fitness:
                overall_best = PromptIndividual(
                    prompt=gen_best.prompt,
                    fitness=gen_best.fitness,
                )

            # Evolve (skip on the last generation — no point generating
            # offspring that will never be evaluated).
            if gen < self.generations - 1:
                self.next_generation()

        return overall_best

    # ── helpers ───────────────────────────────────────────────────────────────

    def _best_individual(self) -> PromptIndividual:
        """Return the individual with the highest fitness."""
        return max(self.population, key=lambda ind: ind.fitness)

    def _average_fitness(self) -> float:
        """Return mean fitness across the population."""
        return sum(ind.fitness for ind in self.population) / len(self.population)

    @staticmethod
    def _log_generation(result: GenerationResult) -> None:
        """Print a one-line summary for the current generation."""
        print(
            f"Generation {result.generation} | "
            f"Best: {result.best_fitness:.4f} | "
            f"Avg: {result.avg_fitness:.4f} | "
            f"Prompt: {result.best_prompt!r}"
        )