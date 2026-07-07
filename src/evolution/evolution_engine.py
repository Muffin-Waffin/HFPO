"""Orchestration engine for HFPO evolutionary optimization runs.

This module defines :class:`EvolutionEngine`, the top-level
orchestration component responsible for coordinating one complete
HFPO (Heterogeneous Federated Prompt Optimization) run.

EvolutionEngine implements no algorithmic logic itself. It contains
no mutation, crossover, tournament selection, elitism, evaluation,
prompt generation, fitness computation, or file persistence code.
All specialized work is delegated to injected collaborator
components; EvolutionEngine's sole responsibility is to call those
collaborators in the correct order.
"""

from __future__ import annotations

import time
import random
from typing import Any


from src.evolution.similarity import too_similar
from src.evolution.prompt_logger import PromptLogger
from src.evolution.mutation_prompt_logger import MutationPromptLogger
from src.core.population import Population
from src.core.prompt_candidate import PromptCandidate
from src.core.mutation_prompt_candidate import MutationPromptCandidate
from src.core.mutation_prompt_manager import MutationPromptManager
from src.evolution.elitism import Elitism
from src.evolution.generation_snapshot import GenerationSnapshot
from src.evolution.output_manager import OutputManager
from src.evolution.prompt_generator.generator import PromptGenerator
from src.evolution.prompt_generator.models import PromptGenerationRequest, ParentPerformance
from src.evolution.tournament_selector import TournamentSelector
from src.federated.federated_server import FederatedServer
from src.evolution.prompt_generator.templates import PromptTemplateBuilder


class EvolutionEngine:
    """Orchestrates a complete HFPO evolutionary optimization run.

    EvolutionEngine coordinates a fixed set of injected collaborators
    -- a Population, a FederatedServer (the federated evaluator), a
    TournamentSelector, an Elitism strategy, a PromptGenerator, and an
    OutputManager -- to execute the generational loop described by
    the HFPO architecture. It does not itself perform mutation,
    crossover, selection, evaluation, prompt generation, fitness
    computation, or persistence; it only sequences calls to the
    components that do.
    """

    __slots__ = (
        "_population",
        "_federated_server",
        "_tournament_selector",
        "_elitism",
        "_prompt_generator",
        "_output_manager",
        "_num_generations",
        "_task_description",
        "_temperature",
        "_prompt_logger",
        "_mutation_prompt_logger",
        "_mutation_rate",
        "_crossover_rate",
        "_operator_rng",
        "_start_generation",
        "_mutation_manager",
        "_mutation_evolution_interval",
        "_min_children_for_evolution",
        "_offspring_mutation_info",
    )

    def __init__(
        self,
        population: Population,
        federated_server: FederatedServer,
        tournament_selector: TournamentSelector,
        elitism: Elitism,
        prompt_generator: PromptGenerator,
        output_manager: OutputManager,
        num_generations: int,
        task_description: str,
        temperature: float = 0.7,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        start_generation: int = 0,
        mutation_manager: MutationPromptManager | None = None,
        mutation_evolution_interval: int = 5,
        min_children_for_evolution: int = 10,
    ) -> None:
        """Initializes the EvolutionEngine with all required collaborators.

        Every collaborator is validated to be an instance of its
        expected type (which also rejects ``None``) before being
        stored. No algorithmic state is initialized here; that is the
        responsibility of ``_initialize()``.
        """
        if not isinstance(population, Population):
            raise TypeError("population must be a Population instance.")
        if not isinstance(federated_server, FederatedServer):
            raise TypeError("federated_server must be a FederatedServer instance.")
        if not isinstance(tournament_selector, TournamentSelector):
            raise TypeError("tournament_selector must be a TournamentSelector instance.")
        if not isinstance(elitism, Elitism):
            raise TypeError("elitism must be an Elitism instance.")
        if not isinstance(prompt_generator, PromptGenerator):
            raise TypeError("prompt_generator must be a PromptGenerator instance.")
        if not isinstance(output_manager, OutputManager):
            raise TypeError("output_manager must be an OutputManager instance.")
        if not isinstance(num_generations, int) or isinstance(num_generations, bool):
            raise TypeError("num_generations must be an integer.")
        if num_generations <= 0:
            raise ValueError("num_generations must be positive.")
        if not isinstance(task_description, str):
            raise TypeError("task_description must be a string.")
        if not task_description.strip():
            raise ValueError("task_description must not be empty.")
        if temperature < 0:
            raise ValueError(
                f"temperature must be >= 0, got {temperature}."
            )
        if mutation_rate < 0:
            raise ValueError(
                f"mutation_rate must be >= 0, got {mutation_rate}."
            )
        if crossover_rate < 0:
            raise ValueError(
                f"crossover_rate must be >= 0, got {crossover_rate}."
            )
        if mutation_rate + crossover_rate <= 0:
            raise ValueError(
                "mutation_rate + crossover_rate must be > 0."
            )
        if not isinstance(start_generation, int) or isinstance(start_generation, bool):
            raise TypeError("start_generation must be an integer.")
        if start_generation < 0:
            raise ValueError(
                f"start_generation must be >= 0, got {start_generation}."
            )
        if start_generation >= num_generations:
            raise ValueError(
                f"start_generation ({start_generation}) must be < "
                f"num_generations ({num_generations})."
            )
        if mutation_manager is not None and not isinstance(mutation_manager, MutationPromptManager):
            raise TypeError("mutation_manager must be a MutationPromptManager instance or None.")
        if not isinstance(mutation_evolution_interval, int) or mutation_evolution_interval <= 0:
            raise ValueError("mutation_evolution_interval must be a positive integer.")
        if not isinstance(min_children_for_evolution, int) or min_children_for_evolution <= 0:
            raise ValueError("min_children_for_evolution must be a positive integer.")

        self._population = population
        self._federated_server = federated_server
        self._tournament_selector = tournament_selector
        self._elitism = elitism
        self._prompt_generator = prompt_generator
        self._output_manager = output_manager
        self._num_generations = num_generations
        self._task_description = task_description
        self._temperature = temperature
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._start_generation = start_generation
        self._operator_rng = random.Random()
        self._prompt_logger = PromptLogger(
            "results/hfpo_run/generated_prompts.jsonl"
        )
        self._mutation_prompt_logger = MutationPromptLogger(
            "results/hfpo_run/mutation_history.jsonl"
        )
        self._mutation_manager = mutation_manager
        self._mutation_evolution_interval = mutation_evolution_interval
        self._min_children_for_evolution = min_children_for_evolution
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Population:
        print("\nStarting Evolution...\n")
        """Executes the complete HFPO evolutionary optimization run.

        This is the only public execution method on EvolutionEngine.
        For each of ``_num_generations`` generations, it:

            1. evaluates the population via the federated server
            2. computes generation statistics
            3. creates a GenerationSnapshot
            4. persists outputs via the OutputManager
            5. selects elites
            6. generates offspring
            7. builds the next population and replaces ``_population``

        After the final generation, returns the resulting population.

        Returns:
            The final Population after all ``_num_generations``
            generations have run.

        Raises:
            ValueError: If the engine's initial state is invalid (see
                ``_initialize()``), or propagated from any
                collaborator for invalid intermediate state.
            RuntimeError: Propagated from ``_federated_server`` or
                ``_generation_statistics()`` if evaluation fails or a
                candidate is left without a FitnessVector.
        """
        self._initialize()
        remaining = self._num_generations - self._start_generation
        if self._start_generation > 0:
            print(
                f"Resuming from generation {self._start_generation + 1}"
                f" ({remaining} generations remaining)..."
            )
        for generation in range(self._start_generation, self._num_generations):
            print(f"\n{'=' * 80}")
            print(f"Generation {generation + 1}/{self._num_generations}")
            print(f"{'=' * 80}")

            generation_start = time.perf_counter()

            print("Evaluating population...")
            evaluated_population = self._evaluate_population(generation)

            best = max(
                evaluated_population,
                key=lambda p: p.fitness.average(),
            )

            print(
                f"Evaluation complete | "
                f"Best fitness: {best.fitness.average():.4f}"
            )

            elites = self._select_elites(evaluated_population)
            print(f"Selected {len(elites)} elites.")

            print("Generating offspring...")
            offspring = self._generate_offspring(
                elites,
                generation,
            )
            print(f"Generated {len(offspring)} offspring.")

            # Record mutation prompt statistics after evaluation
            if self._mutation_manager is not None:
                self._record_mutation_results(offspring, generation)

            elapsed_time_seconds = (
                time.perf_counter() - generation_start
            )

            snapshot = self._create_snapshot(
                generation,
                evaluated_population,
                elapsed_time_seconds,
            )

            print("Saving generation snapshot...")
            self._save_outputs(snapshot)

            print("Building next population...")
            self._population = self._build_next_population(
                elites,
                offspring,
            )

            # Evolve mutation prompts periodically
            if (self._mutation_manager is not None and 
                (generation + 1) % self._mutation_evolution_interval == 0):
                self._evolve_mutation_prompts(generation)

            print("Saving population checkpoint...")
            self._output_manager.save_population_checkpoint(
                self._population, generation + 1
            )

            print(
                f"Generation {generation + 1} completed "
                f"in {elapsed_time_seconds:.2f} seconds."
            )
        return self._population
    # ------------------------------------------------------------------
    # Private orchestration helpers
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Validates the engine's state before the generational loop runs."""
        if len(self._population) == 0:
            raise ValueError(
                "EvolutionEngine cannot run with an empty population."
            )

        if self._num_generations <= 0:
            raise ValueError(
                "EvolutionEngine num_generations must be > 0, got "
                f"{self._num_generations}."
            )

    def _evaluate_population(self, generation: int) -> list[PromptCandidate]:
        """Evaluates the current population via the federated server."""
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise TypeError("generation must be an integer.")
        if generation < 0:
            raise ValueError("generation must be non-negative.")

        return self._federated_server.evaluate_population(list(self._population))

    def _record_mutation_results(
        self, 
        offspring: list[PromptCandidate], 
        generation: int
    ) -> None:
        """Record results for each offspring's mutation prompt."""
        if not hasattr(self, '_offspring_mutation_info'):
            return
            
        for child in offspring:
            child_id = child.id
            if child_id in self._offspring_mutation_info:
                info = self._offspring_mutation_info[child_id]
                mutation_prompt = info['mutation_prompt']
                parent = info['parent']
                
                if child.fitness is not None and parent.fitness is not None:
                    child_fitness = child.fitness.average()
                    parent_fitness = parent.fitness.average()
                    improvement = child_fitness - parent_fitness
                    success = improvement > 0
                    
                    mutation_prompt.record_child_result(
                        parent_id=parent.id,
                        child_id=child.id,
                        parent_fitness=parent_fitness,
                        child_fitness=child_fitness,
                        generation=generation,
                    )
                    
                    # Log to mutation history
                    self._mutation_prompt_logger.log(
                        generation=generation,
                        mutation_prompt=mutation_prompt,
                        parent=parent,
                        child=child,
                        parent_fitness=parent_fitness,
                        child_fitness=child_fitness,
                        improvement=improvement,
                        success=success,
                    )
        
        # Clear for next generation
        self._offspring_mutation_info = {}

    def _create_snapshot(
        self,
        generation: int,
        fitness_results: list[PromptCandidate],
        elapsed_time_seconds: float,
    ) -> GenerationSnapshot:
        """Builds a GenerationSnapshot summarizing one generation."""
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise TypeError("generation must be an integer.")
        if generation < 0:
            raise ValueError("generation must be non-negative.")
        if not isinstance(fitness_results, list):
            raise TypeError("fitness_results must be a list of PromptCandidate.")

        best_candidate, best_score, average_score, worst_score = (
            self._generation_statistics(fitness_results)
        )

        return GenerationSnapshot(
            generation=generation,
            population=fitness_results,
            best_candidate=best_candidate,
            best_score=best_score,
            average_score=average_score,
            worst_score=worst_score,
            population_size=len(fitness_results),
            elapsed_time_seconds=elapsed_time_seconds,
        )

    def _save_outputs(self, snapshot: GenerationSnapshot) -> None:
        """Persists a generation's outputs via the OutputManager."""
        if not isinstance(snapshot, GenerationSnapshot):
            raise TypeError(
                "snapshot must be a GenerationSnapshot."
            )

        self._output_manager.save_snapshot(snapshot)
        self._output_manager.save_best_prompt(snapshot.best_candidate)

    def _select_elites(self, fitness_results: Any) -> list[PromptCandidate]:
        """Selects elite candidates to carry forward unmodified."""
        if fitness_results is None:
            raise ValueError("fitness_results must not be None.")

        return self._elitism.select_elites(self._population)

    def _compute_performance_summary(self, candidate: PromptCandidate) -> ParentPerformance | None:
        """Compute a performance summary for a candidate from its FitnessVector."""
        if candidate.fitness is None:
            return None
        return ParentPerformance.from_fitness_vector(candidate.fitness)

    def _generate_offspring(
        self, elites: list[PromptCandidate], generation: int
    ) -> list[PromptCandidate]:
        """Generates offspring candidates via the PromptGenerator."""
        if elites is None:
            raise ValueError("elites must not be None.")
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise TypeError("generation must be an integer.")
        if generation < 0:
            raise ValueError("generation must be non-negative.")

        offspring: list[PromptCandidate] = []
        offspring_needed = self._population.max_population_size - len(elites)
        max_retries = 5
        print(f"Generating {offspring_needed} offspring...")
        existing_prompt_texts = set(self._population.texts())
        
        # Track which mutation prompt generated each offspring
        self._offspring_mutation_info = {}
        
        for _ in range(offspring_needed):
            generated = False
            for attempt in range(max_retries):
                parent_a = self._tournament_selector.select_parents(
                    self._population,
                    1,
                )[0]

                parent_a_performance = self._compute_performance_summary(parent_a)

                use_crossover = (
                    len(self._population) >= 2
                    and self._operator_rng.random()
                    < self._crossover_rate / (self._mutation_rate + self._crossover_rate)
                )

                if use_crossover:
                    parent_b = self._select_distinct_second_parent(parent_a)
                    parent_b_performance = self._compute_performance_summary(parent_b)
                    request = PromptGenerationRequest(
                        parent_a=parent_a,
                        parent_b=parent_b,
                        generation=self._population.generation + 1,
                        task_description=self._task_description,
                        temperature=self._temperature,
                        existing_prompt_texts=existing_prompt_texts,
                        parent_a_performance=parent_a_performance,
                        parent_b_performance=parent_b_performance,
                        mutation_operator=None,
                    )
                else:
                    request = PromptGenerationRequest(
                        parent_a=parent_a,
                        parent_b=None,
                        generation=self._population.generation + 1,
                        task_description=self._task_description,
                        temperature=self._temperature,
                        existing_prompt_texts=existing_prompt_texts,
                        parent_a_performance=parent_a_performance,
                        mutation_operator=None,
                    )

                try:
                    result = self._prompt_generator.generate(request)

                    if too_similar(
                        result.candidate.text,
                        existing_prompt_texts,
                        threshold=0.95,
                    ):
                        print("Too similar, regenerating...")
                        continue

                    offspring.append(result.candidate)
                    existing_prompt_texts.add(result.candidate.text)
                                        
                    # Track mutation prompt used (for mutation, not crossover)
                    if not use_crossover and self._mutation_manager is not None:
                        # Get the mutation prompt that was selected
                        template_builder = self._prompt_generator._template_builder
                        if hasattr(template_builder, '_last_mutation_operator'):
                            operator_id = template_builder._last_mutation_operator
                            if operator_id in self._mutation_manager._candidates:
                                mutation_prompt = self._mutation_manager._candidates[operator_id]
                                self._offspring_mutation_info[result.candidate.id] = {
                                    'mutation_prompt': mutation_prompt,
                                    'parent': parent_a,
                                }
                    
                    self._prompt_logger.log(
                        generation=request.generation,
                        child=result.candidate,
                        parent_a=parent_a,
                        parent_b=request.parent_b,
                    )
                    generated = True
                    break
                except ValueError as e:
                    print(
                        f"  Attempt {attempt + 1}/{max_retries} failed: {e}"
                    )
            if not generated:
                raise RuntimeError(
                    f"Failed to generate a valid offspring after "
                    f"{max_retries} attempts."
                )
            print(
                f"Generated offspring "
                f"{len(offspring)}/{offspring_needed}"
            )
        print("Offspring generation complete.")
        return offspring

    def _select_distinct_second_parent(
        self, parent_a: PromptCandidate
    ) -> PromptCandidate:
        """Selects a second crossover parent distinct from ``parent_a``."""
        max_attempts = len(self._population)

        for _ in range(max_attempts):
            candidate = self._tournament_selector.select_parents(
                self._population,
                1,
            )[0]
            if candidate.id != parent_a.id:
                return candidate

        raise RuntimeError(
            "Could not find a second crossover parent distinct from "
            f"'{parent_a.id}' after {max_attempts} attempts."
        )

    def _build_next_population(
        self,
        elites: list[PromptCandidate],
        offspring: list[PromptCandidate],
    ) -> Population:
        """Assembles the next generation's Population."""
        if elites is None:
            raise ValueError("elites must not be None.")
        if offspring is None:
            raise ValueError("offspring must not be None.")

        return Population(
            prompts=elites + offspring,
            generation=self._population.generation + 1,
            max_population_size=self._population.max_population_size,
        )

    def _evolve_mutation_prompts(self, generation: int) -> None:
        """Evolve mutation prompts using meta-mutation."""
        if self._mutation_manager is None:
            return

        print("Evolving mutation prompts...")
        llm = self._prompt_generator._llm

        self._mutation_manager.evolve(
            llm=llm,
            task_description=self._task_description,
        )
        print("Mutation prompt evolution complete.")

    def _generation_statistics(
        self, fitness_results: list[PromptCandidate]
    ) -> tuple[PromptCandidate, float, float, float]:
        """Computes reporting statistics for an evaluated generation."""
        if not isinstance(fitness_results, list):
            raise TypeError("fitness_results must be a list of PromptCandidate.")
        if not fitness_results:
            raise ValueError("fitness_results must not be empty.")

        average_scores: list[float] = []
        for candidate in fitness_results:
            if candidate.fitness is None:
                raise RuntimeError(
                    f"Candidate '{candidate.id}' has no FitnessVector; "
                    "cannot compute generation statistics."
                )
            average_scores.append(candidate.fitness.average())

        best_candidate = max(
            fitness_results, key=lambda candidate: candidate.fitness.average()
        )
        best_score = max(average_scores)
        average_score = sum(average_scores) / len(average_scores)
        worst_score = min(average_scores)

        return (best_candidate, best_score, average_score, worst_score)