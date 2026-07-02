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

Stage 1 established the orchestration architecture: ``run()`` calls
every private helper in the correct order, but each helper raised
``NotImplementedError``. Stage 2 implemented ``_initialize()``,
``_evaluate_population()``, ``_generation_statistics()``,
``_create_snapshot()``, and ``_save_outputs()``, making the engine
capable of executing generation 0 end-to-end. Stage 3 implements
``_select_elites()``, ``_generate_offspring()``, and
``_build_next_population()``, and updates ``run()`` to perform the
complete evolutionary loop across ``_num_generations`` generations:
evaluate, snapshot, persist, select elites, generate offspring, and
replace the population, each generation.
"""

from __future__ import annotations

import time
from typing import Any

from src.evolution.prompt_logger import PromptLogger
from src.core.population import Population
from src.core.prompt_candidate import PromptCandidate
from src.evolution.elitism import Elitism
from src.evolution.generation_snapshot import GenerationSnapshot
from src.evolution.output_manager import OutputManager
from src.evolution.prompt_generator.generator import PromptGenerator
from src.evolution.prompt_generator.models import PromptGenerationRequest
from src.evolution.tournament_selector import TournamentSelector
from src.federated.federated_server import FederatedServer


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

    Stage 3 completes the evolutionary loop: ``_select_elites()``
    delegates to ``_elitism``, ``_generate_offspring()`` delegates to
    ``_tournament_selector`` and ``_prompt_generator``, and
    ``_build_next_population()`` assembles the next generation's
    ``Population``. ``run()`` executes this full cycle once per
    generation, for ``_num_generations`` generations.

    Attributes:
        _population: The current Population under optimization.
        _federated_server: The federated evaluator used to compute
            per-client fitness for candidates against hospital data.
        _tournament_selector: Selection strategy used to choose
            parents/elites from an evaluated population.
        _elitism: Elitism strategy used to preserve top-performing
            candidates across generations.
        _prompt_generator: LLM-backed component used to produce
            offspring prompts via mutation/crossover operators.
        _output_manager: Persistence utility used to write snapshots,
            best-prompt reports, summaries, and checkpoints.
        _num_generations: Total number of generations to run.
        _task_description: Description of the task every generated
            prompt must address.
        _temperature: Sampling temperature used for prompt generation.
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
    ) -> None:
        """Initializes the EvolutionEngine with all required collaborators.

        Every collaborator is validated to be an instance of its
        expected type (which also rejects ``None``) before being
        stored. No algorithmic state is initialized here; that is the
        responsibility of ``_initialize()``.

        Args:
            population: The initial Population to evolve.
            federated_server: The federated evaluator responsible for
                computing per-client fitness for candidates.
            tournament_selector: Selection strategy used during the
                evolutionary loop.
            elitism: Elitism strategy used to preserve top performers.
            prompt_generator: LLM-backed mutation/crossover operator.
            output_manager: Persistence utility for experiment
                outputs.
            num_generations: Total number of generations to run. Must
                be a positive integer.
            task_description: Description of the task every generated
                prompt must address, passed through to every
                ``PromptGenerationRequest``. Must not be empty.
            temperature: Sampling temperature passed through to every
                ``PromptGenerationRequest``. Must be non-negative.
                Defaults to 0.7.

        Raises:
            TypeError: If any collaborator is not an instance of its
                expected type, if ``num_generations`` is not an
                integer, or if ``task_description`` is not a string.
            ValueError: If ``num_generations`` is not positive, if
                ``task_description`` is empty or whitespace, or if
                ``temperature`` is negative.
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

        self._population = population
        self._federated_server = federated_server
        self._tournament_selector = tournament_selector
        self._elitism = elitism
        self._prompt_generator = prompt_generator
        self._output_manager = output_manager
        self._num_generations = num_generations
        self._task_description = task_description
        self._temperature = temperature
        self._prompt_logger = PromptLogger(
            "results/hfpo_run/generated_prompts.jsonl"
        )
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
        for generation in range(self._num_generations):
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

            print(
                f"Generation {generation + 1} completed "
                f"in {elapsed_time_seconds:.2f} seconds."
            )
        return self._population
    # ------------------------------------------------------------------
    # Private orchestration helpers
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Validates the engine's state before the generational loop runs.

        Confirms that the current population is non-empty and that
        the engine is configured to run at least one generation.
        ``Population`` itself refuses to be constructed empty, but it
        is a mutable container (``clear()``, ``remove()``), so this
        check defends against the population having been emptied out
        between construction and ``run()``. No collaborator is
        invoked here; this is purely a precondition check.

        Raises:
            ValueError: If the population is empty, or if
                ``_num_generations`` is not positive.
        """
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
        """Evaluates the current population via the federated server.

        Delegates entirely to
        ``_federated_server.evaluate_population()``: this method
        computes no fitness itself. ``FederatedServer`` handles cache
        lookups, hospital broadcast, aggregation into FitnessVectors,
        and lineage registration; this method simply invokes it and
        returns its result unchanged.

        Args:
            generation: The zero-indexed generation currently being
                evaluated. Must be a non-negative integer. Not used to
                select which population is evaluated -- the engine
                always evaluates ``self._population`` -- but validated
                here since it identifies which generation this
                evaluation round belongs to for the caller.

        Returns:
            The list of ``PromptCandidate`` objects from the current
            population, each with its ``fitness`` attribute populated
            by ``_federated_server``.

        Raises:
            TypeError: If ``generation`` is not an integer.
            ValueError: If ``generation`` is negative, or propagated
                from ``_federated_server`` if the population is empty.
            RuntimeError: Propagated from ``_federated_server`` if
                federated evaluation fails (e.g. a mismatched response
                count or a candidate left without a FitnessVector).
        """
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise TypeError("generation must be an integer.")
        if generation < 0:
            raise ValueError("generation must be non-negative.")

        return self._federated_server.evaluate_population(list(self._population))

    def _create_snapshot(
        self,
        generation: int,
        fitness_results: list[PromptCandidate],
        elapsed_time_seconds: float,
    ) -> GenerationSnapshot:
        """Builds a GenerationSnapshot summarizing one generation.

        Computes this generation's reporting statistics via
        ``_generation_statistics()`` and combines them with the
        evaluated population, the generation index, and the measured
        wall-clock time into a single ``GenerationSnapshot``. This
        method performs no domain validation of its own (e.g. score
        ranges or population membership); ``GenerationSnapshot``'s own
        ``__post_init__`` already enforces those invariants, and any
        violation propagates from there rather than being duplicated
        here.

        Args:
            generation: The zero-indexed generation this snapshot
                describes.
            fitness_results: The evaluated population for this
                generation, as returned by ``_evaluate_population()``.
            elapsed_time_seconds: The wall-clock time taken to
                evaluate this generation, in seconds.

        Returns:
            A new ``GenerationSnapshot`` describing this generation.

        Raises:
            TypeError: If ``generation`` is not an integer, or if
                ``fitness_results`` is not a list.
            ValueError: If ``generation`` is negative, or propagated
                from ``_generation_statistics()`` /
                ``GenerationSnapshot`` if any field violates its
                invariants (e.g. an out-of-range score, an empty
                population, or a generation/population mismatch).
            RuntimeError: Propagated from ``_generation_statistics()``
                if any candidate has no ``FitnessVector``.
        """
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
        """Persists a generation's outputs via the OutputManager.

        Delegates entirely to ``_output_manager``: writes the
        generation snapshot via ``save_snapshot()`` and the best
        candidate's prompt report via ``save_best_prompt()``. This
        method performs no file I/O itself.

        Args:
            snapshot: The GenerationSnapshot to persist.

        Raises:
            TypeError: If ``snapshot`` is not a ``GenerationSnapshot``,
                or propagated from ``_output_manager`` if
                ``snapshot.best_candidate`` is not a valid
                PromptCandidate.
            ValueError: Propagated from ``_output_manager`` if
                ``snapshot.best_candidate.generation`` is negative.
        """
        if not isinstance(snapshot, GenerationSnapshot):
            raise TypeError(
                "snapshot must be a GenerationSnapshot."
            )

        self._output_manager.save_snapshot(snapshot)
        self._output_manager.save_best_prompt(snapshot.best_candidate)

    def _select_elites(self, fitness_results: Any) -> list[PromptCandidate]:
        """Selects elite candidates to carry forward unmodified.

        Delegates entirely to
        ``self._elitism.select_elites(self._population)``. Since
        federated evaluation mutates each ``PromptCandidate`` in place
        (setting its ``fitness`` attribute), the candidates referenced
        by ``self._population`` are already evaluated by the time this
        method runs; ``fitness_results`` is accepted only to keep
        ``run()``'s call signature symmetric with the other
        per-generation steps and is not otherwise used. This method
        computes no scores and performs no ranking itself; that logic
        belongs entirely to ``Elitism``.

        Args:
            fitness_results: The evaluated population produced by
                ``_evaluate_population()`` for the current generation.
                Must not be ``None``.

        Returns:
            The elite ``PromptCandidate`` objects selected by
            ``_elitism``, ordered from highest to lowest fitness.

        Raises:
            ValueError: If ``fitness_results`` is ``None``, or
                propagated from ``_elitism`` if the population is
                empty or smaller than its configured elite count.
            RuntimeError: Propagated from ``_elitism`` if any
                candidate has no ``FitnessVector``.
        """
        if fitness_results is None:
            raise ValueError("fitness_results must not be None.")

        return self._elitism.select_elites(self._population)

    def _generate_offspring(
        self, elites: list[PromptCandidate], generation: int
    ) -> list[PromptCandidate]:
        
        """Generates offspring candidates via the PromptGenerator.

        Produces enough offspring that ``len(elites) + len(offspring)``
        equals ``self._population.max_population_size``. For each
        offspring, obtains parents from ``_tournament_selector`` (two
        distinct parents when the population has at least two
        members, one parent otherwise), builds a
        ``PromptGenerationRequest``, and delegates the actual
        mutation/crossover work to ``_prompt_generator.generate()``.
        This method performs no mutation or crossover itself; it only
        constructs requests and delegates.

        ``TournamentSelector.select_parents()`` runs independent
        tournaments with replacement, so two calls can legitimately
        return the same winning candidate. ``PromptGenerationRequest``
        forbids ``parent_b`` sharing an ID with ``parent_a``, so for
        crossover offspring this method re-selects ``parent_b`` until
        it differs from ``parent_a``, bounded by
        ``len(self._population)`` attempts. Neither
        ``TournamentSelector`` nor ``PromptGenerationRequest`` is
        modified to accommodate this; the responsibility for
        guaranteeing distinct crossover parents belongs to
        ``EvolutionEngine``.

        Args:
            elites: The elite candidates selected by
                ``_select_elites()``. Must not be ``None``.
            generation: The zero-indexed generation offspring are
                being produced for. Must be a non-negative integer.

        Returns:
            A list of newly generated ``PromptCandidate`` objects,
            with length equal to
            ``self._population.max_population_size - len(elites)``.

        Raises:
            TypeError: If ``generation`` is not an integer.
            ValueError: If ``elites`` is ``None`` or ``generation`` is
                negative, or propagated from
                ``PromptGenerationRequest`` / ``_tournament_selector``
                / ``_prompt_generator`` for invalid intermediate
                state.
            RuntimeError: If no ``parent_b`` distinct from
                ``parent_a`` can be found within
                ``len(self._population)`` attempts, or propagated from
                ``_tournament_selector`` if any sampled candidate has
                no ``FitnessVector``.
        """
        if elites is None:
            raise ValueError("elites must not be None.")
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise TypeError("generation must be an integer.")
        if generation < 0:
            raise ValueError("generation must be non-negative.")

        offspring: list[PromptCandidate] = []
        offspring_needed = self._population.max_population_size - len(elites)
        print(f"Generating {offspring_needed} offspring...")
        existing_prompt_texts=set(self._population.texts())
        for _ in range(offspring_needed):
            if len(self._population) >= 2:
                parent_a = self._tournament_selector.select_parents(
                    self._population,
                    1,
                )[0]
                parent_b = self._select_distinct_second_parent(parent_a)
                request = PromptGenerationRequest(
                    parent_a=parent_a,
                    parent_b=parent_b,
                    generation=self._population.generation + 1,
                    task_description=self._task_description,
                    temperature=self._temperature,
                    existing_prompt_texts=existing_prompt_texts)
            else:
                parent = self._tournament_selector.select_parents(
                    self._population,
                    1,
                )[0]
                request = PromptGenerationRequest(
                    parent_a=parent,
                    parent_b=None,
                    generation=self._population.generation + 1,
                    task_description=self._task_description,
                    temperature=self._temperature,
                    existing_prompt_texts=existing_prompt_texts,
                )

            result = self._prompt_generator.generate(request)
            offspring.append(result.candidate)
            self._prompt_logger.log(
                generation=request.generation,
                child=result.candidate,
                parent_a=parent_a,
                parent_b=request.parent_b,
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
        """Selects a second crossover parent distinct from ``parent_a``.

        Repeatedly runs ``_tournament_selector.select_parents()`` for
        a single winner until one is found whose ID differs from
        ``parent_a.id``, bounded by ``len(self._population)`` attempts.
        This exists because ``TournamentSelector`` samples with
        replacement across independent tournaments and may legitimately
        return the same candidate twice, while
        ``PromptGenerationRequest`` forbids a crossover request whose
        two parents share an ID.

        Args:
            parent_a: The first crossover parent, already selected.

        Returns:
            A ``PromptCandidate`` whose ID differs from
            ``parent_a.id``.

        Raises:
            RuntimeError: If no distinct candidate is found within
                ``len(self._population)`` attempts.
        """
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
        """Assembles the next generation's Population.

        Combines ``elites`` and ``offspring`` into a brand-new
        ``Population`` instance with an incremented generation index.
        The existing ``self._population`` is not modified; a new
        instance is constructed and returned for the caller to assign.

        Args:
            elites: The elite candidates preserved from the current
                generation. Must not be ``None``.
            offspring: The newly generated candidates produced by
                ``_generate_offspring()``. Must not be ``None``.

        Returns:
            A new ``Population`` containing ``elites + offspring``, at
            generation ``self._population.generation + 1``, with the
            same ``max_population_size`` as the current population.

        Raises:
            ValueError: If ``elites`` or ``offspring`` is ``None``, or
                propagated from ``Population`` if the combined list is
                empty, exceeds ``max_population_size``, or contains
                duplicate candidate IDs.
        """
        if elites is None:
            raise ValueError("elites must not be None.")
        if offspring is None:
            raise ValueError("offspring must not be None.")

        return Population(
            prompts=elites + offspring,
            generation=self._population.generation + 1,
            max_population_size=self._population.max_population_size,
        )

    def _generation_statistics(
        self, fitness_results: list[PromptCandidate]
    ) -> tuple[PromptCandidate, float, float, float]:
        """Computes reporting statistics for an evaluated generation.

        Ranks candidates by ``FitnessVector.average()`` for reporting
        purposes only. This is intentionally independent of the
        ranking strategy configured on ``_tournament_selector`` or
        ``_elitism`` ("average", "minimum", or "weighted"); those
        strategies govern *selection*, while this method governs what
        gets *reported* in the generation snapshot and is not
        configurable at this stage. Performs no file writing and does
        not mutate ``fitness_results``.

        Args:
            fitness_results: The list of evaluated ``PromptCandidate``
                objects returned by ``_evaluate_population()``. Each
                must have a non-None ``fitness``.

        Returns:
            A 4-tuple of ``(best_candidate, best_score, average_score,
            worst_score)``, where ``best_candidate`` is the candidate
            with the highest ``FitnessVector.average()``, and the
            three scores are, respectively, the maximum, mean, and
            minimum of every candidate's ``FitnessVector.average()``.

        Raises:
            TypeError: If ``fitness_results`` is not a list.
            ValueError: If ``fitness_results`` is empty.
            RuntimeError: If any candidate has no ``FitnessVector``.
        """
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