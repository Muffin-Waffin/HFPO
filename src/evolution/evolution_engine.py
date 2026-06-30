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

This is Stage 1 of the EvolutionEngine implementation: it establishes
the orchestration architecture only. Every private helper method
raises ``NotImplementedError`` and will be filled in during later
implementation stages.
"""

from __future__ import annotations

import time
from typing import Any

from src.core.population import Population
from src.core.prompt_candidate import PromptCandidate
from src.evolution.elitism import Elitism
from src.evolution.generation_snapshot import GenerationSnapshot
from src.evolution.output_manager import OutputManager
from src.evolution.prompt_generator.generator import PromptGenerator
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

    Stage 1 established this orchestration structure. Stage 2 fills in
    ``_initialize()``, ``_evaluate_population()``,
    ``_generation_statistics()``, ``_create_snapshot()``, and
    ``_save_outputs()``, making the engine capable of executing
    generation 0 end-to-end: initializing, evaluating the population
    via the federated server, computing reporting statistics,
    building a snapshot, and persisting it. ``_select_elites()``,
    ``_generate_offspring()``, and ``_build_next_population()`` still
    raise ``NotImplementedError``; no evolutionary loop runs yet, and
    ``run()`` executes exactly one generation before returning.

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
            used to construct per-generation snapshots.
    """

    __slots__ = (
        "_population",
        "_federated_server",
        "_tournament_selector",
        "_elitism",
        "_prompt_generator",
        "_output_manager",
        "_num_generations",
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

        Raises:
            TypeError: If any collaborator is not an instance of its
                if ``num_generations`` is not an integer.
            ValueError: If ``num_generations`` is not positive.
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

        self._population = population
        self._federated_server = federated_server
        self._tournament_selector = tournament_selector
        self._elitism = elitism
        self._prompt_generator = prompt_generator
        self._output_manager = output_manager
        self._num_generations = num_generations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Population:
        """Executes generation 0 of the HFPO evolutionary optimization run.

        This is the only public execution method on EvolutionEngine.
        Stage 2 wires together the portion of the orchestration flow
        needed to fully process one generation:

            initialize
            -> evaluate population (via FederatedServer)
            -> compute generation statistics
            -> create a GenerationSnapshot
            -> persist outputs (via OutputManager)
            -> return population

        No evolutionary operations run yet: elitism, tournament
        selection, offspring generation, and population replacement
        are not invoked in Stage 2, so ``run()`` always completes
        after generation 0 rather than looping for
        ``_num_generations`` generations.

        Returns:
            The Population that was evaluated, unchanged. (Population
            replacement is not implemented until a later stage.)

        Raises:
            ValueError: If the engine's initial state is invalid (see
                ``_initialize()``).
            RuntimeError: Propagated from ``_federated_server`` if
                evaluation fails (e.g. a candidate is left without a
                FitnessVector).
        """
        self._initialize()

        start_time = time.perf_counter()
        evaluated_population = self._evaluate_population()
        elapsed_time_seconds = time.perf_counter() - start_time

        generation_statistics = self._generation_statistics(evaluated_population)
        snapshot = self._create_snapshot(
            generation=0,
            population=evaluated_population,
            statistics=generation_statistics,
            elapsed_time_seconds=elapsed_time_seconds,
        )
        self._save_outputs(snapshot)

        return self._population

    # ------------------------------------------------------------------
    # Private orchestration helpers
    # (Stage 2: _initialize, _evaluate_population, _generation_statistics,
    # _create_snapshot, and _save_outputs are implemented below.
    # _select_elites, _generate_offspring, and _build_next_population
    # remain Stage 1 stubs that raise NotImplementedError.)
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Validates the engine's state before generation 0 runs.

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

    def _evaluate_population(self) -> list[PromptCandidate]:
        """Evaluates the current population via the federated server.

        Delegates entirely to
        ``_federated_server.evaluate_population()``: this method
        computes no fitness itself. ``FederatedServer`` handles cache
        lookups, hospital broadcast, aggregation into FitnessVectors,
        and lineage registration; this method simply invokes it and
        returns its result unchanged.

        Returns:
            The list of ``PromptCandidate`` objects from the current
            population, each with its ``fitness`` attribute populated
            by ``_federated_server``.

        Raises:
            ValueError: Propagated from ``_federated_server`` if the
                population is empty.
            RuntimeError: Propagated from ``_federated_server`` if
                federated evaluation fails (e.g. a mismatched response
                count or a candidate left without a FitnessVector).
        """
        return self._federated_server.evaluate_population(list(self._population))

    def _create_snapshot(
        self,
        generation: int,
        population: list[PromptCandidate],
        statistics: tuple[PromptCandidate, float, float, float],
        elapsed_time_seconds: float,
    ) -> GenerationSnapshot:
        """Builds a GenerationSnapshot summarizing one generation.

        Combines the evaluated population, the generation index, the
        aggregate statistics produced by ``_generation_statistics()``,
        and the measured wall-clock time into a single

        Args:
            generation: The zero-indexed generation this snapshot
                describes.
            population: The evaluated population for this generation,
                as returned by ``_evaluate_population()``.
            statistics: The ``(best_candidate, best_score,
                average_score, worst_score)`` tuple produced by
                ``_generation_statistics()``.
            elapsed_time_seconds: The wall-clock time taken to
                evaluate this generation, in seconds.


        Raises:
            TypeError: If ``generation`` is not an integer, if
                ``population`` is not a list, or if ``statistics`` is
                not a 4-tuple.
        """
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise TypeError("generation must be an integer.")
        if not isinstance(population, list):
            raise TypeError("population must be a list of PromptCandidate.")
        if not isinstance(statistics, tuple) or len(statistics) != 4:
            raise TypeError(
                "statistics must be a 4-tuple of (best_candidate, "
                "best_score, average_score, worst_score)."
            )

        best_candidate, best_score, average_score, worst_score = statistics

        return GenerationSnapshot(
            generation=generation,
            population=population,
            best_candidate=best_candidate,
            best_score=best_score,
            average_score=average_score,
            worst_score=worst_score,
            population_size=len(population),
            elapsed_time_seconds=elapsed_time_seconds,
        )

    def _save_outputs(self, snapshot: GenerationSnapshot) -> None:
        """Persists a generation's outputs via the OutputManager.

        Delegates entirely to ``_output_manager``: writes the
        generation snapshot via ``save_snapshot()`` and the best
        candidate's prompt report via ``save_best_prompt()``. This
        method performs no file I/O itself.

        Args:
            snapshot: The GenerationSnapshot to persist. Must be an
                instance of ``_snapshot_class``.

        Raises:
            TypeError: If ``snapshot`` is not an instance of
                ``_snapshot_class``, or propagated from
                ``_output_manager`` if ``snapshot.best_candidate`` is
                not a valid PromptCandidate.
            ValueError: Propagated from ``_output_manager`` if
                ``snapshot.best_candidate.generation`` is negative.
        """
        if not isinstance(snapshot, GenerationSnapshot):
            raise TypeError(
                f"snapshot must be a GenerationSnapshot."
            )

        self._output_manager.save_snapshot(snapshot)
        self._output_manager.save_best_prompt(snapshot.best_candidate)

    def _select_elites(self, fitness_results: Any) -> list[PromptCandidate]:
        """Selects elite candidates to carry forward unmodified.

        Intended to delegate to ``_elitism`` (and/or
        ``_tournament_selector``) to choose the top-performing
        candidates from the current generation, using
        ``fitness_results`` to rank them. No selection logic is
        implemented in Stage 1.

        Args:
            fitness_results: The fitness data produced by
                ``_evaluate_population()`` for the current generation.
                Must not be ``None``.

        Returns:
            This method does not return in Stage 1; it always raises.

        Raises:
            ValueError: If ``fitness_results`` is ``None``.
            NotImplementedError: Always, in Stage 1, once validation
                passes.
        """
        if fitness_results is None:
            raise ValueError("fitness_results must not be None.")

        raise NotImplementedError(
            "_select_elites() will be implemented in a later HFPO stage."
        )

    def _generate_offspring(
        self, elites: list[PromptCandidate], generation: int
    ) -> list[PromptCandidate]:
        """Generates offspring candidates via the PromptGenerator.

        Intended to delegate to ``_prompt_generator`` to produce new
        candidate prompts (via LLM-guided mutation/crossover) from the
        selected elites, tagging offspring with the appropriate
        generation and lineage metadata. No generation logic is
        implemented in Stage 1.

        Args:
            elites: The elite candidates selected by
                ``_select_elites()``. Must not be ``None``.
            generation: The zero-indexed generation offspring are
                being produced for. Must be a non-negative integer.

        Returns:
            This method does not return in Stage 1; it always raises.

        Raises:
            TypeError: If ``generation`` is not an integer.
            ValueError: If ``elites`` is ``None`` or ``generation`` is
                negative.
            NotImplementedError: Always, in Stage 1, once validation
                passes.
        """
        if elites is None:
            raise ValueError("elites must not be None.")
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise TypeError("generation must be an integer.")
        if generation < 0:
            raise ValueError("generation must be non-negative.")

        raise NotImplementedError(
            "_generate_offspring() will be implemented in a later HFPO stage."
        )

    def _build_next_population(
        self,
        elites: list[PromptCandidate],
        offspring: list[PromptCandidate],
    ) -> Population:
        """Assembles the next generation's Population.

        Intended to combine ``elites`` and ``offspring`` into a new
        ``Population`` instance that replaces ``_population`` for the
        next generation. No population-management logic is
        implemented in Stage 1.

        Args:
            elites: The elite candidates preserved from the current
                generation. Must not be ``None``.
            offspring: The newly generated candidates produced by
                ``_generate_offspring()``. Must not be ``None``.

        Returns:
            This method does not return in Stage 1; it always raises.

        Raises:
            ValueError: If ``elites`` or ``offspring`` is ``None``.
            NotImplementedError: Always, in Stage 1, once validation
                passes.
        """
        if elites is None:
            raise ValueError("elites must not be None.")
        if offspring is None:
            raise ValueError("offspring must not be None.")

        raise NotImplementedError(
            "_build_next_population() will be implemented in a later HFPO stage."
        )

    def _generation_statistics(
        self, evaluated_population: list[PromptCandidate]
    ) -> tuple[PromptCandidate, float, float, float]:
        """Computes reporting statistics for an evaluated generation.

        Ranks candidates by ``FitnessVector.average()`` for reporting
        purposes only. This is intentionally independent of the
        ranking strategy configured on ``_tournament_selector`` or
        ``_elitism`` ("average", "minimum", or "weighted"); those
        strategies govern *selection*, while this method governs what
        gets *reported* in the generation snapshot and is not
        configurable at this stage. Performs no file writing and does
        not mutate ``evaluated_population``.

        Args:
            evaluated_population: The list of evaluated
                ``PromptCandidate`` objects returned by
                ``_evaluate_population()``. Each must have a non-None
                ``fitness``.

        Returns:
            A 4-tuple of ``(best_candidate, best_score, average_score,
            worst_score)``, where ``best_candidate`` is the candidate
            with the highest ``FitnessVector.average()``, and the
            three scores are, respectively, the maximum, mean, and
            minimum of every candidate's ``FitnessVector.average()``.

        Raises:
            TypeError: If ``evaluated_population`` is not a list.
            ValueError: If ``evaluated_population`` is empty.
            RuntimeError: If any candidate has no ``FitnessVector``.
        """
        if not isinstance(evaluated_population, list):
            raise TypeError("evaluated_population must be a list of PromptCandidate.")
        if not evaluated_population:
            raise ValueError("evaluated_population must not be empty.")

        average_scores: list[float] = []
        for candidate in evaluated_population:
            if candidate.fitness is None:
                raise RuntimeError(
                    f"Candidate '{candidate.id}' has no FitnessVector; "
                    "cannot compute generation statistics."
                )

        scores = {
        candidate: candidate.fitness.average()
            for candidate in evaluated_population
                }

        best_candidate = max(scores, key=scores.get)
        best_score = scores[best_candidate]
        average_score = sum(scores.values()) / len(scores)
        worst_score = min(scores.values())

        return (
            best_candidate,
            best_score,
            average_score,
            worst_score,
        )