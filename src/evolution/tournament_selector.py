"""Tournament selection for HFPO (Heterogeneous Federated Prompt
Optimization).

This module defines :class:`TournamentSelector`, the component
responsible for selecting parent prompts via tournament selection.
A tournament works by randomly sampling a small group of prompts from the
population (without replacement within that single tournament), scoring
each according to a configurable ranking strategy, and returning the
highest-scoring prompt as the tournament winner. `TournamentSelector`
performs no mutation, crossover, prompt generation, elitism, or population
modification, and it never ranks the entire population at once; its sole
responsibility is selecting individual parents, one tournament at a time.
"""

from __future__ import annotations

from random import Random

from src.evolution.population import Population
from src.core.prompt_candidate import PromptCandidate

_VALID_RANKING_STRATEGIES: frozenset[str] = frozenset(
    {"average", "minimum", "weighted"}
)


class TournamentSelector:
    """Selects parent prompts from a Population via tournament selection.

    `TournamentSelector` repeatedly runs tournaments to select parent
    prompts for the genetic algorithm. Each tournament randomly samples
    `tournament_size` prompts from the population without replacement
    within that tournament, scores them according to the configured
    `ranking_strategy`, and returns the highest-scoring prompt as the
    winner. Across multiple tournaments (for example, when selecting many
    parents via `select_parents`), the same prompt may win more than once,
    since each tournament is independent.

    `TournamentSelector` never modifies the `Population` it is given, and
    never modifies any `PromptCandidate`; it only returns references to
    the winning prompts. It performs no mutation, crossover, prompt
    generation, elitism, or whole-population ranking; those
    responsibilities belong to other components.

    Attributes:
        tournament_size: The number of prompts sampled per tournament.
        ranking_strategy: The strategy used to score prompts within a
            tournament. One of "average", "minimum", or "weighted".

    Raises:
        ValueError: If `tournament_size` is less than 2, or if
            `ranking_strategy` is not one of "average", "minimum", or
            "weighted".
    """

    def __init__(
        self,
        tournament_size: int = 4,
        ranking_strategy: str = "average",
        random_seed: int | None = None,
    ) -> None:
        """Initialize a TournamentSelector with its configuration.

        Args:
            tournament_size: The number of prompts sampled per tournament.
                Defaults to 4.
            ranking_strategy: The strategy used to score prompts within a
                tournament. One of "average", "minimum", or "weighted".
                Defaults to "average".
            random_seed: Optional seed for reproducible tournament
                sampling. If `None`, sampling is non-deterministic.

        Raises:
            ValueError: If `tournament_size` is less than 2, or if
                `ranking_strategy` is not a recognized strategy.
        """
        if tournament_size < 2:
            raise ValueError(
                f"TournamentSelector tournament_size must be >= 2, got "
                f"{tournament_size}."
            )

        if ranking_strategy not in _VALID_RANKING_STRATEGIES:
            raise ValueError(
                f"TournamentSelector ranking_strategy must be one of "
                f"{sorted(_VALID_RANKING_STRATEGIES)}, got "
                f"'{ranking_strategy}'."
            )

        self.tournament_size = tournament_size
        self.ranking_strategy = ranking_strategy
        self._rng = Random(random_seed)

    def _compute_score(self, prompt: PromptCandidate) -> float:
        """Compute a prompt's tournament score under the configured strategy.

        Args:
            prompt: The `PromptCandidate` whose score should be computed.

        Returns:
            The prompt's score: its fitness average if
            `ranking_strategy` is "average", its fitness minimum if
            "minimum", or `0.7 * average + 0.3 * minimum` if "weighted".

        Raises:
            RuntimeError: If `prompt.fitness` is `None`.
        """
        if prompt.fitness is None:
            raise RuntimeError(
                f"Prompt '{prompt.id}' has no FitnessVector; cannot "
                f"compute a tournament score."
            )

        if self.ranking_strategy == "average":
            return prompt.fitness.average()

        if self.ranking_strategy == "minimum":
            return prompt.fitness.minimum()

        return 0.7 * prompt.fitness.average() + 0.3 * prompt.fitness.minimum()

    def run_tournament(self, population: Population) -> PromptCandidate:
        """Run a single tournament and return its winner.

        Randomly samples `tournament_size` prompts from `population`
        without replacement, scores each using the configured ranking
        strategy, and returns the highest-scoring prompt. `population` is
        not modified.

        Args:
            population: The `Population` to sample from.

        Returns:
            The winning `PromptCandidate`.

        Raises:
            ValueError: If `tournament_size` exceeds the size of
                `population`.
            RuntimeError: If any sampled prompt has no `FitnessVector`.
        """
        if self.tournament_size > len(population):
            raise ValueError(
                f"TournamentSelector tournament_size "
                f"({self.tournament_size}) exceeds population size "
                f"({len(population)})."
            )

        contestants = self._rng.sample(list(population), self.tournament_size)

        return max(contestants, key=self._compute_score)

    def select_parents(
        self, population: Population, n_parents: int
    ) -> list[PromptCandidate]:
        """Select multiple parents by running repeated tournaments.

        Runs `run_tournament` repeatedly, with replacement across
        tournaments, until `n_parents` winners have been collected. The
        same prompt may be selected more than once, since each tournament
        is independent.

        Args:
            population: The `Population` to select parents from.
            n_parents: The number of parents to select.

        Returns:
            A list of `n_parents` `PromptCandidate` objects, in the order
            they were selected.

        Raises:
            ValueError: If `n_parents` is not positive, or if
                `tournament_size` exceeds the size of `population`.
            RuntimeError: If any sampled prompt has no `FitnessVector`.
        """
        if n_parents <= 0:
            raise ValueError(
                f"TournamentSelector.select_parents n_parents must be > "
                f"0, got {n_parents}."
            )

        return [self.run_tournament(population) for _ in range(n_parents)]

    def tournament_summary(self, population: Population) -> dict[str, object]:
        """Return a logging-friendly summary of this selector's configuration.

        Args:
            population: The `Population` this summary describes selection
                against.

        Returns:
            A dictionary containing the population size, tournament size,
            and ranking strategy.
        """
        return {
            "population_size": len(population),
            "tournament_size": self.tournament_size,
            "ranking_strategy": self.ranking_strategy,
        }

    def __str__(self) -> str:
        """Return a concise, human-readable summary of this selector.

        Returns:
            A string of the form:
            'TournamentSelector(tournament_size=4,
            ranking_strategy="average")'.
        """
        return (
            f"TournamentSelector(\n"
            f"    tournament_size={self.tournament_size},\n"
            f'    ranking_strategy="{self.ranking_strategy}"\n'
            f")"
        )