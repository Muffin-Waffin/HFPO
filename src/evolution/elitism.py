"""Elitism selection for the evolutionary optimization subsystem.

This module defines Elitism, which is responsible for preserving the
best-performing prompts between generations. It receives an evaluated
Population and returns the top N PromptCandidate objects unchanged.
It performs no mutation, no crossover, no prompt generation, no
population modification, no fitness computation, and no federated
evaluation.
"""

from src.core.fitness_vector import FitnessVector
from src.core.prompt_candidate import PromptCandidate
from src.core.population import Population

_RANKING_STRATEGIES: frozenset[str] = frozenset(
    {"average", "minimum", "weighted"}
)


class Elitism:
    """Selects the top-performing PromptCandidates from a population.

    Elitism ranks an already-evaluated Population's PromptCandidates
    by a configured ranking strategy ("average", "minimum", or
    "weighted") and returns the top elite_count candidates unchanged,
    by reference, without cloning them or modifying the population.
    It performs no mutation, crossover, prompt generation, population
    modification, fitness computation, or federated evaluation.
    """

    __slots__ = ("_elite_count", "_ranking_strategy")

    def __init__(
        self,
        elite_count: int = 2,
        ranking_strategy: str = "average",
    ) -> None:
        """Initializes the elitism selector.

        Args:
            elite_count: The number of top-performing candidates to
                preserve between generations.
            ranking_strategy: One of "average", "minimum", or
                "weighted".

        Raises:
            ValueError: If elite_count <= 0 or ranking_strategy is
                invalid.
        """
        if elite_count <= 0:
            raise ValueError(
                f"elite_count must be > 0, got {elite_count}."
            )

        if ranking_strategy not in _RANKING_STRATEGIES:
            raise ValueError(
                "ranking_strategy must be one of "
                f"{sorted(_RANKING_STRATEGIES)}, got "
                f"{ranking_strategy!r}."
            )

        self._elite_count = elite_count
        self._ranking_strategy = ranking_strategy

    @property
    def elite_count(self) -> int:
        """Return the configured elite count."""
        return self._elite_count

    @property
    def ranking_strategy(self) -> str:
        """Return the configured ranking strategy."""
        return self._ranking_strategy

    def select_elites(
        self,
        population: Population,
    ) -> list[PromptCandidate]:
        """Select the top-performing prompt candidates.

        Args:
            population: An evaluated Population.

        Returns:
            The elite PromptCandidates ordered from highest to lowest
            fitness.

        Raises:
            TypeError: If population is not a Population.
            ValueError: If population is empty or smaller than the
                configured elite count.
            RuntimeError: If any candidate has no FitnessVector.
        """
        if not isinstance(population, Population):
            raise TypeError(
                "population must be a Population, got "
                f"{type(population).__name__}."
            )

        if len(population) == 0:
            raise ValueError("population must not be empty.")

        if len(population) < self._elite_count:
            raise ValueError(
                "population size must be >= elite_count, got "
                f"{len(population)} and elite_count "
                f"{self._elite_count}."
            )

        ranked = sorted(
            population,
            key=self._compute_score,
            reverse=True,
        )

        return ranked[: self._elite_count]

    def summary(
        self,
        population: Population,
    ) -> dict[str, object]:
        """Return a logging-friendly summary.

        Args:
            population: The evaluated Population.

        Returns:
            Dictionary summarizing this selector.
        """
        return {
            "population_size": len(population),
            "elite_count": self._elite_count,
            "ranking_strategy": self._ranking_strategy,
        }

    def _compute_score(
        self,
        prompt: PromptCandidate,
    ) -> float:
        """Compute a candidate's ranking score.

        Args:
            prompt: PromptCandidate whose score should be computed.

        Returns:
            Ranking score according to the configured strategy.

        Raises:
            RuntimeError: If prompt has no FitnessVector.
        """
        fitness: FitnessVector | None = prompt.fitness

        if fitness is None:
            raise RuntimeError(
                f"Candidate '{prompt.id}' has no FitnessVector."
            )

        if self._ranking_strategy == "average":
            return fitness.average()

        if self._ranking_strategy == "minimum":
            return fitness.minimum()

        return (
            0.7 * fitness.average()
            + 0.3 * fitness.minimum()
        )

    def __str__(self) -> str:
        """Return a concise human-readable summary."""
        return (
            "Elitism(\n"
            f"    elite_count={self._elite_count},\n"
            f'    ranking_strategy="{self._ranking_strategy}"\n'
            ")"
        )