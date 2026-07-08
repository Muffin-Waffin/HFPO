"""Immutable generation snapshot for the evolutionary optimization
subsystem.

This module defines GenerationSnapshot, an immutable record
describing one completed generation of the evolutionary optimization
process. It exists for logging, checkpointing, visualization,
experiment analysis, and paper generation. It performs no mutation,
no crossover, no evaluation, no federated communication, no prompt
generation, no population modification, and no file writing -- it is
purely a data object.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from src.core.prompt_candidate import PromptCandidate


@dataclass(slots=True, frozen=True)
class GenerationSnapshot:
    """An immutable record of one completed evolutionary generation.

    GenerationSnapshot captures the full population, the best
    candidate, and summary fitness statistics for a single completed
    generation. It never modifies the supplied population; the
    incoming population is defensively copied into an immutable tuple
    during construction.

    Attributes:
        generation: The index of the completed generation.
        population: The full population of PromptCandidates evaluated
            during this generation, stored as an immutable tuple.
        best_candidate: The highest-scoring PromptCandidate in
            population.
        best_score: The score of best_candidate, in [0.0, 1.0].
        average_score: The mean score across population, in
            [0.0, 1.0].
        worst_score: The lowest score in population, in [0.0, 1.0].
        population_size: The number of candidates in population.
        elapsed_time_seconds: The wall-clock time taken to complete
            this generation, in seconds.
    """

    generation: int
    population: Sequence[PromptCandidate]
    best_candidate: PromptCandidate
    best_score: float
    average_score: float
    worst_score: float
    population_size: int
    elapsed_time_seconds: float

    def __post_init__(self) -> None:
        """Validate field invariants and freeze the population.

        Converts the supplied population into an immutable tuple via
        object.__setattr__, since this dataclass is frozen.

        Raises:
            TypeError: If population contains non-PromptCandidate
                objects or best_candidate is not a PromptCandidate.
            ValueError: If generation is negative, if population is
                empty, if population_size does not equal
                len(population), if best_candidate is not a member of
                population, if elapsed_time_seconds is negative, if
                best_score, average_score, or worst_score lie outside
                [0.0, 1.0], or if the scores do not satisfy
                best_score >= average_score >= worst_score.
        """
        if self.generation < 0:
            raise ValueError(
                f"generation must be >= 0, got {self.generation}."
            )

        population_tuple = tuple(self.population)
        object.__setattr__(self, "population", population_tuple)

        if len(population_tuple) == 0:
            raise ValueError("population must not be empty.")

        for candidate in population_tuple:
            if not isinstance(candidate, PromptCandidate):
                raise TypeError(
                    "population must contain only PromptCandidate "
                    f"objects, got {type(candidate).__name__}."
                )

        if not isinstance(self.best_candidate, PromptCandidate):
            raise TypeError(
                "best_candidate must be a PromptCandidate."
            )

        if self.population_size != len(population_tuple):
            raise ValueError(
                "population_size must equal len(population), got "
                f"population_size={self.population_size} and "
                f"len(population)={len(population_tuple)}."
            )

        if self.best_candidate not in population_tuple:
            raise ValueError(
                "best_candidate must be a member of population."
            )

        if self.elapsed_time_seconds < 0:
            raise ValueError(
                "elapsed_time_seconds must be >= 0, got "
                f"{self.elapsed_time_seconds}."
            )

        for name, value in (
            ("best_score", self.best_score),
            ("average_score", self.average_score),
            ("worst_score", self.worst_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must lie within [0.0, 1.0], got {value}."
                )

        if not (
            self.best_score
            >= self.average_score
            >= self.worst_score
        ):
            raise ValueError(
                "Scores must satisfy "
                "best_score >= average_score >= worst_score."
            )

    def best_prompt(self) -> str:
        """Return the text of the best candidate.

        Returns:
            The prompt text of best_candidate.
        """
        return self.best_candidate.text

    def to_dict(self) -> dict[str, object]:
        """Convert this snapshot into a JSON-serializable dictionary.

        Returns:
            A dictionary representation of this snapshot. Population
            members are represented only by their IDs.
        """
        return {
            "generation": self.generation,
            "population": [
                candidate.id for candidate in self.population
            ],
            "best_candidate_id": self.best_candidate.id,
            "best_score": self.best_score,
            "average_score": self.average_score,
            "worst_score": self.worst_score,
            "population_size": self.population_size,
            "elapsed_time_seconds": self.elapsed_time_seconds,
        }

    def __str__(self) -> str:
        """Return a concise human-readable summary."""
        return (
            "GenerationSnapshot(\n"
            f"    generation={self.generation},\n"
            f"    population_size={self.population_size},\n"
            f"    best_score={self.best_score:.4f},\n"
            f"    average_score={self.average_score:.4f},\n"
            f"    worst_score={self.worst_score:.4f}\n"
            ")"
        )