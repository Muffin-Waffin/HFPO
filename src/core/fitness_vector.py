"""Core fitness representation for HFPO (Heterogeneous Federated Prompt Optimization).

This module defines :class:`FitnessVector`, the primary data structure used
throughout HFPO to represent how a single candidate prompt performs across a
federation of hospitals. Each hospital evaluates a prompt locally and returns
a single normalized score; the collection of these per-hospital scores forms
the prompt's fitness vector. Downstream components (selection, crossover,
mutation, and convergence analysis in the genetic algorithm) operate on
:class:`FitnessVector` instances rather than raw per-hospital data, making it
the canonical fitness abstraction for the system.
"""

from dataclasses import dataclass
from math import sqrt


@dataclass(slots=True)
class FitnessVector:
    """Represents the distributed fitness of a single prompt across hospitals.

    A `FitnessVector` captures how one candidate prompt performed when
    evaluated independently by each participating hospital in the federation.
    Each hospital contributes exactly one normalized score in the range
    [0.0, 1.0], where higher values indicate better performance. This class
    is the primary fitness representation used throughout HFPO: it is
    produced after each round of federated evaluation and consumed by the
    genetic algorithm's selection, ranking, and convergence logic.

    Attributes:
        scores: A mapping from hospital ID to that hospital's normalized
            evaluation score for the prompt. Must contain at least one
            entry, and every score must lie within [0.0, 1.0] inclusive.

    Raises:
        ValueError: If `scores` is empty, or if any score falls outside
            the inclusive range [0.0, 1.0].
    """

    scores: dict[str, float]

    def __post_init__(self) -> None:
        """Validate the fitness scores supplied at construction time.

        Raises:
            ValueError: If `scores` is empty, or if any score is not within
                the inclusive range [0.0, 1.0].
        """
        if not self.scores:
            raise ValueError(
                "FitnessVector requires at least one hospital score; "
                "received an empty scores mapping."
            )

        for hospital_id, score in self.scores.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"Score for hospital '{hospital_id}' must be in the "
                    f"range [0.0, 1.0], but got {score}."
                )

    def average(self) -> float:
        """Compute the mean score across all hospitals.

        Returns:
            The arithmetic mean of all per-hospital scores.
        """
        return sum(self.scores.values()) / len(self.scores)

    def minimum(self) -> float:
        """Return the lowest score among all hospitals.

        Returns:
            The minimum per-hospital score.
        """
        return min(self.scores.values())

    def maximum(self) -> float:
        """Return the highest score among all hospitals.

        Returns:
            The maximum per-hospital score.
        """
        return max(self.scores.values())

    def variance(self) -> float:
        """Compute the population variance of scores across hospitals.

        Returns:
            The population variance of the per-hospital scores, or 0.0 if
            fewer than two hospitals are present.
        """
        if len(self.scores) < 2:
            return 0.0

        mean = self.average()
        return sum((score - mean) ** 2 for score in self.scores.values()) / len(
            self.scores
        )

    def standard_deviation(self) -> float:
        """Compute the population standard deviation of scores.

        Returns:
            The population standard deviation of the per-hospital scores,
            or 0.0 if fewer than two hospitals are present.
        """
        if len(self.scores) < 2:
            return 0.0

        return sqrt(self.variance())

    def best_hospital(self) -> str:
        """Identify the hospital that reported the highest score.

        Returns:
            The hospital ID associated with the maximum score. If multiple
            hospitals share the maximum score, the first one encountered in
            insertion order is returned.
        """
        return max(self.scores, key=lambda hospital_id: self.scores[hospital_id])

    def worst_hospital(self) -> str:
        """Identify the hospital that reported the lowest score.

        Returns:
            The hospital ID associated with the minimum score. If multiple
            hospitals share the minimum score, the first one encountered in
            insertion order is returned.
        """
        return min(self.scores, key=lambda hospital_id: self.scores[hospital_id])

    def as_dict(self) -> dict[str, float]:
        """Return a copy of the hospital-to-score mapping.

        Returns:
            A shallow copy of the mapping from hospital ID to its normalized
            evaluation score. Returning a copy prevents callers from
            accidentally mutating the internal state of the FitnessVector.
        """
        return dict(self.scores)

    def __str__(self) -> str:
        """Return a concise, human-readable summary of the fitness vector.

        Returns:
            A string of the form
            "FitnessVector(avg=0.8231, min=0.7900, max=0.8700)".
        """
        return (
            f"FitnessVector(avg={self.average():.4f}, "
            f"min={self.minimum():.4f}, max={self.maximum():.4f})"
        )