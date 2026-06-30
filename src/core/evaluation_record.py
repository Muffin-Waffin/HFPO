"""Single-hospital evaluation result representation for HFPO.

This module defines :class:`EvaluationRecord`, the communication object
exchanged between a `HospitalClient` and the Federated Server during
HFPO (Heterogeneous Federated Prompt Optimization). Each `EvaluationRecord`
represents the outcome of one hospital locally evaluating one candidate
prompt. The Federated Server later collects multiple `EvaluationRecord`
objects, one per hospital, for the same prompt and aggregates them into a
`FitnessVector`. This module is not responsible for that aggregation; it
only defines the per-hospital evaluation result itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvaluationRecord:
    """Represents one local evaluation performed by one hospital on one prompt.

    `EvaluationRecord` is the communication object exchanged between a
    `HospitalClient` and the Federated Server before `FitnessVector`
    objects are constructed. Each record captures the result of a single
    hospital evaluating a single `PromptCandidate` against its local
    question set: the normalized score achieved, the raw correct/total
    counts behind that score, and any metadata useful for logging or
    reproducibility. Multiple `EvaluationRecord` objects for the same
    prompt, one from each participating hospital, are later aggregated by
    the Federated Server into a single `FitnessVector`. This class performs
    no aggregation or ranking itself.

    Attributes:
        prompt_id: UUID of the `PromptCandidate` that was evaluated.
        hospital_id: Identifier of the hospital that performed the
            evaluation.
        generation: Generation number during which the evaluation
            occurred.
        score: Normalized evaluation score in the range [0.0, 1.0].
        num_correct: Number of correctly answered questions.
        num_total: Total number of questions evaluated.
        metadata: Optional metadata for logging and reproducibility.
            Defaults to an empty dictionary.

    Raises:
        ValueError: If any field fails validation in `__post_init__`.
    """

    prompt_id: str
    hospital_id: str
    generation: int
    score: float
    num_correct: int
    num_total: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate field values for this evaluation record.

        Raises:
            ValueError: If `prompt_id` or `hospital_id` is empty or
                whitespace, `generation` is negative, `score` is outside
                [0.0, 1.0], `num_correct` is negative, `num_total` is not
                positive, or `num_correct` exceeds `num_total`.
        """
        if not self.prompt_id.strip():
            raise ValueError(
                "EvaluationRecord prompt_id must not be empty or whitespace."
            )

        if not self.hospital_id.strip():
            raise ValueError(
                "EvaluationRecord hospital_id must not be empty or whitespace."
            )

        if self.generation < 0:
            raise ValueError(
                f"EvaluationRecord generation must be >= 0, got {self.generation}."
            )

        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"EvaluationRecord score must be in the range [0.0, 1.0], "
                f"got {self.score}."
            )

        if self.num_correct < 0:
            raise ValueError(
                f"EvaluationRecord num_correct must be >= 0, got "
                f"{self.num_correct}."
            )

        if self.num_total <= 0:
            raise ValueError(
                f"EvaluationRecord num_total must be > 0, got {self.num_total}."
            )

        if self.num_correct > self.num_total:
            raise ValueError(
                f"EvaluationRecord num_correct ({self.num_correct}) cannot "
                f"exceed num_total ({self.num_total})."
            )

    def accuracy(self) -> float:
        """Return the evaluation score.

        This method exists for semantic readability throughout the project,
        allowing callers to refer to the score as an accuracy where that
        framing is clearer.

        Returns:
            The stored normalized score.
        """
        return self.score

    def error_rate(self) -> float:
        """Compute the complement of the evaluation score.

        Returns:
            `1.0 - score`.
        """
        return 1.0 - self.score

    def is_perfect(self) -> bool:
        """Check whether this evaluation achieved a perfect score.

        Returns:
            True if `score == 1.0`, False otherwise.
        """
        return self.score == 1.0

    def is_failed(self) -> bool:
        """Check whether this evaluation achieved a zero score.

        Returns:
            True if `score == 0.0`, False otherwise.
        """
        return self.score == 0.0

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            A dictionary containing all fields of this record. Mutable
            fields are returned as copies to prevent external mutation of
            internal state.
        """
        return {
            "prompt_id": self.prompt_id,
            "hospital_id": self.hospital_id,
            "generation": self.generation,
            "score": self.score,
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "metadata": dict(self.metadata),
        }

    def __str__(self) -> str:
        """Return a concise, human-readable summary of this record.

        Returns:
            A string of the form:
            'EvaluationRecord(prompt="abc123...", hospital="hospital_1",
            generation=2, score=0.8500)'.
        """
        return (
            f'EvaluationRecord(\n'
            f'    prompt="{self.prompt_id}",\n'
            f'    hospital="{self.hospital_id}",\n'
            f'    generation={self.generation},\n'
            f'    score={self.score:.4f}\n'
            f')'
        )