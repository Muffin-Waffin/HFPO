"""Core evolutionary unit representation for HFPO.

This module defines :class:`PromptCandidate`, the fundamental evolutionary
unit in HFPO (Heterogeneous Federated Prompt Optimization). Every prompt
that exists in any generation of the genetic algorithm population is
represented by exactly one `PromptCandidate` instance. The class stores the
prompt's text, its lineage (parents and full ancestry chain), how it was
produced (seed, mutation, crossover, or elite survival), its federated
evaluation result once available, and arbitrary metadata used for logging
and experiment reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.fitness_vector import FitnessVector

_VALID_ORIGINS: frozenset[str] = frozenset({"seed", "mutation", "crossover", "elite"})


@dataclass(slots=True)
class PromptCandidate:
    """Represents a single candidate prompt in the evolutionary population.

    `PromptCandidate` is the fundamental evolutionary unit in HFPO. Every
    prompt in every generation, whether created as an initial seed,
    produced via mutation or crossover, or carried forward unchanged as an
    elite, is represented by one `PromptCandidate` object. The class
    captures the prompt's text, lineage information (direct parents and the
    full ancestry chain back to the seed prompts), its evolutionary origin,
    its fitness once federated evaluation has occurred, and arbitrary
    metadata used for logging and experiment reproducibility.

    Attributes:
        id: Unique UUID string identifying this prompt. Generated
            externally by the evolution engine; this class does not
            generate IDs itself.
        text: The actual system prompt text.
        generation: Generation number in which this prompt was created.
            Seed prompts belong to generation 0.
        parent_ids: IDs of the direct parents of this prompt. Empty for
            seed prompts, length 1 for mutation or elite survival, and
            length 2 for crossover.
        ancestry_ids: Complete ancestry chain back to the seed prompts.
            Empty for seed prompts.
        origin: How this prompt was produced. One of "seed", "mutation",
            "crossover", or "elite".
        fitness: The prompt's fitness as a `FitnessVector`, assigned after
            federated evaluation. Newly generated prompts begin with
            `None`.
        metadata: Arbitrary metadata used for logging and experiments.
            Defaults to an empty dictionary.

    Raises:
        ValueError: If any field fails validation in `__post_init__`.
    """

    id: str
    text: str
    generation: int
    parent_ids: list[str]
    ancestry_ids: list[str]
    origin: str
    fitness: FitnessVector | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate field values and lineage consistency for this prompt.

        Raises:
            ValueError: If `text` is empty or whitespace, `generation` is
                negative, `origin` is not a recognized value, or the
                `parent_ids` count is inconsistent with the declared
                `origin` (seed prompts must have no parents or ancestry,
                mutation and elite prompts must have exactly one parent,
                and crossover prompts must have exactly two parents).
        """
        if not self.text.strip():
            raise ValueError("PromptCandidate text must not be empty or whitespace.")

        if self.generation < 0:
            raise ValueError(
                f"PromptCandidate generation must be >= 0, got {self.generation}."
            )

        if self.origin not in _VALID_ORIGINS:
            raise ValueError(
                f"PromptCandidate origin must be one of "
                f"{sorted(_VALID_ORIGINS)}, got '{self.origin}'."
            )

        if self.origin == "seed":
            if self.parent_ids:
                raise ValueError(
                    "Seed prompts must have empty parent_ids, got "
                    f"{self.parent_ids}."
                )
            if self.ancestry_ids:
                raise ValueError(
                    "Seed prompts must have empty ancestry_ids, got "
                    f"{self.ancestry_ids}."
                )
        elif self.origin == "mutation":
            if len(self.parent_ids) != 1:
                raise ValueError(
                    "Mutation prompts must have exactly one parent_id, "
                    f"got {len(self.parent_ids)}."
                )
        elif self.origin == "crossover":
            if len(self.parent_ids) != 2:
                raise ValueError(
                    "Crossover prompts must have exactly two parent_ids, "
                    f"got {len(self.parent_ids)}."
                )
        elif self.origin == "elite":
            if len(self.parent_ids) != 1:
                raise ValueError(
                    "Elite prompts must have exactly one parent_id, "
                    f"got {len(self.parent_ids)}."
                )

    def is_evaluated(self) -> bool:
        """Check whether this prompt has been federated-evaluated.

        Returns:
            True if `fitness` is not `None`, False otherwise.
        """
        return self.fitness is not None

    def has_parents(self) -> bool:
        """Check whether this prompt has any recorded parents.

        Returns:
            True if `parent_ids` is non-empty, False otherwise.
        """
        return len(self.parent_ids) > 0

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            A dictionary containing all fields of this prompt. If `fitness`
            is set, it is represented via `FitnessVector.as_dict()`;
            otherwise the fitness field is stored as `None`.
        """
        return {
            "id": self.id,
            "text": self.text,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "ancestry_ids": self.ancestry_ids,
            "origin": self.origin,
            "fitness": self.fitness.as_dict() if self.fitness is not None else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PromptCandidate":
        """Reconstruct a PromptCandidate from its dictionary representation.

        Args:
            data: A dictionary as produced by ``as_dict()``.

        Returns:
            A new PromptCandidate instance.

        Raises:
            TypeError: If ``data`` is not a dict.
            ValueError: If required fields are missing or invalid
                (propagated from the constructor).
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"data must be a dict, got {type(data).__name__}."
            )

        fitness_data = data.get("fitness")
        fitness = (
            FitnessVector.from_dict(fitness_data)
            if fitness_data is not None
            else None
        )

        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            generation=int(data["generation"]),
            parent_ids=list(data["parent_ids"]),
            ancestry_ids=list(data["ancestry_ids"]),
            origin=str(data["origin"]),
            fitness=fitness,
            metadata=dict(data.get("metadata", {})),
        )

    def __str__(self) -> str:
        """Return a concise, human-readable summary of this prompt.

        Returns:
            A string of the form:
            'PromptCandidate(id="abc123...", generation=3,
            origin="mutation", evaluated=True)'.
        """
        return (
            f'PromptCandidate(\n'
            f'    id="{self.id}",\n'
            f'    generation={self.generation},\n'
            f'    origin="{self.origin}",\n'
            f'    evaluated={self.is_evaluated()}\n'
            f')'
        )