"""Immutable data models for the Prompt Generator subsystem.

This module contains ONLY dataclasses used to move structured data
between PromptGenerator components. It contains no LLM logic, no
prompt generation logic, no PromptCandidate construction logic, no
prompt validation/cleaning logic, no template-building logic, and no
other business logic of any kind.
"""

from dataclasses import dataclass

from src.core.prompt_candidate import PromptCandidate


@dataclass(slots=True, frozen=True)
class GenerationMetadata:
    """Metadata describing a single prompt generation event.

    Attributes:
        generator: Name or identifier of the generator that produced
            the prompt (e.g. an LLM model name or strategy name).
        temperature: Sampling temperature used during generation.
        attempts: Number of attempts taken to produce a valid prompt.
        template_used: Identifier of the template used to construct
            the generation request.
        generation_time_ms: Wall-clock time, in milliseconds, taken to
            generate the prompt.
    """

    generator: str
    temperature: float
    attempts: int
    template_used: str
    generation_time_ms: float
    mutation_operator: str | None = None

    def __post_init__(self) -> None:
        """Validates field invariants after initialization.

        Raises:
            ValueError: If any field violates its documented invariant.
        """
        if not self.generator.strip():
            raise ValueError("generator must not be empty.")
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be >= 0, got {self.temperature}."
            )
        if self.attempts < 1:
            raise ValueError(
                f"attempts must be >= 1, got {self.attempts}."
            )
        if not self.template_used.strip():
            raise ValueError("template_used must not be empty.")
        if self.generation_time_ms < 0:
            raise ValueError(
                "generation_time_ms must be >= 0, got "
                f"{self.generation_time_ms}."
            )

    def to_dict(self) -> dict[str, object]:
        """Converts this metadata instance into a plain dictionary.

        Returns:
            A dictionary containing all fields of this instance, keyed
            by field name.
        """
        return {
            "generator": self.generator,
            "temperature": self.temperature,
            "attempts": self.attempts,
            "template_used": self.template_used,
            "mutation_operator": self.mutation_operator,
            "generation_time_ms": self.generation_time_ms,
        }

    def __str__(self) -> str:
        """Returns a concise, human-readable summary of this metadata.

        Returns:
            A string summarizing the generator, temperature, attempts,
            template used, and generation time.
        """
        return (
            f"GenerationMetadata(generator={self.generator!r}, "
            f"temperature={self.temperature}, attempts={self.attempts}, "
            f"template_used={self.template_used!r}, "
            f"generation_time_ms={self.generation_time_ms})"
        )


@dataclass(slots=True, frozen=True)
class PromptGenerationRequest:
    """Everything needed to generate one new prompt.

    Attributes:
        parent_a: The primary parent candidate used as a generation
            source.
        parent_b: An optional secondary parent candidate used for
            crossover-style generation. None indicates a non-crossover
            request.
        generation: The generation (epoch) index this request belongs
            to.
        task_description: Description of the task the generated
            prompt must address.
        temperature: Sampling temperature to use during generation.
        existing_prompt_texts: Set of prompt texts that already exist,
            used to avoid generating duplicates.
    """

    parent_a: PromptCandidate
    parent_b: PromptCandidate | None
    generation: int
    task_description: str
    temperature: float
    existing_prompt_texts: set[str]

    def __post_init__(self) -> None:
        """Validates field invariants and defensively copies state.

        Raises:
            ValueError: If any field violates its documented invariant,
                including a None parent_a, a None
                existing_prompt_texts, a negative generation index, an
                empty task_description, a negative temperature, or a
                parent_b sharing an id with parent_a.
        """
        if self.parent_a is None:
            raise ValueError("parent_a must not be None.")
        if self.generation < 0:
            raise ValueError(
                f"generation must be >= 0, got {self.generation}."
            )
        if not self.task_description.strip():
            raise ValueError("task_description must not be empty.")
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be >= 0, got {self.temperature}."
            )
        if self.existing_prompt_texts is None:
            raise ValueError("existing_prompt_texts must never be None.")
        if (
            self.parent_b is not None
            and self.parent_b.id == self.parent_a.id
        ):
            raise ValueError(
                "parent_b must not have the same id as parent_a."
            )
        object.__setattr__(
            self, "existing_prompt_texts", set(self.existing_prompt_texts)
        )

    def is_crossover(self) -> bool:
        """Indicates whether this request represents a crossover.

        Returns:
            True if parent_b is present, False otherwise.
        """
        return self.parent_b is not None

    def __str__(self) -> str:
        """Returns a concise, human-readable summary of this request.

        Returns:
            A string summarizing the parents, generation index, task
            description length, and crossover status. The full task
            description is omitted to keep the summary readable.
        """
        return (
            f"PromptGenerationRequest(parent_a={self.parent_a.id}, "
            f"parent_b={self.parent_b.id if self.parent_b else None}, "
            f"generation={self.generation}, "
            f"task_description_length={len(self.task_description)}, "
            f"is_crossover={self.is_crossover()})"
        )


@dataclass(slots=True, frozen=True)
class PromptGenerationResult:
    """One successfully generated prompt and its generation metadata.

    Attributes:
        candidate: The successfully generated prompt candidate.
        metadata: Metadata describing how the candidate was generated.
    """

    candidate: PromptCandidate
    metadata: GenerationMetadata

    def __post_init__(self) -> None:
        """Validates field invariants after initialization.

        Raises:
            ValueError: If candidate is None or metadata is None.
        """
        if self.candidate is None:
            raise ValueError("candidate must not be None.")
        if self.metadata is None:
            raise ValueError("metadata must not be None.")

    def to_dict(self) -> dict[str, object]:
        """Converts this result instance into a plain dictionary.

        Returns:
            A dictionary containing the candidate's dictionary
            representation under the "candidate" key and the
            metadata's dictionary representation under the "metadata"
            key.
        """
        return {
            "candidate": self.candidate.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    def __str__(self) -> str:
        """Returns a concise, human-readable summary of this result.

        Returns:
            A string summarizing the candidate id and its generation
            metadata.
        """
        return (
            f"PromptGenerationResult(candidate={self.candidate.id}, "
            f"metadata={self.metadata})"
        )