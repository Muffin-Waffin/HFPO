"""Generation container for HFPO (Heterogeneous Federated Prompt
Optimization).

This module defines :class:`Population`, a strongly validated container
representing exactly one generation of candidate prompts in HFPO. A
`Population` stores `PromptCandidate` objects, enforces uniqueness of
prompt IDs, enforces a maximum population size, and provides convenient
lookup, iteration, and bookkeeping operations. It performs no genetic
algorithm logic of any kind: no tournament selection, mutation, crossover,
ranking, or prompt generation. Those responsibilities belong to later
milestones. `Population` exists purely to safely hold and manage the set
of prompts that make up one generation.
"""

from __future__ import annotations

from typing import Iterator

from src.core.prompt_candidate import PromptCandidate


class Population:
    """Represents one generation of candidate prompts.

    `Population` is a strongly validated container for exactly one
    generation of `PromptCandidate` objects. It guarantees that prompt IDs
    within it are unique, that it never exceeds its configured maximum
    size, and that prompts can be looked up by ID or by index. It also
    provides a way to verify that every prompt in the population has been
    evaluated (has a `FitnessVector` attached). `Population` performs no
    tournament selection, mutation, crossover, ranking, or prompt
    generation; it is purely a container.

    Attributes:
        generation: The generation number this population represents.
        max_population_size: The maximum number of prompts this population
            may hold.

    Raises:
        ValueError: If `prompts` is empty, `generation` is negative,
            `max_population_size` is not positive, `prompts` exceeds
            `max_population_size`, or `prompts` contains duplicate IDs.
    """

    def __init__(
        self,
        prompts: list[PromptCandidate],
        generation: int,
        max_population_size: int,
    ) -> None:
        """Initialize a Population with its prompts and generation metadata.

        Args:
            prompts: The list of `PromptCandidate` objects that make up
                this generation.
            generation: The generation number this population represents.
            max_population_size: The maximum number of prompts this
                population may hold.

        Raises:
            ValueError: If `prompts` is empty, `generation` is negative,
                `max_population_size` is not positive, `prompts` exceeds
                `max_population_size`, or `prompts` contains duplicate
                IDs.
        """
        if not prompts:
            raise ValueError("Population prompts must not be empty.")

        if generation < 0:
            raise ValueError(
                f"Population generation must be >= 0, got {generation}."
            )

        if max_population_size <= 0:
            raise ValueError(
                f"Population max_population_size must be > 0, got "
                f"{max_population_size}."
            )

        if len(prompts) > max_population_size:
            raise ValueError(
                f"Population received {len(prompts)} prompts, which "
                f"exceeds max_population_size of {max_population_size}."
            )

        seen_ids: set[str] = set()
        for prompt in prompts:
            if prompt.id in seen_ids:
                raise ValueError(
                    f"Population received duplicate prompt id '{prompt.id}'."
                )
            seen_ids.add(prompt.id)

        self.generation = generation
        self.max_population_size = max_population_size
        self._prompts: dict[str, PromptCandidate] = {
            prompt.id: prompt for prompt in prompts
        }

    def add(self, prompt: PromptCandidate) -> None:
        """Add one PromptCandidate to the population.

        Args:
            prompt: The `PromptCandidate` to add.

        Raises:
            ValueError: If the population is already full, or if a prompt
                with the same ID is already present.
        """
        if self.is_full():
            raise ValueError(
                f"Population is full (max_population_size="
                f"{self.max_population_size}); cannot add prompt "
                f"'{prompt.id}'."
            )

        if prompt.id in self._prompts:
            raise ValueError(
                f"Population already contains a prompt with id "
                f"'{prompt.id}'."
            )

        self._prompts[prompt.id] = prompt

    def remove(self, prompt_id: str) -> None:
        """Remove a prompt from the population by ID.

        Args:
            prompt_id: The ID of the prompt to remove.

        Raises:
            KeyError: If no prompt with `prompt_id` exists.
        """
        if prompt_id not in self._prompts:
            raise KeyError(
                f"Population does not contain a prompt with id "
                f"'{prompt_id}'."
            )

        del self._prompts[prompt_id]

    def contains(self, prompt_id: str) -> bool:
        """Check whether a prompt with the given ID is in the population.

        Args:
            prompt_id: The prompt ID to check.

        Returns:
            True if the prompt is present, False otherwise.
        """
        return prompt_id in self._prompts

    def get(self, prompt_id: str) -> PromptCandidate:
        """Retrieve a prompt by its ID.

        Args:
            prompt_id: The ID of the prompt to retrieve.

        Returns:
            The matching `PromptCandidate`.

        Raises:
            KeyError: If no prompt with `prompt_id` exists.
        """
        if prompt_id not in self._prompts:
            raise KeyError(
                f"Population does not contain a prompt with id "
                f"'{prompt_id}'."
            )

        return self._prompts[prompt_id]

    def get_by_index(self, index: int) -> PromptCandidate:
        """Retrieve a prompt by its positional index.

        Args:
            index: The positional index of the prompt to retrieve, in
                insertion order.

        Returns:
            The `PromptCandidate` at the given index.

        Raises:
            IndexError: If `index` is out of range.
        """
        values = list(self._prompts.values())
        if index < 0 or index >= len(values):
            raise IndexError(
                f"Population index {index} is out of range for size "
                f"{len(values)}."
            )

        return values[index]

    def replace(self, prompt: PromptCandidate) -> None:
        """Replace an existing prompt with a new one sharing the same ID.

        Args:
            prompt: The `PromptCandidate` to insert in place of the
                existing prompt with the same ID.

        Raises:
            KeyError: If no prompt with `prompt.id` exists.
        """
        if prompt.id not in self._prompts:
            raise KeyError(
                f"Population does not contain a prompt with id "
                f"'{prompt.id}' to replace."
            )

        self._prompts[prompt.id] = prompt

    def ids(self) -> list[str]:
        """Return the IDs of every prompt in the population.

        Returns:
            A list of prompt IDs in insertion order.
        """
        return list(self._prompts.keys())

    def texts(self) -> list[str]:
        """Return the text of every prompt in the population.

        Returns:
            A list of prompt texts in insertion order.
        """
        return [prompt.text for prompt in self._prompts.values()]

    def validated(self) -> bool:
        """Check that every prompt in the population has been evaluated.

        Returns:
            True if every prompt has a non-`None` `fitness`.

        Raises:
            RuntimeError: If any prompt's `fitness` is `None`.
        """
        for prompt in self._prompts.values():
            if prompt.fitness is None:
                raise RuntimeError(
                    f"Prompt '{prompt.id}' has no FitnessVector; "
                    f"population is not fully evaluated."
                )

        return True

    def copy(self) -> Population:
        """Return a shallow copy of this population.

        The contained `PromptCandidate` objects are not copied; the new
        `Population` references the same objects.

        Returns:
            A new `Population` instance with the same prompts, generation,
            and max_population_size.
        """
        return Population(
            prompts=list(self._prompts.values()),
            generation=self.generation,
            max_population_size=self.max_population_size,
        )

    def clear(self) -> None:
        """Remove all prompts from the population."""
        self._prompts.clear()

    def is_full(self) -> bool:
        """Check whether the population has reached its maximum size.

        Returns:
            True if the number of prompts equals `max_population_size`.
        """
        return len(self._prompts) >= self.max_population_size

    def remaining_capacity(self) -> int:
        """Return how many more prompts can be added before reaching capacity.

        Returns:
            `max_population_size` minus the current number of prompts.
        """
        return self.max_population_size - len(self._prompts)

    def summary(self) -> dict[str, object]:
        """Return a logging-friendly summary of this population.

        Returns:
            A dictionary containing the generation number, current
            population size, maximum population size, and whether every
            prompt has been evaluated.
        """
        evaluated = all(
            prompt.fitness is not None for prompt in self._prompts.values()
        )

        return {
            "generation": self.generation,
            "population_size": len(self._prompts),
            "max_population_size": self.max_population_size,
            "evaluated": evaluated,
        }

    def __len__(self) -> int:
        """Return the number of prompts in the population.

        Returns:
            The number of prompts currently stored.
        """
        return len(self._prompts)

    def __iter__(self) -> Iterator[PromptCandidate]:
        """Iterate over the prompts in the population.

        Returns:
            An iterator over the `PromptCandidate` objects, in insertion
            order.
        """
        return iter(self._prompts.values())

    def __contains__(self, prompt_id: str) -> bool:
        """Check whether a prompt with the given ID is in the population.

        Args:
            prompt_id: The prompt ID to check.

        Returns:
            True if the prompt is present, False otherwise.
        """
        return prompt_id in self._prompts

    def __getitem__(self, key: int | str) -> PromptCandidate:
        """Retrieve a prompt by index or by ID.

        Args:
            key: Either an integer positional index or a string prompt ID.

        Returns:
            The matching `PromptCandidate`.

        Raises:
            IndexError: If `key` is an integer and out of range.
            KeyError: If `key` is a string and no matching prompt exists.
        """
        if isinstance(key, int):
            return self.get_by_index(key)

        return self.get(key)

    def __str__(self) -> str:
        """Return a concise, human-readable summary of this population.

        Returns:
            A string of the form:
            'Population(generation=4, size=20, evaluated=True)'.
        """
        evaluated = all(
            prompt.fitness is not None for prompt in self._prompts.values()
        )

        return (
            f"Population(\n"
            f"    generation={self.generation},\n"
            f"    size={len(self._prompts)},\n"
            f"    evaluated={evaluated}\n"
            f")"
        )
    
    