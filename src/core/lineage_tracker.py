"""Evolutionary lineage tracking for HFPO (Heterogeneous Federated Prompt
Optimization).

This module defines :class:`LineageTracker`, which records the complete
evolutionary history of the prompt population produced during HFPO. As the
genetic algorithm runs, every `PromptCandidate` that is created, whether a
seed, mutation, crossover, or elite survivor, is registered with the
tracker along with its direct parent IDs. The tracker maintains the
parent-child relationships needed to reconstruct full ancestry chains,
compute evolutionary depth, and export the evolutionary tree for
visualization and analysis. This class exists for reproducibility,
experiment logging, and generating paper figures; it performs no genetic
algorithm operations such as selection, mutation, or crossover.
"""

from __future__ import annotations

from src.core.prompt_candidate import PromptCandidate


class LineageTracker:
    """Tracks the complete evolutionary ancestry of the prompt population.

    `LineageTracker` is owned by the Federated Server and records every
    `PromptCandidate` produced during HFPO evolution, along with its direct
    parent relationships. From this information it can reconstruct the
    full ancestry chain of any prompt back to its seed ancestors, compute
    the evolutionary depth of any prompt, identify a prompt's direct
    children, and export the entire evolutionary tree as a JSON-serializable
    structure for visualization and analysis. This class is used purely for
    record-keeping and reproducibility; it does not perform selection,
    mutation, crossover, or any other genetic algorithm operation.
    """

    def __init__(self) -> None:
        """Initialize an empty lineage tracker."""
        self._nodes: dict[str, PromptCandidate] = {}
        self._parents: dict[str, list[str]] = {}

    @staticmethod
    def _validate_prompt_id(prompt_id: str) -> None:
        """Validate that a prompt ID argument is non-empty.

        Args:
            prompt_id: The prompt ID to validate.

        Raises:
            ValueError: If `prompt_id` is empty or whitespace.
        """
        if not prompt_id.strip():
            raise ValueError("LineageTracker prompt_id must not be empty.")

    def register(self, prompt: PromptCandidate) -> None:
        """Register a PromptCandidate and its direct parent relationships.

        Args:
            prompt: The `PromptCandidate` to register.

        Raises:
            ValueError: If a prompt with the same ID has already been
                registered, or if any declared parent ID does not exist in
                the tracker and the prompt's origin is not "seed".
        """
        if prompt.id in self._nodes:
            raise ValueError(
                f"PromptCandidate with id '{prompt.id}' is already "
                f"registered in the LineageTracker."
            )

        if prompt.origin != "seed":
            for parent_id in prompt.parent_ids:
                if parent_id not in self._nodes:
                    raise ValueError(
                        f"Cannot register PromptCandidate '{prompt.id}': "
                        f"parent '{parent_id}' has not been registered."
                    )

        self._nodes[prompt.id] = prompt
        self._parents[prompt.id] = list(prompt.parent_ids)

    def exists(self, prompt_id: str) -> bool:
        """Check whether a prompt has already been registered.

        Args:
            prompt_id: The prompt ID to check.

        Returns:
            True if the prompt has been registered, False otherwise.

        Raises:
            ValueError: If `prompt_id` is empty or whitespace.
        """
        self._validate_prompt_id(prompt_id)
        return prompt_id in self._nodes

    def get(self, prompt_id: str) -> PromptCandidate:
        """Retrieve the registered PromptCandidate for a given ID.

        Args:
            prompt_id: The prompt ID to look up.

        Returns:
            The registered `PromptCandidate`.

        Raises:
            ValueError: If `prompt_id` is empty or whitespace.
            KeyError: If `prompt_id` has not been registered.
        """
        self._validate_prompt_id(prompt_id)
        if prompt_id not in self._nodes:
            raise KeyError(
                f"PromptCandidate with id '{prompt_id}' is not registered "
                f"in the LineageTracker."
            )
        return self._nodes[prompt_id]

    def parents(self, prompt_id: str) -> list[str]:
        """Return the direct parent IDs of a prompt.

        Args:
            prompt_id: The prompt ID to look up.

        Returns:
            A copy of the list of direct parent IDs.

        Raises:
            ValueError: If `prompt_id` is empty or whitespace.
            KeyError: If `prompt_id` has not been registered.
        """
        self._validate_prompt_id(prompt_id)
        if prompt_id not in self._parents:
            raise KeyError(
                f"PromptCandidate with id '{prompt_id}' is not registered "
                f"in the LineageTracker."
            )
        return list(self._parents[prompt_id])

    def ancestry(self, prompt_id: str) -> list[str]:
        """Return the complete ancestry chain of a prompt.

        The ancestry chain is stored directly inside each PromptCandidate at the
        time the prompt is created by the evolution engine. LineageTracker does
        not recompute ancestry recursively; it simply returns the canonical
        ancestry recorded for that prompt.

        Args:
            prompt_id: The prompt ID whose ancestry should be returned.

        Returns:
            A copy of the ancestry chain ordered from the oldest ancestor to the
            newest ancestor. The prompt itself is not included. Seed prompts
            return an empty list.

        Raises:
            ValueError:
                If ``prompt_id`` is empty or whitespace.

            KeyError:
                If the prompt has not been registered.
        """
        self._validate_prompt_id(prompt_id)

        if prompt_id not in self._nodes:
            raise KeyError(
                f"PromptCandidate with id '{prompt_id}' is not registered "
                f"in the LineageTracker."
            )

        return list(self._nodes[prompt_id].ancestry_ids)

    def depth(self, prompt_id: str) -> int:
        """Compute the evolutionary depth of a prompt.

        Seed prompts have depth 0. Mutation and elite prompts have depth equal
        to their parent's depth plus one. Crossover prompts have depth equal to
        the maximum depth of their parents plus one.

        Args:
            prompt_id: The prompt ID whose depth should be computed.

        Returns:
            The evolutionary depth.

        Raises:
            ValueError:
                If ``prompt_id`` is empty or whitespace.

            KeyError:
                If the prompt has not been registered.
        """
        self._validate_prompt_id(prompt_id)

        if prompt_id not in self._nodes:
            raise KeyError(
                f"PromptCandidate with id '{prompt_id}' is not registered "
                f"in the LineageTracker."
            )

        parents = self._parents[prompt_id]

        if not parents:
            return 0

        return max(self.depth(parent) for parent in parents) + 1
    
    def children(self, prompt_id: str) -> list[str]:
        """Return the direct children of a prompt.

        Args:
            prompt_id: The prompt ID whose children should be found.

        Returns:
            A list of prompt IDs whose direct parent list contains
            `prompt_id`.

        Raises:
            ValueError: If `prompt_id` is empty or whitespace.
            KeyError: If `prompt_id` has not been registered.
        """
        self._validate_prompt_id(prompt_id)
        if prompt_id not in self._nodes:
            raise KeyError(
                f"PromptCandidate with id '{prompt_id}' is not registered "
                f"in the LineageTracker."
            )

        return [
            candidate_id
            for candidate_id, parent_ids in self._parents.items()
            if prompt_id in parent_ids
        ]

    def export_tree(self) -> dict[str, object]:
        """Export the complete evolutionary tree as a serializable structure.

        Returns:
            A dictionary with two keys: "nodes", a list of dictionaries
            each containing "id", "generation", and "origin" for every
            registered prompt; and "edges", a list of dictionaries each
            containing "parent" and "child" for every direct parent-child
            relationship in the tracker.
        """
        nodes = [
            {
                "id": prompt.id,
                "generation": prompt.generation,
                "origin": prompt.origin,
            }
            for prompt in self._nodes.values()
        ]

        edges = [
            {"parent": parent_id, "child": child_id}
            for child_id, parent_ids in self._parents.items()
            for parent_id in parent_ids
        ]

        return {"nodes": nodes, "edges": edges}

    def clear(self) -> None:
        """Remove all stored lineage information."""
        self._nodes.clear()
        self._parents.clear()

    def __len__(self) -> int:
        """Return the number of registered PromptCandidates.

        Returns:
            The number of registered prompts.
        """
        return len(self._nodes)

    def __str__(self) -> str:
        """Return a concise, human-readable summary of this tracker.

        Returns:
            A string of the form 'LineageTracker(nodes=128)'.
        """
        return f"LineageTracker(nodes={len(self._nodes)})"