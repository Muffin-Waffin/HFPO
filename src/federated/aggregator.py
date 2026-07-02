"""Cross-hospital aggregation for HFPO (Heterogeneous Federated Prompt
Optimization).

This module defines :class:`Aggregator`, the component responsible for
converting the independent `EvaluationRecord` objects returned by each
hospital into a single `FitnessVector` per prompt and attaching that
`FitnessVector` to the corresponding `PromptCandidate`. Hospitals never
exchange information with one another: each hospital evaluates a prompt
in complete isolation, using only its own local data, and reports back a
single `EvaluationRecord`. The `Aggregator` is the first and only point in
the system where these independent, per-hospital results are combined into
a distributed view of a prompt's fitness. It therefore represents the
logical federation point of HFPO. The `Aggregator` performs no genetic
algorithm logic: it does not rank prompts, select survivors, or generate
new candidates. Its sole responsibility is constructing `FitnessVector`
objects from `EvaluationRecord` objects.
"""

from __future__ import annotations

from src.core.evaluation_record import EvaluationRecord
from src.core.fitness_vector import FitnessVector
from src.core.prompt_candidate import PromptCandidate


class Aggregator:
    """Combines per-hospital EvaluationRecords into prompt-level FitnessVectors.

    `Aggregator` is the logical federation point of HFPO. Hospitals never
    communicate with each other; each one independently evaluates a prompt
    against its own local data and produces an `EvaluationRecord`. The
    `Aggregator` collects the `EvaluationRecord` objects produced by all
    hospitals for a given evaluation round, groups them by prompt, and
    constructs a `FitnessVector` for each prompt representing its
    distributed fitness across the federation. This class performs no
    ranking, selection, mutation, crossover, or prompt generation; it only
    builds `FitnessVector` objects from `EvaluationRecord` objects and
    attaches them to the matching `PromptCandidate` instances.
    """

    def aggregate(
        self,
        population: list[PromptCandidate],
        evaluations: list[EvaluationRecord],
    ) -> list[PromptCandidate]:
        """Attach a FitnessVector to each prompt from its evaluation records.

        For every `PromptCandidate` in `population`, this method collects
        all `EvaluationRecord` objects with a matching `prompt_id`,
        constructs a `FitnessVector` mapping hospital ID to score, and
        assigns it to that prompt's `fitness` attribute.

        Args:
            population: The list of `PromptCandidate` objects to attach
                fitness to.
            evaluations: The list of `EvaluationRecord` objects returned
                by all hospitals for this evaluation round.

        Returns:
            The same `population` list, with each prompt's `fitness`
            attribute populated.

        Raises:
            ValueError: If `population` is empty, if `evaluations` is
                empty, if any prompt in `population` has zero matching
                evaluations, if duplicate hospital evaluations exist for
                the same prompt, if the evaluations span more than one
                generation, or if any evaluation's `prompt_id` does not
                correspond to a prompt in `population`.
        """
        if not population:
            raise ValueError("Aggregator.aggregate received an empty population.")

        if not evaluations:
            raise ValueError("Aggregator.aggregate received an empty evaluations list.")

        self.validate_generation(evaluations)

        population_ids = {prompt.id for prompt in population}
        grouped = self.group_by_prompt(evaluations)

        for prompt_id in grouped:
            if prompt_id not in population_ids:
                raise ValueError(
                    f"EvaluationRecord references prompt_id '{prompt_id}', "
                    f"which does not exist in the supplied population."
                )

        for prompt in population:
            records = grouped.get(prompt.id, [])

            if not records:
                raise ValueError(
                    f"PromptCandidate '{prompt.id}' has zero evaluations."
                )

            scores: dict[str, float] = {}
            predictions_by_hospital: dict[str, object] = {}
            for record in records:
                if record.hospital_id in scores:
                    raise ValueError(
                        f"Duplicate evaluation for prompt '{prompt.id}' "
                        f"from hospital '{record.hospital_id}'."
                    )
                scores[record.hospital_id] = record.score
                if "predictions" in record.metadata:
                    predictions_by_hospital[record.hospital_id] = list(
                        record.metadata["predictions"]
                    )

            prompt.fitness = FitnessVector(scores=scores)
            if predictions_by_hospital:
                prompt.metadata["evaluation_predictions"] = predictions_by_hospital

        return population

    def group_by_prompt(
        self, evaluations: list[EvaluationRecord]
    ) -> dict[str, list[EvaluationRecord]]:
        """Group evaluation records by their prompt ID.

        Args:
            evaluations: The list of `EvaluationRecord` objects to group.

        Returns:
            A dictionary mapping each prompt ID to the list of
            `EvaluationRecord` objects associated with it, preserving the
            relative order in which records were encountered.
        """
        grouped: dict[str, list[EvaluationRecord]] = {}
        for record in evaluations:
            grouped.setdefault(record.prompt_id, []).append(record)
        return grouped

    def validate_generation(self, evaluations: list[EvaluationRecord]) -> None:
        """Ensure all evaluation records belong to exactly one generation.

        Args:
            evaluations: The list of `EvaluationRecord` objects to check.

        Raises:
            ValueError: If the records span more than one distinct
                generation.
        """
        generations = {record.generation for record in evaluations}
        if len(generations) > 1:
            raise ValueError(
                f"EvaluationRecords span multiple generations: "
                f"{sorted(generations)}. All evaluations in a single "
                f"aggregation batch must belong to the same generation."
            )

    def hospital_ids(self, evaluations: list[EvaluationRecord]) -> list[str]:
        """Identify the unique hospitals represented in an evaluation batch.

        Args:
            evaluations: The list of `EvaluationRecord` objects to inspect.

        Returns:
            A sorted list of unique hospital IDs present in `evaluations`,
            useful for logging.
        """
        return sorted({record.hospital_id for record in evaluations})

    def __str__(self) -> str:
        """Return a concise, human-readable representation of this aggregator.

        Returns:
            The string 'Aggregator()'.
        """
        return "Aggregator()"
