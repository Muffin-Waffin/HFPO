"""Federated evaluation orchestration for HFPO (Heterogeneous Federated
Prompt Optimization).

This module defines :class:`FederatedServer`, the orchestration layer of
HFPO. Hospitals never communicate with each other, never share datasets,
and never share patient information; the `FederatedServer` is the sole
coordinator that broadcasts candidate prompts to every hospital, collects
their independent `EvaluationRecord` results, and combines them via the
`Aggregator` into `FitnessVector` objects. It also consults and maintains
the `EvaluationCache` to avoid redundant federated evaluation rounds for
prompts that have already been scored under the same model, dataset, and
evaluation pipeline version, and registers every prompt it processes with
the `LineageTracker` for reproducibility and experiment logging.

The `FederatedServer` performs no genetic algorithm operations. It does
not mutate, crossover, select, rank, or generate prompts; those
responsibilities belong to later milestones. Its sole responsibility is
coordinating federated evaluation across an isolated set of hospitals,
entirely in-process, with no networking, threading, or external
federated learning framework involved.
"""

from __future__ import annotations

from src.core.evaluation_cache import EvaluationCache
from src.core.evaluation_record import EvaluationRecord
from src.core.lineage_tracker import LineageTracker
from src.core.prompt_candidate import PromptCandidate
from src.federated.aggregator import Aggregator
from src.federated.hospital_client import HospitalClient


class FederatedServer:
    """Coordinates federated evaluation of prompts across isolated hospitals.

    `FederatedServer` is the orchestration layer of HFPO. It never accesses
    raw patient data and never lets hospitals communicate with one
    another; it only broadcasts prompt text to each `HospitalClient`,
    collects the `EvaluationRecord` objects they return, and uses the
    `Aggregator` to combine those independent results into `FitnessVector`
    objects attached to each prompt. Before broadcasting, it consults the
    `EvaluationCache` to avoid re-evaluating prompts that have already been
    scored under the current model, dataset, and evaluation pipeline
    version. After evaluation, it registers every prompt with the
    `LineageTracker` for reproducibility and experiment logging.

    `FederatedServer` performs no genetic algorithm logic of any kind: no
    mutation, crossover, selection, ranking, or prompt generation. It is
    purely a federated evaluation coordinator.

    Attributes:
        hospitals: The list of `HospitalClient` instances participating in
            the federation.
        aggregator: The `Aggregator` used to combine `EvaluationRecord`
            objects into `FitnessVector` objects.
        evaluation_cache: The `EvaluationCache` used to avoid redundant
            federated evaluation rounds.
        lineage_tracker: The `LineageTracker` used to record the
            evolutionary ancestry of every prompt processed.
        model_name: Name of the model used for evaluation, used as part of
            the cache key.
        dataset_name: Name of the dataset used for evaluation, used as
            part of the cache key.
        evaluation_version: Version identifier of the evaluation pipeline,
            used as part of the cache key.

    Raises:
        ValueError: If `hospitals` is empty, if `aggregator`,
            `evaluation_cache`, or `lineage_tracker` is `None`, or if
            `model_name`, `dataset_name`, or `evaluation_version` is empty
            or whitespace.
    """

    def __init__(
        self,
        hospitals: list[HospitalClient],
        aggregator: Aggregator,
        evaluation_cache: EvaluationCache,
        lineage_tracker: LineageTracker,
        model_name: str,
        dataset_name: str,
        evaluation_version: str,
    ) -> None:
        """Initialize a FederatedServer with its hospitals and dependencies.

        Args:
            hospitals: The list of `HospitalClient` instances participating
                in the federation.
            aggregator: The `Aggregator` used to combine `EvaluationRecord`
                objects into `FitnessVector` objects.
            evaluation_cache: The `EvaluationCache` used to avoid redundant
                federated evaluation rounds.
            lineage_tracker: The `LineageTracker` used to record the
                evolutionary ancestry of every prompt processed.
            model_name: Name of the model used for evaluation.
            dataset_name: Name of the dataset used for evaluation.
            evaluation_version: Version identifier of the evaluation
                pipeline.

        Raises:
            ValueError: If `hospitals` is empty, if `aggregator`,
                `evaluation_cache`, or `lineage_tracker` is `None`, or if
                `model_name`, `dataset_name`, or `evaluation_version` is
                empty or whitespace.
        """
        if not hospitals:
            raise ValueError("FederatedServer hospitals must not be empty.")

        if aggregator is None:
            raise ValueError("FederatedServer aggregator must not be None.")

        if evaluation_cache is None:
            raise ValueError("FederatedServer evaluation_cache must not be None.")

        if lineage_tracker is None:
            raise ValueError("FederatedServer lineage_tracker must not be None.")

        if not model_name.strip():
            raise ValueError("FederatedServer model_name must not be empty.")

        if not dataset_name.strip():
            raise ValueError("FederatedServer dataset_name must not be empty.")

        if not evaluation_version.strip():
            raise ValueError(
                "FederatedServer evaluation_version must not be empty."
            )

        self._hospitals: list[HospitalClient] = hospitals
        self._aggregator: Aggregator = aggregator
        self._evaluation_cache: EvaluationCache = evaluation_cache
        self._lineage_tracker: LineageTracker = lineage_tracker
        self._model_name: str = model_name
        self._dataset_name: str = dataset_name
        self._evaluation_version: str = evaluation_version

    @property
    def hospitals(self) -> list[HospitalClient]:
        """Return the list of HospitalClient instances in this federation.

        Returns:
            The list of `HospitalClient` instances participating in the
            federation. This avoids requiring callers to access the
            server's private attributes directly.
        """
        return self._hospitals

    def evaluate_population(
        self, population: list[PromptCandidate]
    ) -> list[PromptCandidate]:
        """Evaluate a population of prompts via federated evaluation.

        For every prompt, first consults the `EvaluationCache` using the
        prompt's text together with this server's model name, dataset
        name, and evaluation version. Cached prompts have their cached
        `FitnessVector` attached immediately. Prompts that are not cached
        are broadcast to every hospital for fresh evaluation, and the
        resulting `EvaluationRecord` objects are aggregated into
        `FitnessVector` objects via the `Aggregator`. Newly computed
        `FitnessVector` objects are stored in the `EvaluationCache`. Every
        prompt in the population, whether cached or newly evaluated, is
        then registered with the `LineageTracker`, skipping any prompt
        already registered.

        Args:
            population: The list of `PromptCandidate` objects to evaluate.

        Returns:
            The same `population` list, with every prompt's `fitness`
            attribute populated.

        Raises:
            ValueError: If `population` is empty.
            RuntimeError: If broadcasting returns an unexpected number of
                hospital responses, if the total number of collected
                `EvaluationRecord` objects does not match the expected
                count, or if any prompt unexpectedly has no `FitnessVector`
                after aggregation.
        """
        if not population:
            raise ValueError(
                "FederatedServer.evaluate_population received an empty "
                "population."
            )

        uncached_prompts: list[PromptCandidate] = []

        for prompt in population:
            cached_fitness = self._evaluation_cache.get(
                prompt.text,
                self._model_name,
                self._dataset_name,
                self._evaluation_version,
            )

            if cached_fitness is not None:
                prompt.fitness = cached_fitness
            else:
                uncached_prompts.append(prompt)

        if uncached_prompts:
            per_hospital_records = self.broadcast(uncached_prompts)

            if len(per_hospital_records) != len(self._hospitals):
                raise RuntimeError(
                    "Broadcast returned an unexpected number of hospital "
                    f"responses: expected {len(self._hospitals)}, got "
                    f"{len(per_hospital_records)}."
                )

            flattened_records: list[EvaluationRecord] = [
                record
                for hospital_records in per_hospital_records
                for record in hospital_records
            ]

            expected_record_count = len(self._hospitals) * len(uncached_prompts)
            if len(flattened_records) != expected_record_count:
                raise RuntimeError(
                    f"Expected {expected_record_count} EvaluationRecords "
                    f"but received {len(flattened_records)}."
                )

            self._aggregator.aggregate(uncached_prompts, flattened_records)

            for prompt in uncached_prompts:
                if prompt.fitness is None:
                    raise RuntimeError(
                        f"Prompt '{prompt.id}' has no FitnessVector after "
                        f"aggregation."
                    )

                self._evaluation_cache.set(
                    prompt.text,
                    self._model_name,
                    self._dataset_name,
                    self._evaluation_version,
                    prompt.fitness,
                )

        self.register_population(population)

        return population

    def broadcast(
        self, population: list[PromptCandidate]
    ) -> list[list[EvaluationRecord]]:
        """Broadcast a population of prompts to every hospital for evaluation.

        Each `HospitalClient` independently evaluates the entire supplied
        population against its own private dataset. No networking,
        threading, or external federated learning framework is involved;
        every hospital is called in-process, sequentially.

        Args:
            population: The list of `PromptCandidate` objects to broadcast.

        Returns:
            A list containing one list of `EvaluationRecord` objects per
            hospital, in the same order as `self.hospitals`.
        """
        responses: list[list[EvaluationRecord]] = []

        for hospital in self._hospitals:
            responses.append(hospital.evaluate_population(population))

        return responses

    def register_population(self, population: list[PromptCandidate]) -> None:
        """Register every prompt in the population with the LineageTracker.

        Prompts that have already been registered are skipped.

        Args:
            population: The list of `PromptCandidate` objects to register.
        """
        for prompt in population:
            if not self._lineage_tracker.exists(prompt.id):
                self._lineage_tracker.register(prompt)

    def hospital_summary(self) -> list[dict[str, object]]:
        """Return a logging-friendly summary of every hospital.

        Returns:
            A list containing the result of `summary()` for every
            `HospitalClient` in the federation.
        """
        return [hospital.summary() for hospital in self._hospitals]

    def cache_statistics(self) -> dict[str, object]:
        """Return statistics about the evaluation cache.

        Returns:
            A dictionary containing the number of entries currently
            stored in the `EvaluationCache`, along with the current model
            and dataset name, to make experiment logs easier to interpret.
        """
        return {
            "entries": len(self._evaluation_cache),
            "model": self._model_name,
            "dataset": self._dataset_name,
        }

    def summary(self) -> dict[str, object]:
        """Return a comprehensive summary of this server's current state.

        Returns:
            A dictionary containing the number of hospitals, the number of
            cache entries, the number of registered prompts, and the
            current model, dataset, and evaluation version.
        """
        return {
            "hospitals": len(self._hospitals),
            "cache_entries": len(self._evaluation_cache),
            "registered_prompts": len(self._lineage_tracker),
            "model": self._model_name,
            "dataset": self._dataset_name,
            "evaluation_version": self._evaluation_version,
        }

    def __str__(self) -> str:
        """Return a concise, human-readable summary of this server.

        Returns:
            A string of the form:
            'FederatedServer(hospitals=3, model="qwen3-8b",
            cache_entries=25)'.
        """
        return (
            "FederatedServer(\n"
            f"    hospitals={len(self._hospitals)},\n"
            f'    model="{self._model_name}",\n'
            f"    cache_entries={len(self._evaluation_cache)}\n"
            ")"
        )