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

import json
import os
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
        max_parallel_workers: int | None = None,
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
            max_parallel_workers: Optional limit on concurrent hospital
                evaluations. If None (default), uses one worker per hospital.

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

        if max_parallel_workers is not None and max_parallel_workers < 1:
            raise ValueError("max_parallel_workers must be >= 1")

        self._hospitals: list[HospitalClient] = hospitals
        self._aggregator: Aggregator = aggregator
        self._evaluation_cache: EvaluationCache = evaluation_cache
        self._lineage_tracker: LineageTracker = lineage_tracker
        self._model_name: str = model_name
        self._dataset_name: str = dataset_name
        self._evaluation_version: str = evaluation_version
        self._max_parallel_workers: int | None = max_parallel_workers
        self._diagnostics_path = Path("results/hfpo_run/parent_child_diagnostics.jsonl")
        self._diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

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
        print(f"Evaluating {len(population)} prompts...")
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
        print(
            f"Federated evaluation: "
            f"{len(population)} prompts "
            f"across {len(self.hospitals)} hospitals."
        )
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
        print(
            f"Cache hits: {len(population) - len(uncached_prompts)} | "
            f"Need evaluation: {len(uncached_prompts)}"
        )
        if uncached_prompts:
            per_hospital_records = self.broadcast(uncached_prompts)
            print("Broadcast complete.")

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
        self._log_parent_child_diagnostics(population)
        print("Federated evaluation complete.")
        return population

    def _log_parent_child_diagnostics(
        self,
        population: list[PromptCandidate],
    ) -> None:
        """Log parent fitness to child fitness deltas and prediction agreement."""
        for child in population:
            if not child.parent_ids or child.fitness is None:
                continue

            child_predictions = child.metadata.get("evaluation_predictions", {})
            for parent_id in child.parent_ids:
                if not self._lineage_tracker.exists(parent_id):
                    continue

                parent = self._lineage_tracker.get(parent_id)
                if parent.fitness is None:
                    continue

                parent_predictions = parent.metadata.get("evaluation_predictions", {})
                agreement = self._prediction_agreement(
                    parent_predictions,
                    child_predictions,
                )

                record = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "child_id": child.id,
                    "child_generation": child.generation,
                    "child_origin": child.origin,
                    "parent_id": parent.id,
                    "parent_generation": parent.generation,
                    "parent_origin": parent.origin,
                    "parent_fitness": parent.fitness.as_dict(),
                    "child_fitness": child.fitness.as_dict(),
                    "fitness_delta": {
                        "average": child.fitness.average() - parent.fitness.average(),
                        "minimum": child.fitness.minimum() - parent.fitness.minimum(),
                        "maximum": child.fitness.maximum() - parent.fitness.maximum(),
                    },
                    "prediction_agreement": agreement,
                }

                with self._diagnostics_path.open("a", encoding="utf-8") as file:
                    json.dump(record, file, ensure_ascii=False)
                    file.write("\n")

    def _prediction_agreement(
        self,
        parent_predictions: object,
        child_predictions: object,
    ) -> dict[str, object]:
        """Compute exact prediction agreement by hospital and overall."""
        if not isinstance(parent_predictions, dict) or not isinstance(
            child_predictions,
            dict,
        ):
            return {"available": False, "reason": "missing_predictions"}

        per_hospital: dict[str, object] = {}
        total_matches = 0
        total_compared = 0

        for hospital_id in sorted(set(parent_predictions) & set(child_predictions)):
            parent_values = parent_predictions[hospital_id]
            child_values = child_predictions[hospital_id]
            if not isinstance(parent_values, list) or not isinstance(child_values, list):
                continue

            compared = min(len(parent_values), len(child_values))
            if compared == 0:
                continue

            matches = sum(
                1
                for parent_value, child_value in zip(
                    parent_values[:compared],
                    child_values[:compared],
                )
                if parent_value == child_value
            )
            per_hospital[hospital_id] = {
                "agreement": matches / compared,
                "matches": matches,
                "compared": compared,
            }
            total_matches += matches
            total_compared += compared

        if total_compared == 0:
            return {"available": False, "reason": "no_overlap"}

        return {
            "available": True,
            "overall": total_matches / total_compared,
            "matches": total_matches,
            "compared": total_compared,
            "by_hospital": per_hospital,
        }


    def broadcast(
        self, population: list[PromptCandidate]
    ) -> list[list[EvaluationRecord]]:
        """Broadcast a population of prompts to every hospital for evaluation.

        Each `HospitalClient` independently evaluates the entire supplied
        population against its own private dataset. The calls are executed
        concurrently with a ThreadPoolExecutor; results are returned in the
        same order as `self.hospitals` so that downstream logic is unchanged.

        Concurrent inference requests can overlap because PyTorch releases
        the GIL during most tensor operations. Actual speedup depends on
        GPU utilization, memory availability, and the behavior of
        `model.generate()`.
        """
        if self._max_parallel_workers is None:
            num_workers = len(self._hospitals)
        else:
            num_workers = min(self._max_parallel_workers, len(self._hospitals))

        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(hospital.evaluate_population, population)
                for hospital in self._hospitals
            ]
            responses = []
            for hospital, future in zip(self._hospitals, futures):
                try:
                    responses.append(future.result())
                except Exception as e:
                    print(f"Hospital {hospital.hospital_id} failed: {e}")
                    raise

        elapsed = time.perf_counter() - start
        print(
            f"Broadcast completed in {elapsed:.2f}s | "
            f"workers={num_workers} | hospitals={len(self._hospitals)}"
        )

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
