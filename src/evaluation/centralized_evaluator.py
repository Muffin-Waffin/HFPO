"""Centralized evaluation backend for HFPO ablation experiments.

This module defines :class:`CentralizedEvaluator`, a drop-in alternative to
:class:`src.federated.federated_server.FederatedServer` that evaluates prompt
candidates against a POOLED combination of all hospital datasets rather than
federating evaluation across isolated hospitals.

KEY DESIGN DIFFERENCE FROM FEDERATED EVALUATION:
================================================
Federated (macro-average): Each hospital evaluates on its own 100-sample
local subset independently. Fitness = mean(hospital_1_score, hospital_2_score,
hospital_3_score). This is an UNWEIGHTED mean of per-hospital accuracies.

Centralized (micro-average): All three 100-sample subsets are pooled into a
single 300-sample combined set. Fitness = total_correct / 300. This is a
MICRO-average over the pooled data, which weights each sample equally
regardless of which hospital's subset it came from.

These are MATHEMATICALLY DIFFERENT aggregation strategies. The centralized
micro-average gives equal weight to every sample; the federated macro-average
gives equal weight to every hospital. This difference is INTENTIONAL and is
the actual thing being tested in the centralized-vs-federated ablation
(Experiment 4). Do NOT "fix" this to make them match — the divergence is the
experimental variable.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, List

from src.core.fitness_vector import FitnessVector
from src.core.prompt_candidate import PromptCandidate
from src.data.loader import load_dataset
from src.evaluation.qwen_evaluator import QwenEvaluator
from src.evaluation.evaluator import Evaluator


class CentralizedEvaluator:
    """Evaluates prompts against a POOLED combination of all hospital datasets.

    Unlike :class:`FederatedServer`, which broadcasts candidates to isolated
    hospitals and macro-averages their independent scores, this evaluator
    concatenates the three evaluation subsets (medqa, pubmedqa, medmcqa —
    100 samples each, same RANDOM_SEED=51 as the federated run) into one
    300-sample combined dataset and scores each candidate once against it.

    The resulting single scalar accuracy (micro-average over 300 samples) is
    wrapped in a :class:`FitnessVector` with key "centralized" so that
    ``.average()`` returns the same value and existing evolution engine
    selection/elitism logic remains unchanged.

    Attributes:
        dataset: The pooled 300-sample combined evaluation dataset.
        evaluator: The underlying :class:`Evaluator` used to score samples.
        model_name: Model name used for cache keys (mirrors FederatedServer).
        dataset_name: Dataset label for cache keys (pooled dataset identifier).
        evaluation_version: Evaluation pipeline version for cache keys.
        evaluation_cache: Optional cache to avoid re-evaluating identical
            prompts. Mirrors FederatedServer's cache interface.
        lineage_tracker: Optional tracker to register evaluated prompts.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        dataset_name: str,
        evaluation_version: str,
        evaluation_subset_size: int = 100,
        random_seed: int = 51,
        evaluation_cache: Any = None,
        lineage_tracker: Any = None,
    ) -> None:
        """Initialize the centralized evaluator with pooled datasets.

        Args:
            model: The loaded language model for evaluation.
            tokenizer: The tokenizer paired with ``model``.
            model_name: Model identifier for cache keys.
            dataset_name: Dataset identifier for cache keys (should describe
                the pooled combination, e.g., "medqa+pubmedqa+medmcqa").
            evaluation_version: Evaluation pipeline version for cache keys.
            evaluation_subset_size: Number of samples per dataset to pool
                (default 100, matching federated per-hospital subset).
            random_seed: Random seed for dataset sampling (default 51,
                matching federated per-hospital sampling).
            evaluation_cache: Optional cache with ``get``/``set`` methods
                matching :class:`EvaluationCache` interface.
            lineage_tracker: Optional tracker with ``register``/``exists``
                methods matching :class:`LineageTracker` interface.
        """
        if model is None:
            raise ValueError("model must not be None.")
        if tokenizer is None:
            raise ValueError("tokenizer must not be None.")
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")
        if not dataset_name.strip():
            raise ValueError("dataset_name must not be empty.")
        if not evaluation_version.strip():
            raise ValueError("evaluation_version must not be empty.")
        if evaluation_subset_size <= 0:
            raise ValueError("evaluation_subset_size must be positive.")

        self._model = model
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._dataset_name = dataset_name
        self._evaluation_version = evaluation_version
        self._evaluation_subset_size = evaluation_subset_size
        self._random_seed = random_seed
        self._evaluation_cache = evaluation_cache
        self._lineage_tracker = lineage_tracker

        self._evaluator: Evaluator = QwenEvaluator(model=model, tokenizer=tokenizer)
        self._dataset: List[Any] = self._build_pooled_dataset()

    def _build_pooled_dataset(self) -> List[Any]:
        """Load and pool all three hospital datasets with consistent sampling.

        Uses the SAME per-dataset subset size (100) and SAME random seed (51)
        as the federated run, so the underlying data is IDENTICAL — only the
        aggregation method differs (pooled micro-average vs. federated
        macro-average).

        Returns:
            List of 300 samples: 100 from medqa + 100 from pubmedqa +
            100 from medmcqa, all sampled with RANDOM_SEED=51.
        """
        dataset_names = ("medqa", "pubmedqa", "medmcqa")
        pooled: List[Any] = []

        for name in dataset_names:
            dataset = load_dataset(name, split="train")
            if len(dataset) > self._evaluation_subset_size:
                rng = random.Random(self._random_seed)
                indices = rng.sample(
                    range(len(dataset)), self._evaluation_subset_size
                )
                if hasattr(dataset, "select"):
                    subset = dataset.select(indices)
                else:
                    subset = [dataset[i] for i in indices]
            else:
                subset = dataset
            pooled.extend(subset)

        return pooled

    def evaluate_population(
        self, population: List[PromptCandidate]
    ) -> List[PromptCandidate]:
        """Evaluate a population of prompts against the pooled dataset.

        For each candidate, checks cache first. If not cached, evaluates
        once on the full 300-sample pooled set, computes micro-average
        accuracy, wraps in a FitnessVector with key "centralized", caches
        the result, and registers with lineage tracker if provided.

        Args:
            population: List of PromptCandidate objects to evaluate.

        Returns:
            The same population list with each candidate's ``fitness``
            attribute populated (as a FitnessVector with a single
            "centralized" key).
        """
        if not population:
            raise ValueError("CentralizedEvaluator received an empty population.")

        uncached: List[PromptCandidate] = []
        for prompt in population:
            cached_fitness = None
            if self._evaluation_cache is not None:
                cached_fitness = self._evaluation_cache.get(
                    prompt.text,
                    self._model_name,
                    self._dataset_name,
                    self._evaluation_version,
                )
            if cached_fitness is not None:
                prompt.fitness = cached_fitness
            else:
                uncached.append(prompt)

        print(
            f"Centralized evaluation: "
            f"{len(population)} prompts | "
            f"Cache hits: {len(population) - len(uncached)} | "
            f"Need evaluation: {len(uncached)} | "
            f"Pooled dataset size: {len(self._dataset)}"
        )

        for prompt in uncached:
            num_correct = 0
            num_total = 0
            predictions: List[Any] = []

            for sample in self._dataset:
                score, prediction, _ = self._evaluator.score_sample_details(
                    prompt.text, sample
                )
                num_correct += score
                num_total += 1
                predictions.append(prediction)

            accuracy = num_correct / num_total if num_total > 0 else 0.0
            fitness = FitnessVector(scores={"centralized": accuracy})
            prompt.fitness = fitness
            prompt.metadata["evaluation_predictions"] = {"centralized": predictions}

            if self._evaluation_cache is not None:
                self._evaluation_cache.set(
                    prompt.text,
                    self._model_name,
                    self._dataset_name,
                    self._evaluation_version,
                    fitness,
                )

        if self._lineage_tracker is not None:
            for prompt in population:
                if not self._lineage_tracker.exists(prompt.id):
                    self._lineage_tracker.register(prompt)
                else:
                    registered = self._lineage_tracker.get(prompt.id)
                    if prompt.fitness is not None:
                        registered.fitness = prompt.fitness
                    if prompt.metadata:
                        registered.metadata.update(prompt.metadata)

        print("Centralized evaluation complete.")
        return population

    def evaluate_on_single_dataset(
        self, prompt_text: str, dataset_name: str, subset_size: int = 100
    ) -> float:
        """Evaluate a single prompt on ONE dataset (for per-dataset breakdown).

        Used at the end of a centralized run to report the winning prompt's
        performance on each individual hospital's dataset subset.

        Args:
            prompt_text: The prompt text to evaluate.
            dataset_name: One of "medqa", "pubmedqa", "medmcqa".
            subset_size: Number of samples to evaluate (default 100).

        Returns:
            Accuracy (micro-average) on that dataset's subset.
        """
        dataset = load_dataset(dataset_name, split="train")
        if len(dataset) > subset_size:
            rng = random.Random(self._random_seed)
            indices = rng.sample(range(len(dataset)), subset_size)
            if hasattr(dataset, "select"):
                dataset = dataset.select(indices)
            else:
                dataset = [dataset[i] for i in indices]

        num_correct = 0
        num_total = 0
        for sample in dataset:
            score, _, _ = self._evaluator.score_sample_details(prompt_text, sample)
            num_correct += score
            num_total += 1

        return num_correct / num_total if num_total > 0 else 0.0

    @property
    def dataset(self) -> List[Any]:
        """Return the pooled evaluation dataset."""
        return self._dataset

    def cache_statistics(self) -> dict[str, object]:
        """Return cache statistics (mirrors FederatedServer interface)."""
        if self._evaluation_cache is not None:
            return {
                "entries": len(self._evaluation_cache),
                "model": self._model_name,
                "dataset": self._dataset_name,
            }
        return {"entries": 0, "model": self._model_name, "dataset": self._dataset_name}

    def summary(self) -> dict[str, object]:
        """Return evaluator summary (mirrors FederatedServer interface)."""
        return {
            "mode": "centralized",
            "pooled_dataset_size": len(self._dataset),
            "dataset_name": self._dataset_name,
            "model": self._model_name,
            "evaluation_version": self._evaluation_version,
            "cache_entries": self.cache_statistics().get("entries", 0),
        }

    def __str__(self) -> str:
        return (
            "CentralizedEvaluator(\n"
            f"    mode=centralized,\n"
            f"    pooled_dataset_size={len(self._dataset)},\n"
            f'    model="{self._model_name}",\n'
            f"    cache_entries={self.cache_statistics().get('entries', 0)}\n"
            ")"
        )