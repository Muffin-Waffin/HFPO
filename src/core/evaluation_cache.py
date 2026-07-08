"""Evaluation caching for HFPO (Heterogeneous Federated Prompt Optimization).

This module defines :class:`EvaluationCache`, which is owned by the
Federated Server and exists to eliminate repeated evaluations of identical
prompts across generations. Federated evaluation of a prompt requires
broadcasting it to every participating hospital and running expensive LLM
inference at each one; if the same prompt text has already been evaluated
under the same experimental configuration (model, dataset, and evaluation
pipeline version), the previously computed `FitnessVector` can be reused
instead. This dramatically reduces redundant inference cost during HFPO
evolution while preserving correctness, since the cache key incorporates
the model name, dataset name, and evaluation version, ensuring that cached
results are automatically invalidated whenever any part of the evaluation
configuration changes.
"""

from __future__ import annotations

import hashlib

from src.core.fitness_vector import FitnessVector


class EvaluationCache:
    """Caches FitnessVector results to avoid redundant federated evaluations.

    `EvaluationCache` is owned by the Federated Server. Before broadcasting
    a prompt to hospitals for evaluation, the server can consult this cache
    using the prompt text together with the current model name, dataset
    name, and evaluation pipeline version. If an identical combination has
    been evaluated before, the cached `FitnessVector` is returned instead
    of re-running federated evaluation. This is especially valuable in
    genetic algorithm search, where identical or near-identical prompts may
    reappear across generations (for example, via elite survival or
    convergence toward similar phrasings).

    The cache key is a SHA256 hash of the concatenation of the prompt text,
    model name, dataset name, and evaluation version. Including the latter
    three values in the key ensures that cached evaluations are
    automatically invalidated whenever the model, dataset, or evaluation
    pipeline changes, even if the prompt text itself is unchanged.

    This class does not persist its cache to disk and does not implement
    any eviction policy (such as LRU or TTL); entries remain until
    `clear()` is called or the process ends.
    """

    def __init__(self) -> None:
        """Initialize an empty evaluation cache."""
        self._store: dict[str, FitnessVector] = {}

    @staticmethod
    def _hash(
        prompt_text: str,
        model_name: str,
        dataset_name: str,
        evaluation_version: str,
    ) -> str:
        """Compute the SHA256 cache key for an evaluation configuration.

        Args:
            prompt_text: The prompt text being evaluated.
            model_name: Name of the model used for evaluation.
            dataset_name: Name of the dataset used for evaluation.
            evaluation_version: Version identifier of the evaluation
                pipeline.

        Returns:
            A hexadecimal SHA256 digest computed from the concatenation of
            `prompt_text`, `model_name`, `dataset_name`, and
            `evaluation_version`.
        """
        combined = prompt_text + model_name + dataset_name + evaluation_version
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_inputs(
        prompt_text: str,
        model_name: str,
        dataset_name: str,
        evaluation_version: str,
    ) -> None:
        """Validate the components used to build a cache key.

        Args:
            prompt_text: The prompt text being evaluated.
            model_name: Name of the model used for evaluation.
            dataset_name: Name of the dataset used for evaluation.
            evaluation_version: Version identifier of the evaluation
                pipeline.

        Raises:
            ValueError: If any of the supplied values is empty or
                whitespace.
        """
        if not prompt_text.strip():
            raise ValueError("EvaluationCache prompt_text must not be empty.")

        if not model_name.strip():
            raise ValueError("EvaluationCache model_name must not be empty.")

        if not dataset_name.strip():
            raise ValueError("EvaluationCache dataset_name must not be empty.")

        if not evaluation_version.strip():
            raise ValueError(
                "EvaluationCache evaluation_version must not be empty."
            )

    def get(
        self,
        prompt_text: str,
        model_name: str,
        dataset_name: str,
        evaluation_version: str,
    ) -> FitnessVector | None:
        """Retrieve a cached FitnessVector, if one exists.

        Args:
            prompt_text: The prompt text being evaluated.
            model_name: Name of the model used for evaluation.
            dataset_name: Name of the dataset used for evaluation.
            evaluation_version: Version identifier of the evaluation
                pipeline.

        Returns:
            A new `FitnessVector` reconstructed from the cached entry's
            data if present, or `None` if no matching entry exists. The
            returned object is independent of the cached object, so
            callers cannot mutate cached state.

        Raises:
            ValueError: If any input is empty or whitespace.
        """
        self._validate_inputs(
            prompt_text, model_name, dataset_name, evaluation_version
        )
        key = self._hash(prompt_text, model_name, dataset_name, evaluation_version)
        cached_vector = self._store.get(key)

        if cached_vector is None:
            return None

        return FitnessVector(scores=dict(cached_vector.as_dict()))

    def set(
        self,
        prompt_text: str,
        model_name: str,
        dataset_name: str,
        evaluation_version: str,
        fitness: FitnessVector,
    ) -> None:
        """Store a FitnessVector in the cache.

        Args:
            prompt_text: The prompt text that was evaluated.
            model_name: Name of the model used for evaluation.
            dataset_name: Name of the dataset used for evaluation.
            evaluation_version: Version identifier of the evaluation
                pipeline.
            fitness: The `FitnessVector` to cache.

        Raises:
            ValueError: If any of the string inputs is empty or whitespace.
        """
        self._validate_inputs(
            prompt_text, model_name, dataset_name, evaluation_version
        )
        key = self._hash(prompt_text, model_name, dataset_name, evaluation_version)
        self._store[key] = FitnessVector(scores=dict(fitness.as_dict()))

    def contains(
        self,
        prompt_text: str,
        model_name: str,
        dataset_name: str,
        evaluation_version: str,
    ) -> bool:
        """Check whether an evaluation result is cached.

        Args:
            prompt_text: The prompt text being evaluated.
            model_name: Name of the model used for evaluation.
            dataset_name: Name of the dataset used for evaluation.
            evaluation_version: Version identifier of the evaluation
                pipeline.

        Returns:
            True if a cached entry exists for this exact combination,
            False otherwise.

        Raises:
            ValueError: If any input is empty or whitespace.
        """
        self._validate_inputs(
            prompt_text, model_name, dataset_name, evaluation_version
        )
        key = self._hash(prompt_text, model_name, dataset_name, evaluation_version)
        return key in self._store

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    def size(self) -> int:
        """Return the number of cached evaluations.

        Returns:
            The number of entries currently stored in the cache.
        """
        return len(self._store)

    def __len__(self) -> int:
        """Return the number of cached evaluations.

        Returns:
            The same value as `size()`.
        """
        return self.size()

    def __str__(self) -> str:
        """Return a concise, human-readable summary of this cache.

        Returns:
            A string of the form 'EvaluationCache(size=142)'.
        """
        return f"EvaluationCache(size={self.size()})"