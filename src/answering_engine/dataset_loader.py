"""Dataset loading for the Answering Engine.

This module defines :class:`DatasetLoader`, a thin wrapper around the
existing ``src.data.loader.load_dataset()`` dispatcher. It provides a
unified interface for the Answering Engine to load one or more medical
QA datasets without modifying the existing data loading infrastructure.

The loader delegates all actual dataset fetching and preprocessing to
the existing ``src.data`` package, which normalizes every dataset into
the standardized format::

    {
        "question": str,
        "context": str,
        "choices": list[str],
        "answer": int | str,
    }

Each dataset has a known default evaluation split chosen to match
what is appropriate for paper benchmarking:

- **MedQA** → ``"test"`` (standard USMLE test split)
- **PubMedQA** (pqa_labeled) → ``"train"`` (the only available split)
- **MedMCQA** → ``"validation"`` (test labels are unavailable on HF)
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset

from src.data.loader import load_dataset as _load_dataset


# Canonical set of datasets known to the existing data package.
_KNOWN_DATASETS: frozenset[str] = frozenset({"medqa", "pubmedqa", "medmcqa"})

# Default evaluation split per dataset.  Chosen to reflect the split
# that is (a) intended for final benchmarking and (b) actually present
# in the Hugging Face download.
#
#   medqa     → "test"        Standard USMLE test partition.
#   pubmedqa  → "train"       pqa_labeled has only a "train" split.
#   medmcqa   → "validation"  Test-set labels are hidden on HF.
_DATASET_EVALUATION_SPLITS: dict[str, str] = {
    "medqa": "test",
    "pubmedqa": "train",
    "medmcqa": "validation",
}


class DatasetLoader:
    """Loads medical QA datasets via the existing data infrastructure.

    ``DatasetLoader`` is a stateless utility that wraps the existing
    ``src.data.loader.load_dataset()`` function, adding dataset-name
    validation, per-dataset default splits, and multi-dataset loading
    convenience. It never modifies the underlying data loaders.

    The class is designed to be used by the Answering Engine's
    orchestrator (``AnsweringEngine``) to load evaluation datasets
    independently of the evolutionary pipeline.
    """

    @staticmethod
    def load(name: str, split: str | None = None) -> Dataset:
        """Load a single dataset by name.

        Delegates to ``src.data.loader.load_dataset()`` after
        normalizing the dataset name to lowercase.

        Args:
            name: Dataset name. Must be one of ``"medqa"``,
                ``"pubmedqa"``, or ``"medmcqa"`` (case-insensitive).
            split: Dataset split to load. If ``None``, the default
                evaluation split for the dataset is used (see
                ``get_default_split``).

        Returns:
            A Hugging Face ``Dataset`` object with samples in the
            standardized format (question, context, choices, answer).

        Raises:
            ValueError: If ``name`` is not a recognized dataset name.
        """
        name = name.lower().strip()
        if name not in _KNOWN_DATASETS:
            raise ValueError(
                f"Unknown dataset: '{name}'. "
                f"Available: {sorted(_KNOWN_DATASETS)}"
            )

        if split is None:
            split = DatasetLoader.get_default_split(name)

        return _load_dataset(name, split=split)

    @staticmethod
    def load_multiple(
        names: list[str],
        split: str | None = None,
        split_overrides: dict[str, str] | None = None,
    ) -> dict[str, Dataset]:
        """Load multiple datasets by name.

        Convenience method that loads several datasets and returns them
        keyed by their normalized name.

        Args:
            names: List of dataset names to load.
            split: Global dataset split override. If ``None``, each
                dataset uses its own default evaluation split. If set,
                all datasets are loaded with this split unless
                overridden by ``split_overrides``.
            split_overrides: Optional per-dataset split overrides.
                Keys are dataset names, values are split names.
                Takes precedence over ``split``.

        Returns:
            A dictionary mapping each normalized dataset name to its
            loaded ``Dataset`` object.

        Raises:
            ValueError: If any name is not a recognized dataset.
        """
        split_overrides = split_overrides or {}

        datasets: dict[str, Dataset] = {}
        for name in names:
            normalized = name.lower().strip()
            effective_split = split_overrides.get(normalized, split)
            datasets[normalized] = DatasetLoader.load(
                normalized, split=effective_split
            )
        return datasets

    @staticmethod
    def get_default_split(name: str) -> str:
        """Return the default evaluation split for a dataset.

        Args:
            name: Dataset name (case-insensitive).

        Returns:
            The default split string (e.g., ``"test"``, ``"train"``).

        Raises:
            ValueError: If the dataset name is not recognized.
        """
        name = name.lower().strip()
        if name not in _DATASET_EVALUATION_SPLITS:
            raise ValueError(
                f"Unknown dataset: '{name}'. "
                f"Available: {sorted(_KNOWN_DATASETS)}"
            )
        return _DATASET_EVALUATION_SPLITS[name]

    @staticmethod
    def available_datasets() -> list[str]:
        """Return the list of supported dataset names.

        Returns:
            A sorted list of recognized dataset name strings.
        """
        return sorted(_KNOWN_DATASETS)

    @staticmethod
    def dataset_info(name: str, split: str | None = None) -> dict[str, Any]:
        """Load a dataset and return basic metadata.

        Useful for verifying dataset availability and size before
        running a full evaluation.

        Args:
            name: Dataset name (case-insensitive).
            split: Dataset split to inspect. If ``None``, uses the
                default evaluation split.

        Returns:
            A dictionary with keys ``"name"``, ``"split"``,
            ``"num_samples"``, and ``"sample_keys"``.

        Raises:
            ValueError: If the dataset name is not recognized.
        """
        name_normalized = name.lower().strip()
        if split is None:
            split = DatasetLoader.get_default_split(name_normalized)

        dataset = DatasetLoader.load(name_normalized, split=split)
        return {
            "name": name_normalized,
            "split": split,
            "num_samples": len(dataset),
            "sample_keys": dataset.column_names,
        }
