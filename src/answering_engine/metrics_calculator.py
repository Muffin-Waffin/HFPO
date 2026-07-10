"""Metrics computation for the Answering Engine.

This module defines :class:`MetricsCalculator`, responsible for computing
paper-ready evaluation metrics from a collection of
:class:`AnsweringEvaluationRecord` objects. It computes accuracy, macro
precision, macro recall, macro F1, per-dataset accuracy, average
accuracy across datasets, confusion matrices, and performance statistics
(average latency and average generated tokens).

All metric computations are implemented from scratch using only the
Python standard library to avoid adding external dependencies (e.g.,
scikit-learn) to the project.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.answering_engine.models import (
    AnsweringEvaluationRecord,
    AnsweringEvaluationSummary,
)


class MetricsCalculator:
    """Computes paper-ready evaluation metrics from evaluation records.

    ``MetricsCalculator`` is a stateless utility that takes a list of
    :class:`AnsweringEvaluationRecord` objects and produces aggregate
    metrics suitable for inclusion in a research paper. It computes:

    - **Accuracy**: Fraction of correctly answered questions.
    - **Macro Precision**: Average of per-class precision values.
    - **Macro Recall**: Average of per-class recall values.
    - **Macro F1**: Average of per-class F1 scores.
    - **Confusion Matrix**: Counts of (ground_truth, prediction) pairs.
    - **Average Latency**: Mean generation time in seconds.
    - **Average Tokens**: Mean generated token count.

    Unparseable predictions (``None``) are counted as incorrect for
    accuracy and contribute to false negatives for the actual ground
    truth class in macro metrics, correctly penalizing models that
    produce unparseable outputs.
    """

    @staticmethod
    def compute_summary(
        records: list[AnsweringEvaluationRecord],
        dataset: str,
        model: str,
        prompt_id: str,
    ) -> AnsweringEvaluationSummary:
        """Compute aggregate metrics from a list of evaluation records.

        Args:
            records: The per-question evaluation records to aggregate.
                All records should share the same dataset, model, and
                prompt_id.
            dataset: Name of the evaluated dataset.
            model: Model registry key used for generation.
            prompt_id: Identifier of the prompt used.

        Returns:
            An :class:`AnsweringEvaluationSummary` containing the
            computed aggregate metrics.

        Raises:
            ValueError: If ``records`` is empty.
        """
        if not records:
            raise ValueError(
                "Cannot compute summary from an empty list of records."
            )

        total = len(records)

        # Accuracy.
        num_correct = sum(1 for r in records if r.correct)
        accuracy = num_correct / total

        # Collect predictions and ground truths.
        predictions = [r.prediction for r in records]
        ground_truths = [r.ground_truth for r in records]

        # Macro precision, recall, F1.
        precision, recall, macro_f1 = MetricsCalculator._compute_macro_metrics(
            predictions, ground_truths
        )

        # Latency and token statistics.
        avg_latency = sum(r.latency_seconds for r in records) / total
        avg_tokens = sum(r.generated_tokens for r in records) / total

        return AnsweringEvaluationSummary(
            dataset=dataset,
            model=model,
            prompt_id=prompt_id,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            macro_f1=macro_f1,
            average_latency=avg_latency,
            average_tokens=avg_tokens,
            total_questions=total,
        )

    @staticmethod
    def compute_confusion_matrix(
        records: list[AnsweringEvaluationRecord],
    ) -> dict[str, Any]:
        """Compute a confusion matrix from evaluation records.

        The matrix is returned as a dictionary with sorted label lists
        and a nested dictionary of counts.

        Args:
            records: The per-question evaluation records.

        Returns:
            A dictionary with keys:
            - ``"labels"``: Sorted list of all unique labels (including
              ``"N/A"`` for unparseable predictions).
            - ``"matrix"``: A dict of dicts where
              ``matrix[ground_truth][prediction]`` gives the count.
            - ``"total"``: Total number of records.
        """
        _UNPARSEABLE = "N/A"

        # Collect all labels.
        all_labels: set[str] = set()
        for r in records:
            gt = str(r.ground_truth)
            pred = str(r.prediction) if r.prediction is not None else _UNPARSEABLE
            all_labels.add(gt)
            all_labels.add(pred)

        sorted_labels = sorted(all_labels, key=lambda x: (x == _UNPARSEABLE, x))

        # Build matrix.
        matrix: dict[str, dict[str, int]] = {
            gt: {pred: 0 for pred in sorted_labels}
            for gt in sorted_labels
        }

        for r in records:
            gt = str(r.ground_truth)
            pred = str(r.prediction) if r.prediction is not None else _UNPARSEABLE
            matrix[gt][pred] += 1

        return {
            "labels": sorted_labels,
            "matrix": matrix,
            "total": len(records),
        }

    @staticmethod
    def compute_per_dataset_accuracy(
        summaries: list[AnsweringEvaluationSummary],
    ) -> dict[str, float]:
        """Extract per-dataset accuracy from a list of summaries.

        Args:
            summaries: A list of evaluation summaries, potentially
                spanning multiple datasets.

        Returns:
            A dictionary mapping each dataset name to its accuracy.
        """
        return {s.dataset: s.accuracy for s in summaries}

    @staticmethod
    def compute_average_accuracy(
        summaries: list[AnsweringEvaluationSummary],
    ) -> float:
        """Compute the average accuracy across all summaries.

        Args:
            summaries: A list of evaluation summaries.

        Returns:
            The arithmetic mean of the accuracy values across all
            summaries, or 0.0 if the list is empty.
        """
        if not summaries:
            return 0.0
        return sum(s.accuracy for s in summaries) / len(summaries)

    @staticmethod
    def _compute_macro_metrics(
        predictions: list[object],
        ground_truths: list[object],
    ) -> tuple[float, float, float]:
        """Compute macro-averaged precision, recall, and F1.

        Computes per-class precision, recall, and F1 using all unique
        ground-truth labels, then averages across classes. Predictions
        that are ``None`` (unparseable) are never equal to any real
        label, so they naturally contribute to false negatives for the
        actual ground-truth class without inflating any class's false
        positive count.

        Args:
            predictions: List of predicted answers (may contain
                ``None`` for unparseable responses).
            ground_truths: List of ground-truth answers.

        Returns:
            A tuple of (macro_precision, macro_recall, macro_f1).
        """
        # Use ground-truth labels as the class set to avoid creating
        # spurious classes from unparseable predictions.
        labels = sorted(set(str(gt) for gt in ground_truths))

        if not labels:
            return 0.0, 0.0, 0.0

        per_class_precision: list[float] = []
        per_class_recall: list[float] = []
        per_class_f1: list[float] = []

        for label in labels:
            tp = 0
            fp = 0
            fn = 0

            for pred, gt in zip(predictions, ground_truths):
                pred_str = str(pred) if pred is not None else None
                gt_str = str(gt)

                if pred_str == label and gt_str == label:
                    tp += 1
                elif pred_str == label and gt_str != label:
                    fp += 1
                elif pred_str != label and gt_str == label:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_class_precision.append(precision)
            per_class_recall.append(recall)
            per_class_f1.append(f1)

        macro_precision = sum(per_class_precision) / len(labels)
        macro_recall = sum(per_class_recall) / len(labels)
        macro_f1 = sum(per_class_f1) / len(labels)

        return macro_precision, macro_recall, macro_f1
