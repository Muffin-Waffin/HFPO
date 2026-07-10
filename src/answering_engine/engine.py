"""Top-level orchestrator for the Answering Engine.

This module defines :class:`AnsweringEngine`, the primary entry point
for post-evolution prompt benchmarking. It orchestrates the full
evaluation pipeline:

    Load Prompts → Load Datasets → For each (Prompt × Dataset × Model)
    → Generate Responses → Extract Predictions → Compute Metrics
    → Generate Reports → Export Results

The engine is completely independent from the evolutionary pipeline.
It consumes evolved prompts (or any other prompts) as read-only inputs
and produces paper-ready evaluation results without modifying any
existing subsystem.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.answering_engine.models import (
    AnsweringEvaluationRecord,
    AnsweringEvaluationSummary,
    Prompt,
)
from src.answering_engine.prompt_loader import PromptLoader
from src.answering_engine.dataset_loader import DatasetLoader
from src.answering_engine.response_generator import ResponseGenerator
from src.answering_engine.prediction_extractor import PredictionExtractor
from src.answering_engine.metrics_calculator import MetricsCalculator
from src.answering_engine.report_generator import ReportGenerator
from src.answering_engine.export_manager import ExportManager


class AnsweringEngine:
    """Orchestrates post-evolution prompt evaluation and reporting.

    ``AnsweringEngine`` is the top-level coordinator for benchmarking
    evolved (or baseline) prompts across multiple datasets and models.
    It assembles the full evaluation pipeline from independent,
    single-responsibility components:

    - :class:`PromptLoader` — loads prompts from various sources.
    - :class:`DatasetLoader` — loads medical QA datasets.
    - :class:`ResponseGenerator` — queries models for responses.
    - :class:`PredictionExtractor` — extracts predicted answers.
    - :class:`MetricsCalculator` — computes paper metrics.
    - :class:`ReportGenerator` — formats results for display.
    - :class:`ExportManager` — writes results to disk.

    The engine evaluates every combination of (prompt × dataset × model)
    and produces per-question records, aggregate summaries, and
    formatted reports.

    Attributes:
        _output_dir: Directory where all result files are written.
        _export_manager: The export manager for file output.
    """

    __slots__ = ("_output_dir", "_export_manager")

    def __init__(self, output_dir: str | Path = "results/answering_engine") -> None:
        """Initialize the Answering Engine.

        Args:
            output_dir: Path to the directory where evaluation results
                will be written. Created automatically if it does not
                exist. Defaults to ``"results/answering_engine"``.
        """
        self._output_dir = Path(output_dir)
        self._export_manager = ExportManager(self._output_dir)

    @property
    def output_dir(self) -> Path:
        """Return the configured output directory path."""
        return self._output_dir

    def evaluate(
        self,
        prompts: list[Prompt],
        dataset_names: list[str],
        model_keys: list[str],
        dataset_split: str | None = None,
    ) -> list[AnsweringEvaluationSummary]:
        """Run the full evaluation pipeline.

        Evaluates every combination of (prompt × dataset × model),
        computing per-question records and aggregate summaries.
        Results are automatically exported to the output directory.

        The evaluation order is model → prompt → dataset, so that
        each model is loaded only once for all prompt/dataset
        combinations.

        Args:
            prompts: The prompts to evaluate. Can be evolved prompts,
                baseline prompts, or any other system prompts.
            dataset_names: Names of the datasets to evaluate on
                (e.g., ``["medqa", "pubmedqa", "medmcqa"]``).
            model_keys: Model registry keys to use as answer models
                (e.g., ``["qwen3-8b", "phi-4"]``).
            dataset_split: Global dataset split override.  If ``None``
                (default), each dataset uses its own default evaluation
                split (e.g., ``"test"`` for MedQA, ``"train"`` for
                PubMedQA).  If set, all datasets are loaded with this
                split.

        Returns:
            A list of :class:`AnsweringEvaluationSummary` objects, one
            per (dataset × model × prompt) combination.
        """
        total_start = time.perf_counter()

        # Resolve per-dataset splits for display.
        resolved_splits = {
            ds.lower().strip(): (
                dataset_split
                if dataset_split is not None
                else DatasetLoader.get_default_split(ds)
            )
            for ds in dataset_names
        }

        print()
        print("=" * 72)
        print("  ANSWERING ENGINE — STARTING EVALUATION")
        print("=" * 72)
        print(f"  Prompts:  {len(prompts)}")
        print(f"  Datasets: {dataset_names}")
        print(f"  Splits:   {resolved_splits}")
        print(f"  Models:   {model_keys}")
        total_combos = len(prompts) * len(dataset_names) * len(model_keys)
        print(f"  Total combinations: {total_combos}")
        print("=" * 72)
        print()

        all_records: list[AnsweringEvaluationRecord] = []
        all_summaries: list[AnsweringEvaluationSummary] = []

        # Preload all datasets to avoid reloading per model.
        print("[AnsweringEngine] Loading datasets...")
        datasets = DatasetLoader.load_multiple(dataset_names, split=dataset_split)
        for ds_name, ds in datasets.items():
            print(f"  {ds_name} ({resolved_splits[ds_name]}): {len(ds)} samples")
        print()

        # Evaluate: model → prompt → dataset.
        combo_idx = 0
        for model_key in model_keys:
            generator = ResponseGenerator.from_model_key(model_key)

            for prompt in prompts:
                for ds_name in dataset_names:
                    combo_idx += 1
                    dataset = datasets[ds_name.lower().strip()]

                    print(
                        f"[AnsweringEngine] Evaluating combination "
                        f"{combo_idx}/{total_combos}: "
                        f"model={model_key}, prompt={prompt.id[:12]}..., "
                        f"dataset={ds_name} ({len(dataset)} samples)"
                    )

                    records = self._evaluate_single(
                        generator=generator,
                        prompt=prompt,
                        dataset=dataset,
                        dataset_name=ds_name,
                        model_key=model_key,
                    )

                    all_records.extend(records)

                    # Compute summary for this combination.
                    summary = MetricsCalculator.compute_summary(
                        records=records,
                        dataset=ds_name,
                        model=model_key,
                        prompt_id=prompt.id,
                    )
                    all_summaries.append(summary)

                    print(
                        f"  → Accuracy: {summary.accuracy:.4f} | "
                        f"F1: {summary.macro_f1:.4f} | "
                        f"Avg Latency: {summary.average_latency:.2f}s"
                    )
                    print()

        # Generate report.
        avg_accuracy = MetricsCalculator.compute_average_accuracy(all_summaries)
        report = ReportGenerator.generate_console_report(
            all_summaries, average_accuracy=avg_accuracy
        )
        print(report)

        # Export all results.
        print("[AnsweringEngine] Exporting results...")
        self._export_manager.export_all(
            records=all_records,
            summaries=all_summaries,
            report_text=report,
        )

        total_elapsed = time.perf_counter() - total_start
        print(
            f"[AnsweringEngine] Evaluation complete in "
            f"{total_elapsed:.1f}s. Results written to: {self._output_dir}"
        )
        print()

        return all_summaries

    def evaluate_single_prompt(
        self,
        prompt_text: str,
        dataset_names: list[str],
        model_keys: list[str],
        prompt_id: str | None = None,
        dataset_split: str | None = None,
    ) -> list[AnsweringEvaluationSummary]:
        """Convenience method to evaluate a single prompt string.

        Wraps the prompt text into a :class:`Prompt` object and
        delegates to :meth:`evaluate`.

        Args:
            prompt_text: The system prompt text to evaluate.
            dataset_names: Names of the datasets to evaluate on.
            model_keys: Model registry keys to use.
            prompt_id: Optional prompt identifier. If ``None``, one
                is generated automatically.
            dataset_split: Global split override.  ``None`` means use
                each dataset's default evaluation split.

        Returns:
            A list of :class:`AnsweringEvaluationSummary` objects.
        """
        prompt = PromptLoader.from_string(prompt_text, prompt_id=prompt_id)
        return self.evaluate(
            prompts=[prompt],
            dataset_names=dataset_names,
            model_keys=model_keys,
            dataset_split=dataset_split,
        )

    def evaluate_from_file(
        self,
        prompt_path: str | Path,
        dataset_names: list[str],
        model_keys: list[str],
        dataset_split: str | None = None,
    ) -> list[AnsweringEvaluationSummary]:
        """Convenience method to evaluate prompts loaded from a file.

        Loads prompts using :class:`PromptLoader` auto-detection and
        delegates to :meth:`evaluate`.

        Args:
            prompt_path: Path to a file containing one or more prompts
                (``.txt``, ``.json``, or ``.jsonl``).
            dataset_names: Names of the datasets to evaluate on.
            model_keys: Model registry keys to use.
            dataset_split: Global split override.  ``None`` means use
                each dataset's default evaluation split.

        Returns:
            A list of :class:`AnsweringEvaluationSummary` objects.
        """
        prompts = PromptLoader.load(prompt_path)
        print(f"[AnsweringEngine] Loaded {len(prompts)} prompt(s) from {prompt_path}")
        return self.evaluate(
            prompts=prompts,
            dataset_names=dataset_names,
            model_keys=model_keys,
            dataset_split=dataset_split,
        )

    def _evaluate_single(
        self,
        generator: ResponseGenerator,
        prompt: Prompt,
        dataset: Any,
        dataset_name: str,
        model_key: str,
    ) -> list[AnsweringEvaluationRecord]:
        """Evaluate a single (prompt, dataset, model) combination.

        Iterates over every sample in the dataset, generates a response,
        extracts the prediction, and records the result.

        Args:
            generator: The response generator (with a loaded model).
            prompt: The prompt to evaluate.
            dataset: The loaded dataset.
            dataset_name: Name of the dataset.
            model_key: Model registry key.

        Returns:
            A list of :class:`AnsweringEvaluationRecord` objects, one
            per dataset sample.
        """
        records: list[AnsweringEvaluationRecord] = []

        for sample_id, sample in enumerate(dataset):
            # Generate response.
            result = generator.generate(
                system_prompt=prompt.text,
                sample=sample,
            )

            # Extract prediction.
            prediction = PredictionExtractor.extract(
                response=result.clean_response,
                sample=sample,
            )

            # Determine correctness.
            correct = prediction == sample["answer"]

            # Record result.
            record = AnsweringEvaluationRecord(
                dataset=dataset_name,
                sample_id=sample_id,
                prompt_id=prompt.id,
                model=model_key,
                prediction=prediction,
                ground_truth=sample["answer"],
                correct=correct,
                latency_seconds=result.latency_seconds,
                generated_tokens=result.generated_tokens,
                raw_response=result.raw_response,
            )
            records.append(record)

            # Progress logging every 50 samples.
            if (sample_id + 1) % 50 == 0:
                running_acc = sum(1 for r in records if r.correct) / len(records)
                print(
                    f"    [{sample_id + 1}/{len(dataset)}] "
                    f"Running accuracy: {running_acc:.4f}"
                )

        return records
