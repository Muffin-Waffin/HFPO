"""Export management for the Answering Engine.

This module defines :class:`ExportManager`, responsible for writing
evaluation results and reports to disk in multiple formats (CSV, JSON,
and plain text). It is the only component in the Answering Engine that
performs file I/O for results output.

Typical output structure::

    results/answering_engine/
        medqa_results.csv
        pubmedqa_results.csv
        medmcqa_results.csv
        summary.csv
        summary.json
        report.txt
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.answering_engine.models import (
    AnsweringEvaluationRecord,
    AnsweringEvaluationSummary,
)


class ExportManager:
    """Writes evaluation results and reports to disk.

    ``ExportManager`` manages the output directory and provides methods
    for exporting evaluation records (per-question CSV), evaluation
    summaries (CSV and JSON), and formatted text reports. It creates
    the output directory automatically if it does not exist.

    Attributes:
        _output_dir: The directory where all output files are written.
    """

    __slots__ = ("_output_dir",)

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize the export manager with an output directory.

        Args:
            output_dir: Path to the directory where results will be
                written. Created automatically if it does not exist.

        Raises:
            ValueError: If ``output_dir`` is empty or whitespace.
        """
        if not str(output_dir).strip():
            raise ValueError("output_dir must not be empty.")

        self._output_dir = Path(output_dir)

    @property
    def output_dir(self) -> Path:
        """Return the configured output directory path."""
        return self._output_dir

    def _ensure_dir(self) -> None:
        """Create the output directory if it does not exist."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export_records_csv(
        self,
        records: list[AnsweringEvaluationRecord],
        filename: str | None = None,
    ) -> Path:
        """Export per-question evaluation records to a CSV file.

        If ``filename`` is not provided, records are grouped by dataset
        and written to separate files (e.g., ``medqa_results.csv``).
        If ``filename`` is provided, all records are written to a single
        file.

        Args:
            records: The evaluation records to export.
            filename: Optional output filename. If ``None``, records are
                split by dataset into separate files.

        Returns:
            The path to the last written CSV file, or the single output
            file if ``filename`` was provided.
        """
        self._ensure_dir()

        if filename is not None:
            return self._write_records_csv(
                records, self._output_dir / filename
            )

        # Group by dataset and write separate files.
        by_dataset: dict[str, list[AnsweringEvaluationRecord]] = {}
        for record in records:
            by_dataset.setdefault(record.dataset, []).append(record)

        last_path = self._output_dir
        for dataset_name, dataset_records in sorted(by_dataset.items()):
            path = self._output_dir / f"{dataset_name}_results.csv"
            last_path = self._write_records_csv(dataset_records, path)

        return last_path

    def export_summary_csv(
        self,
        summaries: list[AnsweringEvaluationSummary],
        filename: str = "summary.csv",
    ) -> Path:
        """Export evaluation summaries to a CSV file.

        Args:
            summaries: The evaluation summaries to export.
            filename: Output filename. Defaults to ``"summary.csv"``.

        Returns:
            The path to the written CSV file.
        """
        self._ensure_dir()
        path = self._output_dir / filename

        fieldnames = [
            "dataset",
            "model",
            "prompt_id",
            "accuracy",
            "precision",
            "recall",
            "macro_f1",
            "average_latency",
            "average_tokens",
            "total_questions",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for summary in summaries:
                writer.writerow(summary.as_dict())

        print(f"[ExportManager] Summary CSV written: {path}")
        return path

    def export_summary_json(
        self,
        summaries: list[AnsweringEvaluationSummary],
        filename: str = "summary.json",
    ) -> Path:
        """Export evaluation summaries to a JSON file.

        Args:
            summaries: The evaluation summaries to export.
            filename: Output filename. Defaults to ``"summary.json"``.

        Returns:
            The path to the written JSON file.
        """
        self._ensure_dir()
        path = self._output_dir / filename

        data = [s.as_dict() for s in summaries]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"[ExportManager] Summary JSON written: {path}")
        return path

    def export_report(self, report_text: str, filename: str = "report.txt") -> Path:
        """Export a formatted text report to a file.

        Args:
            report_text: The formatted report text (as produced by
                ``ReportGenerator``).
            filename: Output filename. Defaults to ``"report.txt"``.

        Returns:
            The path to the written text file.
        """
        self._ensure_dir()
        path = self._output_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"[ExportManager] Report written: {path}")
        return path

    def export_all(
        self,
        records: list[AnsweringEvaluationRecord],
        summaries: list[AnsweringEvaluationSummary],
        report_text: str | None = None,
    ) -> dict[str, Path]:
        """Export all result artifacts in a single call.

        Convenience method that writes per-dataset CSV files, the
        summary CSV and JSON, and optionally the text report.

        Args:
            records: The per-question evaluation records.
            summaries: The aggregate evaluation summaries.
            report_text: Optional formatted report text. If provided,
                it is written to ``report.txt``.

        Returns:
            A dictionary mapping artifact names to their file paths.
        """
        paths: dict[str, Path] = {}

        # Per-dataset record CSVs.
        self.export_records_csv(records)

        # Collect written dataset files.
        by_dataset: set[str] = {r.dataset for r in records}
        for ds in sorted(by_dataset):
            csv_path = self._output_dir / f"{ds}_results.csv"
            if csv_path.exists():
                paths[f"{ds}_results.csv"] = csv_path

        # Summary files.
        paths["summary.csv"] = self.export_summary_csv(summaries)
        paths["summary.json"] = self.export_summary_json(summaries)

        # Text report.
        if report_text is not None:
            paths["report.txt"] = self.export_report(report_text)

        return paths

    @staticmethod
    def _write_records_csv(
        records: list[AnsweringEvaluationRecord],
        path: Path,
    ) -> Path:
        """Write a list of evaluation records to a CSV file.

        Args:
            records: The records to write.
            path: The output file path.

        Returns:
            The output file path.
        """
        fieldnames = [
            "dataset",
            "sample_id",
            "prompt_id",
            "model",
            "prediction",
            "ground_truth",
            "correct",
            "latency_seconds",
            "generated_tokens",
            "raw_response",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record.as_dict())

        print(f"[ExportManager] Records CSV written: {path}")
        return path
