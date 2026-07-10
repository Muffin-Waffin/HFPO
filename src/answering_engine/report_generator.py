"""Report generation for the Answering Engine.

This module defines :class:`ReportGenerator`, responsible for formatting
evaluation results into human-readable summary tables and structured
report text suitable for inclusion in a research paper or for quick
console review during experiments.

The generator produces formatted text reports from
:class:`AnsweringEvaluationSummary` objects. Actual file I/O is handled
by :class:`ExportManager`; this module only formats data.
"""

from __future__ import annotations

from typing import Any

from src.answering_engine.models import (
    AnsweringEvaluationRecord,
    AnsweringEvaluationSummary,
)


class ReportGenerator:
    """Formats evaluation results into human-readable reports.

    ``ReportGenerator`` is a stateless utility that takes evaluation
    summaries and records and produces formatted text output. It does
    not perform any file I/O; that responsibility belongs to
    :class:`ExportManager`.

    Report formats include:
    - **Summary table**: A formatted text table of aggregate metrics
      per (dataset, model, prompt) combination.
    - **Console report**: A formatted string suitable for printing to
      the terminal during evaluation.
    - **Confusion matrix display**: A formatted text representation of
      a confusion matrix dictionary.
    """

    @staticmethod
    def generate_summary_table(
        summaries: list[AnsweringEvaluationSummary],
    ) -> str:
        """Generate a formatted text table of evaluation summaries.

        Each row represents one (dataset, model, prompt) combination
        with columns for accuracy, precision, recall, F1, latency,
        tokens, and question count.

        Args:
            summaries: The evaluation summaries to format.

        Returns:
            A formatted multi-line string containing the summary table.
            Returns a message indicating no results if the list is empty.
        """
        if not summaries:
            return "No evaluation results to display."

        # Column definitions: (header, format_func, width).
        columns = [
            ("Dataset", lambda s: s.dataset, 12),
            ("Model", lambda s: s.model, 14),
            ("Prompt ID", lambda s: s.prompt_id[:12], 14),
            ("Accuracy", lambda s: f"{s.accuracy:.4f}", 10),
            ("Precision", lambda s: f"{s.precision:.4f}", 10),
            ("Recall", lambda s: f"{s.recall:.4f}", 10),
            ("Macro F1", lambda s: f"{s.macro_f1:.4f}", 10),
            ("Avg Lat(s)", lambda s: f"{s.average_latency:.2f}", 10),
            ("Avg Tokens", lambda s: f"{s.average_tokens:.1f}", 10),
            ("N", lambda s: str(s.total_questions), 6),
        ]

        # Build header.
        header = " | ".join(
            h.ljust(w) for h, _, w in columns
        )
        separator = "-+-".join("-" * w for _, _, w in columns)

        # Build rows.
        rows: list[str] = []
        for summary in summaries:
            row = " | ".join(
                fmt(summary).ljust(w) for _, fmt, w in columns
            )
            rows.append(row)

        lines = [header, separator] + rows
        return "\n".join(lines)

    @staticmethod
    def generate_console_report(
        summaries: list[AnsweringEvaluationSummary],
        average_accuracy: float | None = None,
    ) -> str:
        """Generate a complete console report.

        Includes a header, the summary table, and optionally the
        average accuracy across all evaluations.

        Args:
            summaries: The evaluation summaries to report.
            average_accuracy: Optional pre-computed average accuracy.
                If ``None``, it is computed from the summaries.

        Returns:
            A formatted multi-line string suitable for printing.
        """
        lines: list[str] = []
        lines.append("")
        lines.append("=" * 72)
        lines.append("  ANSWERING ENGINE — EVALUATION REPORT")
        lines.append("=" * 72)
        lines.append("")

        table = ReportGenerator.generate_summary_table(summaries)
        lines.append(table)

        if summaries:
            if average_accuracy is None:
                average_accuracy = (
                    sum(s.accuracy for s in summaries) / len(summaries)
                )

            lines.append("")
            lines.append(f"  Average Accuracy: {average_accuracy:.4f}")
            lines.append(
                f"  Total Evaluations: {len(summaries)} "
                f"(Prompt × Dataset × Model combinations)"
            )
            total_questions = sum(s.total_questions for s in summaries)
            lines.append(f"  Total Questions Answered: {total_questions}")

        lines.append("")
        lines.append("=" * 72)
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_confusion_matrix(
        cm_data: dict[str, Any],
        title: str = "Confusion Matrix",
    ) -> str:
        """Format a confusion matrix dictionary as a readable text table.

        Args:
            cm_data: A confusion matrix dictionary as returned by
                ``MetricsCalculator.compute_confusion_matrix()``, with
                keys ``"labels"``, ``"matrix"``, and ``"total"``.
            title: Optional title for the matrix display.

        Returns:
            A formatted multi-line string showing the confusion matrix
            with ground-truth labels as rows and predicted labels as
            columns.
        """
        labels = cm_data["labels"]
        matrix = cm_data["matrix"]

        if not labels:
            return f"{title}: No data."

        # Determine column width.
        col_width = max(len(str(l)) for l in labels)
        col_width = max(col_width, 5)

        lines: list[str] = []
        lines.append(f"  {title}")
        lines.append(f"  (rows = ground truth, columns = predicted)")
        lines.append("")

        # Header row.
        header = " " * (col_width + 2)
        header += "  ".join(str(l).rjust(col_width) for l in labels)
        lines.append(f"  {header}")

        # Separator.
        sep_width = (col_width + 2) + len(labels) * (col_width + 2)
        lines.append(f"  {'-' * sep_width}")

        # Data rows.
        for gt_label in labels:
            if gt_label not in matrix:
                continue
            row_data = matrix[gt_label]
            row_str = str(gt_label).rjust(col_width) + " |"
            for pred_label in labels:
                count = row_data.get(pred_label, 0)
                row_str += str(count).rjust(col_width + 2)
            lines.append(f"  {row_str}")

        lines.append("")
        lines.append(f"  Total samples: {cm_data['total']}")

        return "\n".join(lines)
