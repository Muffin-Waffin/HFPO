"""Answering Engine — Post-evolution prompt benchmarking subsystem.

This package provides the Answering Engine, a completely independent
evaluation module that sits after the evolutionary pipeline and
benchmarks evolved (or baseline) prompts across multiple medical QA
datasets using multiple answer models.

The Answering Engine is **not** an optimizer. It is purely an evaluation
and reporting module that produces paper-ready results (CSV, JSON,
summary tables) for inclusion in research publications.

Architecture::

    FedGAPrompt
        │
        ▼
    Best Prompt(s)
        │
        ▼
    ----------------------------
      Answering Engine
    ----------------------------
        │
        ├── PromptLoader           — Load prompts from files, objects, or strings
        ├── DatasetLoader          — Load medical QA datasets
        ├── ResponseGenerator      — Query answer models
        ├── PredictionExtractor    — Extract predicted answers
        ├── MetricsCalculator      — Compute paper metrics
        ├── ReportGenerator        — Format results for display
        └── ExportManager          — Write results to disk

Usage::

    from src.answering_engine import AnsweringEngine, PromptLoader

    engine = AnsweringEngine(output_dir="results/answering_engine")
    prompts = PromptLoader.load("generated_prompts.jsonl")

    summaries = engine.evaluate(
        prompts=prompts,
        dataset_names=["medqa", "pubmedqa"],
        model_keys=["qwen3-8b"],
    )
"""

from src.answering_engine.engine import AnsweringEngine
from src.answering_engine.models import (
    AnsweringEvaluationRecord,
    AnsweringEvaluationSummary,
    GenerationResult,
    Prompt,
)
from src.answering_engine.prompt_loader import PromptLoader
from src.answering_engine.dataset_loader import DatasetLoader
from src.answering_engine.response_generator import ResponseGenerator
from src.answering_engine.prediction_extractor import PredictionExtractor
from src.answering_engine.metrics_calculator import MetricsCalculator
from src.answering_engine.report_generator import ReportGenerator
from src.answering_engine.export_manager import ExportManager

__all__ = [
    "AnsweringEngine",
    "AnsweringEvaluationRecord",
    "AnsweringEvaluationSummary",
    "DatasetLoader",
    "ExportManager",
    "GenerationResult",
    "MetricsCalculator",
    "PredictionExtractor",
    "Prompt",
    "PromptLoader",
    "ReportGenerator",
    "ResponseGenerator",
]
