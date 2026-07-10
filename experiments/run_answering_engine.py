"""CLI entry point for the Answering Engine.

This script provides a command-line interface for running post-evolution
prompt benchmarking. It loads prompts from a file (or uses a default
prompt), evaluates them across specified datasets and models, and
exports paper-ready results.

This script is completely independent from the evolution pipeline.
It does NOT modify any existing experiments, logging, or architecture.

Usage examples::

    # Evaluate prompts from a JSONL file on MedQA using Qwen3-8B:
    python -m experiments.run_answering_engine \\
        --prompts generated_prompts.jsonl \\
        --datasets medqa \\
        --models qwen3-8b

    # Evaluate a single prompt text on multiple datasets:
    python -m experiments.run_answering_engine \\
        --prompt-text "You are a medical expert. Answer accurately." \\
        --datasets medqa pubmedqa medmcqa \\
        --models qwen3-8b phi-4

    # Evaluate on a specific split with a custom output directory:
    python -m experiments.run_answering_engine \\
        --prompts best_prompt.json \\
        --datasets medqa \\
        --models qwen3-8b \\
        --split test \\
        --output-dir results/my_experiment
"""

from __future__ import annotations

import argparse
import sys

from src.answering_engine import AnsweringEngine, PromptLoader


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the Answering Engine CLI.

    Returns:
        A configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Answering Engine — Post-evolution prompt benchmarking. "
            "Evaluates evolved or baseline prompts on medical QA "
            "datasets and exports paper-ready results."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Prompt source (mutually exclusive).
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompts",
        type=str,
        help=(
            "Path to a file containing one or more prompts. "
            "Supported formats: .txt, .json, .jsonl"
        ),
    )
    prompt_group.add_argument(
        "--prompt-text",
        type=str,
        help="A single prompt string to evaluate.",
    )

    # Datasets.
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more dataset names to evaluate on. "
            "Available: medqa, pubmedqa, medmcqa"
        ),
    )

    # Models.
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more model registry keys to use as answer models. "
            "Available: qwen3-8b, phi-4, llama3-8b, mistral-7b, II-medical"
        ),
    )

    # Optional arguments.
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help=(
            "Global dataset split override. If omitted, each dataset "
            "uses its default evaluation split (medqa=test, "
            "pubmedqa=train, medmcqa=validation)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/answering_engine",
        help=(
            "Directory to write results to. "
            "Default: results/answering_engine"
        ),
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help=(
            "Optional prompt identifier when using --prompt-text. "
            "If not provided, a UUID is generated."
        ),
    )
    parser.add_argument(
        "--best-only",
        "--pick-best",
        action="store_true",
        dest="best_only",
        help=(
            "If set, automatically selects the single best evolved prompt "
            "from the loaded file (verifying snapshot scores, fitness vectors, "
            "or generation number) instead of evaluating all loaded prompts."
        ),
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=None,
        help="Filter loaded prompts by 0-based index (e.g., 0 for first, -1 for last).",
    )
    parser.add_argument(
        "--filter-id",
        type=str,
        default=None,
        help="Filter loaded prompts by ID prefix or exact match.",
    )

    return parser


def main() -> None:
    """Parse arguments and run the Answering Engine."""
    parser = _build_parser()
    args = parser.parse_args()

    # Load prompts.
    if args.prompts:
        prompts = PromptLoader.load(args.prompts)
        print(f"Loaded {len(prompts)} prompt(s) from {args.prompts}")
        prompts = PromptLoader.filter_prompts(
            prompts,
            prompt_id=args.filter_id,
            prompt_index=args.prompt_index,
            best_only=args.best_only,
        )
    else:
        prompt = PromptLoader.from_string(
            text=args.prompt_text,
            prompt_id=args.prompt_id,
        )
        prompts = [prompt]
        print(f"Using provided prompt text (id={prompt.id[:12]}...)")

    # Create and run the engine.
    engine = AnsweringEngine(output_dir=args.output_dir)

    summaries = engine.evaluate(
        prompts=prompts,
        dataset_names=args.datasets,
        model_keys=args.models,
        dataset_split=args.split,
    )

    # Print final summary.
    if summaries:
        avg_acc = sum(s.accuracy for s in summaries) / len(summaries)
        print(f"Average accuracy across all evaluations: {avg_acc:.4f}")
    else:
        print("No evaluations were performed.")


if __name__ == "__main__":
    main()
