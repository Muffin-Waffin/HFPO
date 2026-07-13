#!/usr/bin/env python3
"""
Zero-shot baseline evaluation for MedQA, PubMedQA, and MedMCQA.

Runs two prompts ("expert" naive floor, "cot" chain-of-thought floor)
on all three datasets, full sample size. Writes:
  - results/baseline_results.csv  (dataset, prompt, accuracy, n_samples)
  - results/cot_debug_log.jsonl   (raw responses + parse paths for CoT)

Usage:
    python -m experiments.run_baseline              # full datasets
    python -m experiments.run_baseline --limit 20   # smoke test on 20 samples per dataset
"""

import argparse
import csv
import json
from pathlib import Path

from src.llms.loader import load_model
from src.data.loader import load_dataset
from src.evaluation.scorer import score_prompt
from src.prompts.baseline import BASELINE_PROMPTS
from configs.config import DATASET_SPLIT, MAX_NEW_TOKENS, MAX_NEW_TOKENS_COT


def main():
    parser = argparse.ArgumentParser(description="Zero-shot baseline evaluation")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples per dataset (default: full dataset). Use --limit 20 for quick smoke test.",
    )
    args = parser.parse_args()

    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading model...")
    model, tokenizer = load_model()

    datasets_to_eval = ["medqa", "pubmedqa", "medmcqa"]
    # Only run two prompts: expert (naive floor) and cot (CoT floor)
    prompts_to_run = {k: v for k, v in BASELINE_PROMPTS.items() if k in ("expert", "cot")}

    results = []
    all_cot_debug_logs = []

    for dataset_name in datasets_to_eval:
        print("=" * 80)
        print(f"Loading dataset: {dataset_name} (split={DATASET_SPLIT})")
        dataset = load_dataset(dataset_name, split=DATASET_SPLIT)

        if args.limit is not None and len(dataset) > args.limit:
            dataset = dataset.select(range(args.limit))
            print(f"  Limited to {args.limit} samples for smoke test")
        else:
            print(f"  Using full dataset: {len(dataset)} samples")

        for prompt_name, system_prompt in prompts_to_run.items():
            is_cot = prompt_name == "cot"
            max_new_tokens = MAX_NEW_TOKENS_COT if is_cot else MAX_NEW_TOKENS

            print("=" * 80)
            print(f"Dataset: {dataset_name} | Prompt: {prompt_name} | CoT: {is_cot} | Samples: {len(dataset)} | max_new_tokens: {max_new_tokens}")

            if is_cot:
                accuracy, debug_logs = score_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    dataset=dataset,
                    cot=True,
                    max_new_tokens=max_new_tokens,
                    debug=True,
                )
                # Add dataset name to each debug log
                for log in debug_logs:
                    log["dataset"] = dataset_name
                all_cot_debug_logs.extend(debug_logs)
            else:
                accuracy = score_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    dataset=dataset,
                    cot=False,
                    max_new_tokens=max_new_tokens,
                )

            results.append({
                "dataset": dataset_name,
                "prompt": prompt_name,
                "accuracy": accuracy,
                "n_samples": len(dataset),
            })

            print(f"Accuracy: {accuracy:.4f}")

            if len(dataset) > 100:
                print(f"  ... completed {len(dataset)} samples")

    # Write results CSV
    output_path = results_dir / "baseline_results.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "prompt", "accuracy", "n_samples"])
        writer.writeheader()
        writer.writerows(results)

    # Write CoT debug log
    if all_cot_debug_logs:
        debug_path = results_dir / "cot_debug_log.jsonl"
        with open(debug_path, "w") as f:
            for log in all_cot_debug_logs:
                f.write(json.dumps(log) + "\n")
        print(f"\nCoT debug log written to {debug_path} ({len(all_cot_debug_logs)} entries)")

    # Print summary table
    print("=" * 80)
    print(f"Baseline evaluation complete. Results saved to {output_path}")
    print("\nPer-dataset results:")
    for r in results:
        print(f"  {r['dataset']:8s} | {r['prompt']:12s} | acc={r['accuracy']:.4f} | n={r['n_samples']}")

    # Macro-average per prompt (mean of 3 dataset accuracies)
    print("\nMacro-average per prompt (mean across 3 datasets):")
    for prompt_name in ("expert", "cot"):
        prompt_results = [r for r in results if r["prompt"] == prompt_name]
        if prompt_results:
            macro_avg = sum(r["accuracy"] for r in prompt_results) / len(prompt_results)
            print(f"  {prompt_name:12s}: {macro_avg:.4f} (n_datasets={len(prompt_results)})")


if __name__ == "__main__":
    main()