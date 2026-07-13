#!/usr/bin/env python3
"""
Random-search baseline for Experiment 2 (compute-matched to 10-gen HFPO).

Generates 182 random prompt candidates (compute-matched to HFPO 10 generations:
gen-0 seed population of 20 + 9 generations × 18 offspring = 182 total),
evaluates each on the SAME fixed 100-sample subset per hospital (medqa,
pubmedqa, medmcqa) using RANDOM_SEED=51, computes FitnessVector.average()
as scalar fitness, and tracks running best-fitness-so-far.

Outputs: results/random_search_results.csv with columns:
  candidate_index, prompt_text, medqa_score, pubmedqa_score, medmcqa_score,
  fitness, best_fitness_so_far

Usage:
    python -u -m experiments.run_random_search              # full 182 candidates
    python -u -m experiments.run_random_search --limit 10   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import random
import time
import uuid
from pathlib import Path

from configs import config
from src.core.fitness_vector import FitnessVector
from src.core.lineage_tracker import LineageTracker
from src.core.evaluation_cache import EvaluationCache
from src.core.prompt_candidate import PromptCandidate
from src.core.seed_prompts import SEED_PROMPTS
from src.data.loader import load_dataset
from src.evaluation.qwen_evaluator import QwenEvaluator
from src.federated.aggregator import Aggregator
from src.federated.federated_server import FederatedServer
from src.federated.hospital_client import HospitalClient
from src.llms.loader import load_model
from src.evolution.prompt_generator.cleaner import PromptCleaner
from src.evolution.prompt_generator.generator import PromptGenerator
from src.evolution.prompt_generator.templates import PromptTemplateBuilder
from src.evolution.prompt_generator.validator import PromptValidator
from src.evolution.prompt_generator.candidate_parser import CandidateParser
from src.evolution.prompt_generator.similarity_selector import PromptSimilaritySelector
from src.evolution.prompt_generator.models import PromptGenerationRequest, ParentPerformance

cleaner = PromptCleaner()
validator = PromptValidator()

# Experiment constants (compute-matched to HFPO 10 generations)
GA_POPULATION_SIZE = config.GA_POPULATION_SIZE        # 20
GA_GENERATIONS = 10                                   # 10 generations (not 20)
ELITE_COUNT = 2
EVALUATION_SUBSET_SIZE = config.EVALUATION_SUBSET_SIZE  # 100
RANDOM_SEED = config.RANDOM_SEED                        # 51
HOSPITAL_DATASET_NAMES = ("medqa", "pubmedqa", "medmcqa")
DATASET_SPLIT = config.DATASET_SPLIT                    # "train"
TASK_DESCRIPTION = (
    "Answer multiple-choice medical licensing exam questions correctly, "
    "choosing exactly one option."
)
OUTPUT_CSV = "results/random_search_results.csv"
SEED_MUTATION_STRATEGIES = [
    "Chain of Thought",
    "Trigger Chain of Thought",
    "Self Consistency",
    "Tree of Thoughts",
    "Metacognitive Prompting",
    "Uncertainty-Based Prompting",
    "Role-Based Prompting",
    "Guided Prompting",
]

OFFSPRING_PER_GENERATION = GA_POPULATION_SIZE - ELITE_COUNT  # 18
TOTAL_CANDIDATES = GA_POPULATION_SIZE + OFFSPRING_PER_GENERATION * (GA_GENERATIONS - 1)  # 182


def _build_hospitals(model, tokenizer):
    """Build hospital clients with the SAME fixed 100-sample subsets (RANDOM_SEED=51)."""
    hospitals = []
    rng = random.Random(RANDOM_SEED)

    for dataset_name in HOSPITAL_DATASET_NAMES:
        dataset = load_dataset(dataset_name, split=DATASET_SPLIT)

        if len(dataset) > EVALUATION_SUBSET_SIZE:
            indices = rng.sample(range(len(dataset)), EVALUATION_SUBSET_SIZE)
            if hasattr(dataset, "select"):
                dataset = dataset.select(indices)
            else:
                dataset = [dataset[i] for i in indices]

        print(f"{dataset_name}: using {len(dataset)} evaluation samples (seed={RANDOM_SEED})")

        hospitals.append(
            HospitalClient(
                hospital_id=dataset_name,
                dataset=dataset,
                evaluator=QwenEvaluator(model=model, tokenizer=tokenizer),
            )
        )

    return hospitals


def _build_prompt_generator(llm, lineage_tracker):
    """Build the prompt generator pipeline (same as HFPO)."""
    template_builder = PromptTemplateBuilder(mutation_manager=None)

    prompt_generator = PromptGenerator(
        llm=llm,
        template_builder=template_builder,
        candidate_parser=CandidateParser(),
        similarity_selector=PromptSimilaritySelector(),
        cleaner=cleaner,
        validator=validator,
        lineage_tracker=lineage_tracker,
    )
    return prompt_generator


def _generate_random_candidate(
    candidate_index: int,
    existing_prompt_texts: set[str],
    prompt_generator: PromptGenerator,
    lineage_tracker: LineageTracker,
) -> PromptCandidate:
    """Generate one random prompt candidate without fitness-guided selection."""
    seed_prompt_text = random.choice(SEED_PROMPTS)
    mutation_strategy = random.choice(SEED_MUTATION_STRATEGIES)

    parent = PromptCandidate(
        id=str(uuid.uuid4()),
        text=seed_prompt_text,
        generation=0,
        parent_ids=[],
        ancestry_ids=[],
        origin="seed",
        fitness=None,
    )

    # Register parent in lineage tracker so build_ancestry can find it
    lineage_tracker.register(parent)

    request = PromptGenerationRequest(
        parent_a=parent,
        parent_b=None,
        generation=0,
        task_description=TASK_DESCRIPTION,
        temperature=config.TEMPERATURE,
        existing_prompt_texts=existing_prompt_texts,
        parent_a_performance=None,
        parent_b_performance=None,
        mutation_operator=mutation_strategy,
    )

    result = prompt_generator.generate(request)
    candidate = result.candidate
    candidate.metadata["candidate_index"] = candidate_index
    candidate.metadata["seed_prompt"] = seed_prompt_text
    candidate.metadata["mutation_strategy"] = mutation_strategy
    candidate.origin = "random_search"

    return candidate


def main():
    parser = argparse.ArgumentParser(description="Random search baseline (compute-matched to 10-gen HFPO)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of candidates to generate/evaluate (for smoke testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint instead of starting fresh",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N candidates (default: 1)",
    )
    args = parser.parse_args()

    limit = args.limit if args.limit is not None else TOTAL_CANDIDATES
    print(f"Random search baseline: {limit} candidates (full run = {TOTAL_CANDIDATES})")
    print(f"Config: pop_size={GA_POPULATION_SIZE}, generations={GA_GENERATIONS}, "
          f"elite={ELITE_COUNT}, offspring/gen={OFFSPRING_PER_GENERATION}, "
          f"subset_size={EVALUATION_SUBSET_SIZE}, seed={RANDOM_SEED}")

    random.seed(RANDOM_SEED)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = results_dir / "random_search_checkpoint.json"
    results_path = results_dir / "random_search_results.csv"

    # Load existing state if resuming
    start_idx = 0
    existing_prompt_texts = set()
    results_data = []
    best_so_far = 0.0
    candidates = []

    if args.resume and checkpoint_path.exists():
        import json
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            state = json.load(f)
        start_idx = state.get("next_candidate_index", 0)
        existing_prompt_texts = set(state.get("existing_prompt_texts", []))
        results_data = state.get("results_data", [])
        best_so_far = state.get("best_so_far", 0.0)
        candidates = state.get("candidates", [])
        print(f"  Resuming from candidate {start_idx + 1}/{limit}")
        print(f"  Already evaluated: {len(results_data)} candidates")
        print(f"  Best so far: {best_so_far:.4f}")
    elif results_path.exists():
        # Load results CSV to get existing state (even without explicit resume)
        print(f"Found existing results at {results_path}, loading...")
        with open(results_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["candidate_index"])
                if idx > start_idx:
                    start_idx = idx
                existing_prompt_texts.add(row["prompt_text"])
                results_data.append({
                    "idx": idx,
                    "text": row["prompt_text"],
                    "medqa_score": float(row["medqa_score"]),
                    "pubmedqa_score": float(row["pubmedqa_score"]),
                    "medmcqa_score": float(row["medmcqa_score"]),
                    "fitness": float(row["fitness"]),
                    "best_fitness_so_far": float(row["best_fitness_so_far"]),
                })
                best_so_far = max(best_so_far, float(row["best_fitness_so_far"]))
        print(f"  Loaded {len(results_data)} existing results")
        print(f"  Best so far: {best_so_far:.4f}")
        print(f"  Will continue from candidate {start_idx + 1}")

    print("=" * 80)
    print("Loading model...")
    model, tokenizer = load_model()

    llm = _QwenReasoningLLM(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.MAX_NEW_TOKENS,
    )

    hospitals = _build_hospitals(model, tokenizer)
    aggregator = Aggregator()
    evaluation_cache = EvaluationCache()
    lineage_tracker = LineageTracker()

    federated_server = FederatedServer(
        hospitals=hospitals,
        aggregator=aggregator,
        evaluation_cache=evaluation_cache,
        lineage_tracker=lineage_tracker,
        model_name=config.DEFAULT_MODEL,
        dataset_name="+".join(HOSPITAL_DATASET_NAMES),
        evaluation_version="v1",
        diagnostics_path=Path("results/random_search_diagnostics.jsonl"),
    )

    # Batch size for evaluation - evaluate multiple candidates at once for GPU parallelism
    BATCH_SIZE = 5

    prompt_generator = _build_prompt_generator(llm, lineage_tracker)

    start_time = time.perf_counter()

    print(f"\n{'=' * 80}")
    print(f"Generating and evaluating {limit} random candidates (batch size: {BATCH_SIZE})...")
    print(f"{'=' * 80}")

    def save_checkpoint(next_idx):
        import json
        state = {
            "next_candidate_index": next_idx,
            "existing_prompt_texts": list(existing_prompt_texts),
            "results_data": results_data,
            "best_so_far": best_so_far,
            "candidates": [
                {"id": c.id, "text": c.text, "generation": c.generation, "origin": c.origin}
                for c in candidates
            ],
        }
        with open(checkpoint_path, "w") as f:
            json.dump(state, f)

    # Generate and evaluate in batches
    batch = []
    for i in range(start_idx, limit):
        candidate_idx = i + 1
        gen_start = time.perf_counter()

        candidate = _generate_random_candidate(
            candidate_index=candidate_idx,
            existing_prompt_texts=existing_prompt_texts,
            prompt_generator=prompt_generator,
            lineage_tracker=lineage_tracker,
        )

        existing_prompt_texts.add(candidate.text)
        candidates.append(candidate)
        batch.append(candidate)

        print(f"Candidate {candidate_idx}/{limit}: generated")

        # Evaluate batch when full or at end
        if len(batch) >= BATCH_SIZE or candidate_idx == limit:
            print(f"Evaluating batch of {len(batch)} candidates...")
            eval_start = time.perf_counter()
            evaluated = federated_server.evaluate_population(batch)
            eval_time = time.perf_counter() - eval_start
            print(f"Batch evaluated in {eval_time:.1f}s ({eval_time/len(batch):.1f}s/candidate)")

            for j, candidate in enumerate(evaluated):
                idx = candidate_idx - len(batch) + 1 + j
                fitness = candidate.fitness
                scores = fitness.scores
                medqa_score = scores.get("medqa", 0.0)
                pubmedqa_score = scores.get("pubmedqa", 0.0)
                medmcqa_score = scores.get("medmcqa", 0.0)
                fitness_value = fitness.average()

                best_so_far = max(best_so_far, fitness_value)

                total_elapsed = time.perf_counter() - start_time
                avg_time = total_elapsed / (idx - start_idx + 1) if idx > start_idx else 0
                remaining = limit - idx
                eta = avg_time * remaining if idx > start_idx else 0

                print(f"  Candidate {idx}/{limit} | "
                      f"fitness={fitness_value:.4f} | "
                      f"best_so_far={best_so_far:.4f} | "
                      f"medqa={medqa_score:.4f} pubmedqa={pubmedqa_score:.4f} medmcqa={medmcqa_score:.4f} | "
                      f"batch={eval_time:.1f}s | ETA={eta:.1f}s", flush=True)

                results_data.append({
                    "idx": idx,
                    "text": candidate.text,
                    "medqa_score": medqa_score,
                    "pubmedqa_score": pubmedqa_score,
                    "medmcqa_score": medmcqa_score,
                    "fitness": fitness_value,
                    "best_fitness_so_far": best_so_far,
                })

            # Save checkpoint and CSV after each batch
            save_checkpoint(candidate_idx)
            with open(results_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "candidate_index", "prompt_text", "medqa_score", "pubmedqa_score",
                    "medmcqa_score", "fitness", "best_fitness_so_far"
                ])
                for row in results_data:
                    writer.writerow([
                        row["idx"],
                        row["text"],
                        f"{row['medqa_score']:.4f}",
                        f"{row['pubmedqa_score']:.4f}",
                        f"{row['medmcqa_score']:.4f}",
                        f"{row['fitness']:.4f}",
                        f"{row['best_fitness_so_far']:.4f}",
                    ])

            if candidate_idx % 10 == 0 or candidate_idx == limit:
                print(f"  Progress: {candidate_idx}/{limit} ({100*candidate_idx/limit:.1f}%) "
                      f"| Best so far: {best_so_far:.4f} | ETA: {eta:.1f}s", flush=True)

            batch = []

    print(f"Results written to {OUTPUT_CSV}")
    print(f"\nRandom search complete ({limit} candidates)")
    print(f"Best fitness found: {best_so_far:.4f}")
    best_idx = max(range(len(results_data)), key=lambda i: results_data[i]["fitness"]) + 1
    print(f"Found at candidate index: {best_idx}")
    print(f"Final best_fitness_so_far: {best_so_far:.4f}")
    print(f"  (Reference: HFPO 10-gen best ~0.73)")
    print(f"  Note: 182 candidates = compute-matched to 10 HFPO generations "
          f"(gen-0 seed population + 9 subsequent generations of 18 offspring each), "
          f"based on the observation that HFPO's fitness plateaus around generation 5.")


class _QwenReasoningLLM:
    """Adapter for Qwen model to ReasoningLLM protocol."""

    __slots__ = ("_model", "_tokenizer", "_max_new_tokens")

    def __init__(self, model, tokenizer, max_new_tokens: int):
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens

        print("=" * 80)
        print("Random Search Baseline")
        print("=" * 80)
        print(f"Model: {config.DEFAULT_MODEL}")
        print(f"Population Size: {GA_POPULATION_SIZE}")
        print(f"Generations: {GA_GENERATIONS}")
        print(f"Elite Count: {ELITE_COUNT}")
        print(f"Offspring/Gen: {OFFSPRING_PER_GENERATION}")
        print(f"Total Candidates: {TOTAL_CANDIDATES}")
        print(f"Eval Subset Size: {EVALUATION_SUBSET_SIZE}")
        print(f"Random Seed: {RANDOM_SEED}")
        print("=" * 80)

    def generate(self, prompt: str, temperature: float) -> str:
        messages = [{"role": "user", "content": prompt}]
        chat_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(chat_text, return_tensors="pt").to(self._model.device)

        generate_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature

        output_ids = self._model.generate(**inputs, **generate_kwargs)
        new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_token_ids, skip_special_tokens=True)


if __name__ == "__main__":
    main()