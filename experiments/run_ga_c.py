"""
experiments/run_ga.py

Milestone 5 experiment: Genetic Algorithm on MedQA.

Loads the model, takes the first 20 MedQA samples, runs the GA
for 5 generations, and prints per-generation results plus the
final best prompt and its accuracy.

Usage
-----
    python -m experiments.run_ga
"""

from src.models.loader import load_model
from src.data.loader import load_dataset
from src.ga.genetic_algorithm import GeneticAlgorithm


# ── baseline prompts from Milestone 4 ────────────────────────────────────────

BASELINE_PROMPTS = [
    "You are a medical expert. Answer the following question accurately.",
    "You are a concise medical assistant. Answer with only the correct option.",
    "You are a careful clinician. Think through the question before answering.",
    "You are a medical board examiner. Select the single best answer.",
    "You are a strict medical professional. Only respond with the answer letter.",
]

# ── GA hyperparameters ────────────────────────────────────────────────────────

POPULATION_SIZE = 5
GENERATIONS     = 5
MUTATION_RATE   = 0.3
CROSSOVER_RATE  = 0.7
TOURNAMENT_SIZE = 3
DATASET_SUBSET  = 20


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("FedGAPrompt — Milestone 5: Genetic Algorithm")
    print("=" * 72)

    # 1. Load model.
    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.")

    # 2. Load MedQA and take the first N samples.
    print(f"\nLoading MedQA (first {DATASET_SUBSET} samples)...")
    full_dataset = load_dataset("medqa")
    dataset = full_dataset[:DATASET_SUBSET]
    print(f"Dataset ready: {len(dataset)} samples.")

    # 3. Initialise GA.
    ga = GeneticAlgorithm(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        tournament_size=TOURNAMENT_SIZE,
    )

    # 4. Run.
    print("\nStarting GA...\n")
    print("-" * 72)

    best = ga.run(baseline_prompts=BASELINE_PROMPTS)

    # 5. Print final results.
    print("-" * 72)
    print("\nGA complete.\n")
    print("Generation history:")
    print("-" * 72)

    for result in ga.history:
        print(
            f"  Gen {result.generation:>2} | "
            f"Best: {result.best_fitness:.4f} | "
            f"Avg: {result.avg_fitness:.4f}"
        )

    print("-" * 72)
    print(f"\nFinal Best Prompt:\n  {best.prompt!r}")
    print(f"\nFinal Accuracy : {best.fitness:.4f}  "
          f"({int(best.fitness * DATASET_SUBSET)}/{DATASET_SUBSET} correct)")
    print("=" * 72)


if __name__ == "__main__":
    main()