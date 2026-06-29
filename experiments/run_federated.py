"""
experiments/run_federated.py

Milestone 6A experiment: Federated Simulation on MedQA.

Simulates 3 hospital institutions each running the Genetic Algorithm
locally on their own data partition, then aggregating results via a
central server — all on one machine, no Flower yet.

Usage
-----
    python -m experiments.run_federated
"""

from datasets import Dataset

from configs.config import (
    GA_POPULATION_SIZE,
    GA_GENERATIONS,
    GA_MUTATION_RATE,
    GA_CROSSOVER_RATE,
    GA_TOURNAMENT_SIZE,
)

from src.models.loader import load_model
from src.data.loader import load_dataset
from src.prompts.baseline import BASELINE_PROMPTS
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer


# ── experiment config ─────────────────────────────────────────────────────────

DATASET_SIZE = 60
NUM_CLIENTS = 3
COMMUNICATION_ROUNDS = 3


# ── helpers ───────────────────────────────────────────────────────────────────

def partition_dataset(
    dataset: Dataset,
    num_clients: int,
) -> list[Dataset]:
    """
    Split a dataset into equal partitions, one per client.

    Parameters
    ----------
    dataset     : Full HuggingFace Dataset.
    num_clients : Number of partitions to create.

    Returns
    -------
    List of Dataset partitions.
    """

    size = len(dataset) // num_clients

    return [
        dataset.select(
            range(i * size, (i + 1) * size)
        )
        for i in range(num_clients)
    ]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print("  FedGAPrompt — Milestone 6A: Federated Simulation")
    print("=" * 64)

    # 1. Load model.
    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.")

    # 2. Load MedQA.
    print(f"\nLoading MedQA (first {DATASET_SIZE} samples)...")
    full_dataset = load_dataset("medqa")
    dataset = full_dataset.select(range(DATASET_SIZE))
    print(f"Dataset ready: {len(dataset)} samples.")

    # 3. Split into partitions.
    partitions = partition_dataset(dataset, NUM_CLIENTS)

    for i, partition in enumerate(partitions):
        print(f"  Client {i + 1}: {len(partition)} samples")

    # 4. GA configuration.
    ga_kwargs = {
        "population_size": GA_POPULATION_SIZE,
        "generations": GA_GENERATIONS,
        "mutation_rate": GA_MUTATION_RATE,
        "crossover_rate": GA_CROSSOVER_RATE,
        "tournament_size": GA_TOURNAMENT_SIZE,
    }

    # 5. Create clients.
    clients = [
        FederatedClient(
            client_id=f"Hospital_{i + 1}",
            dataset=partitions[i],
            model=model,
            tokenizer=tokenizer,
            ga_kwargs=ga_kwargs,
        )
        for i in range(NUM_CLIENTS)
    ]

    # 6. Create server.
    server = FederatedServer(
        clients=clients,
        baseline_prompts=list(BASELINE_PROMPTS.values()),
    )

    # 7. Run federated optimisation.
    print(
        f"\nRunning {COMMUNICATION_ROUNDS} communication rounds "
        f"across {NUM_CLIENTS} clients...\n"
    )

    best = server.run(rounds=COMMUNICATION_ROUNDS)

    # 8. Final summary.
    print("=" * 64)
    print("  Final Results")
    print("=" * 64)

    print("\nRound history:")

    for result in server.round_history:
        print(
            f"  Round {result.round_number} | "
            f"Best Fitness: {result.best_fitness:.4f}"
        )

    print(f"\nFinal Best Prompt:\n{best.prompt}")
    print(f"\nFinal Best Fitness : {best.fitness:.4f}")

    print("=" * 64)


if __name__ == "__main__":
    main()