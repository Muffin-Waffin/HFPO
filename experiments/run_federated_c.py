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

from src.models.loader import load_model
from src.data.loader import load_dataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer


# ── baseline prompts from Milestone 4 ────────────────────────────────────────

BASELINE_PROMPTS = [
    "You are a medical expert. Answer the following question accurately.",
    "You are a concise medical assistant. Answer with only the correct option.",
    "You are a careful clinician. Think through the question before answering.",
    "You are a medical board examiner. Select the single best answer.",
    "You are a strict medical professional. Only respond with the answer letter.",
]

# ── experiment config ─────────────────────────────────────────────────────────

DATASET_SIZE       = 60   # total samples — split evenly across 3 clients
NUM_CLIENTS        = 3
COMMUNICATION_ROUNDS = 3


# ── helpers ───────────────────────────────────────────────────────────────────

def partition_dataset(
    dataset: list[dict],
    num_clients: int,
) -> list[list[dict]]:
    """
    Split a dataset into equal partitions, one per client.

    If the dataset does not divide evenly, the last partition
    may be slightly smaller.

    Parameters
    ----------
    dataset     : Full list of standardised sample dicts.
    num_clients : Number of partitions to create.

    Returns
    -------
    List of dataset partitions.
    """
    size = len(dataset) // num_clients
    return [
        dataset[i * size: (i + 1) * size]
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
    dataset = full_dataset[:DATASET_SIZE]
    print(f"Dataset ready: {len(dataset)} samples.")

    # 3. Split into partitions.
    partitions = partition_dataset(dataset, NUM_CLIENTS)

    for i, partition in enumerate(partitions):
        print(f"  Client {i + 1}: {len(partition)} samples")

    # 4. Create clients.
    clients = [
        FederatedClient(
            client_id=f"Hospital_{i + 1}",
            dataset=partitions[i],
            model=model,
            tokenizer=tokenizer,
        )
        for i in range(NUM_CLIENTS)
    ]

    # 5. Create server.
    server = FederatedServer(
        clients=clients,
        baseline_prompts=BASELINE_PROMPTS,
    )

    # 6. Run federated optimisation.
    print(f"\nRunning {COMMUNICATION_ROUNDS} communication rounds "
          f"across {NUM_CLIENTS} clients...\n")

    best = server.run(rounds=COMMUNICATION_ROUNDS)

    # 7. Final summary.
    print("=" * 64)
    print("  Final Results")
    print("=" * 64)

    print("\nRound history:")
    for i, result in enumerate(server.round_history, start=1):
        print(f"  Round {i} | Best Fitness: {result.fitness:.4f}")

    print(f"\nFinal Best Prompt  : {best.prompt!r}")
    print(f"Final Best Fitness : {best.fitness:.4f} "
          f"({int(best.fitness * DATASET_SIZE)}/{DATASET_SIZE} correct)")
    print("=" * 64)


if __name__ == "__main__":
    main()