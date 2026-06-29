"""
src/federated/server.py

Federated server for FedGAPrompt simulation.

Coordinates communication rounds between clients.
Never accesses client datasets directly — only receives
PromptIndividuals from clients after local training.

Designed so Flower can replace the communication loop in
Milestone 6B without changing client or strategy logic.
"""

from dataclasses import dataclass

from src.federated.client import FederatedClient
from src.federated.strategy import aggregate_best_prompt
from src.ga.population import PromptIndividual


# ── round history ─────────────────────────────────────────────────────────────

@dataclass
class RoundResult:
    """
    Stores the outcome of one communication round.
    """

    round_number: int
    best_fitness: float
    best_prompt: str


# ── federated server ──────────────────────────────────────────────────────────

class FederatedServer:
    """
    Central coordinator for the federated prompt optimisation loop.

    Parameters
    ----------
    clients         : List of FederatedClient instances (one per hospital).
    baseline_prompts: Seed prompts used to initialise each client's GA.
    """

    def __init__(
        self,
        clients: list[FederatedClient],
        baseline_prompts: list[str],
    ) -> None:
        self.clients = clients
        self.baseline_prompts = baseline_prompts
        self.global_best: PromptIndividual | None = None
        self.round_history: list[RoundResult] = []

    def run_round(
        self,
        round_number: int,
    ) -> PromptIndividual:
        """
        Execute one communication round.
        """

        print(f"\n{'=' * 64}")
        print(f"  Round {round_number}")
        print(f"{'=' * 64}")

        client_results: list[PromptIndividual] = []

        # Local training
        for client in self.clients:

            best = client.train(
                baseline_prompts=self.baseline_prompts,
            )

            client_results.append(best)

            print(f"\n  {client.client_id}")
            print(f"  Fitness : {best.fitness:.4f}")
            print(f"  Prompt  : {best.prompt!r}")

        # Aggregate
        round_best = aggregate_best_prompt(client_results)

        # Update global best
        if (
            self.global_best is None
            or round_best.fitness > self.global_best.fitness
        ):
            self.global_best = round_best

        # Keep a constant-size seed population.
        # Global best + first four original prompts.
        self.baseline_prompts = [
            self.global_best.prompt,
            *self.baseline_prompts[:4],
        ]

        self.round_history.append(
            RoundResult(
                round_number=round_number,
                best_fitness=self.global_best.fitness,
                best_prompt=self.global_best.prompt,
            )
        )

        print(f"\n  Global Best")
        print(f"  Fitness : {self.global_best.fitness:.4f}")
        print(f"  Prompt  : {self.global_best.prompt!r}")

        return round_best

    def run(
        self,
        rounds: int,
    ) -> PromptIndividual:
        """
        Run multiple federated communication rounds.
        """

        for round_number in range(1, rounds + 1):
            self.run_round(round_number)

        print(f"\n{'=' * 64}")
        print("  Federated Training Complete")
        print(f"{'=' * 64}")
        print(f"  Final Best Fitness : {self.global_best.fitness:.4f}")
        print(f"  Final Best Prompt  : {self.global_best.prompt!r}")
        print(f"{'=' * 64}\n")

        return self.global_best