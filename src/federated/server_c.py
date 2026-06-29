"""
src/federated/server.py

Federated server for FedGAPrompt simulation.

Coordinates communication rounds between clients.
Never accesses client datasets directly — only receives
PromptIndividuals from clients after local training.

Designed so Flower can replace the communication loop in
Milestone 6B without changing client or strategy logic.
"""

from src.federated.client import FederatedClient
from src.federated.strategy import aggregate_best_prompt
from src.ga.population import PromptIndividual


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
        self.round_history: list[PromptIndividual] = []

    def run_round(self, round_number: int) -> PromptIndividual:
        """
        Execute one communication round.

        Steps
        -----
        1. Each client runs its local GA on the current baseline prompts.
        2. Server collects the best PromptIndividual from each client.
        3. Aggregation strategy selects the global best.
        4. Results are printed and the global best is updated.

        Parameters
        ----------
        round_number : 1-based round index for display.

        Returns
        -------
        The globally best PromptIndividual after this round.
        """
        print(f"\n{'=' * 64}")
        print(f"  Round {round_number}")
        print(f"{'=' * 64}")

        # 1 & 2. Collect best prompt from each client.
        client_results: list[PromptIndividual] = []

        for client in self.clients:
            best = client.train(baseline_prompts=self.baseline_prompts)
            client_results.append(best)

            print(f"\n  {client.client_id}")
            print(f"  Fitness : {best.fitness:.4f}")
            print(f"  Prompt  : {best.prompt!r}")

        # 3. Aggregate.
        round_best = aggregate_best_prompt(client_results)

        # 4. Update global best across all rounds.
        if self.global_best is None or round_best.fitness > self.global_best.fitness:
            self.global_best = PromptIndividual(
                prompt=round_best.prompt,
                fitness=round_best.fitness,
            )

        # Use the global best as the seed for the next round.
        self.baseline_prompts = [self.global_best.prompt] + self.baseline_prompts

        self.round_history.append(round_best)

        print(f"\n  Global Best (Round {round_number})")
        print(f"  Fitness : {self.global_best.fitness:.4f}")
        print(f"  Prompt  : {self.global_best.prompt!r}")

        return round_best

    def run(self, rounds: int) -> PromptIndividual:
        """
        Run the full federated optimisation loop for N rounds.

        Parameters
        ----------
        rounds : Number of communication rounds to execute.

        Returns
        -------
        The globally best PromptIndividual found across all rounds.
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