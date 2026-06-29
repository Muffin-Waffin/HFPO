"""
src/federated/client.py

Federated client for FedGAPrompt.

Each client represents one hospital institution.
It owns a local dataset partition and runs the Genetic Algorithm
independently without knowledge of other clients.

Designed so the communication layer can be replaced by Flower
in Milestone 6B without changing any GA logic.
"""

from src.ga.genetic_algorithm import GeneticAlgorithm
from src.ga.population import PromptIndividual
from src.evaluation.scorer import score_prompt
from configs.config import (
    GA_POPULATION_SIZE,
    GA_GENERATIONS,
    GA_MUTATION_RATE,
    GA_CROSSOVER_RATE,
    GA_TOURNAMENT_SIZE,
)


class FederatedClient:
    """
    A single federated institution that optimises prompts locally.

    The client never shares its dataset — only its best PromptIndividual
    is returned to the server after local training.

    Parameters
    ----------
    client_id  : Human-readable identifier e.g. "hospital_a".
    dataset    : Local data partition (list of standardised sample dicts).
    model      : Shared HuggingFace model instance.
    tokenizer  : Corresponding tokenizer.
    """

    def __init__(
        self,
        client_id: str,
        dataset: list[dict],
        model,
        tokenizer,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer

        self._ga = GeneticAlgorithm(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            population_size=GA_POPULATION_SIZE,
            generations=GA_GENERATIONS,
            mutation_rate=GA_MUTATION_RATE,
            crossover_rate=GA_CROSSOVER_RATE,
            tournament_size=GA_TOURNAMENT_SIZE,
        )

    def train(self, baseline_prompts: list[str]) -> PromptIndividual:
        """
        Run the local Genetic Algorithm and return the best prompt found.

        Parameters
        ----------
        baseline_prompts : Seed prompts passed in from the server each round.

        Returns
        -------
        The PromptIndividual with the highest local fitness.
        """
        print(f"  [{self.client_id}] Starting local GA "
              f"({len(self.dataset)} samples, "
              f"{self._ga.generations} generations)...")

        best = self._ga.run(baseline_prompts=baseline_prompts)

        print(f"  [{self.client_id}] Done. "
              f"Best fitness: {best.fitness:.4f} | Prompt: {best.prompt!r}")

        return best

    def evaluate(self, prompt: str) -> float:
        """
        Evaluate a given prompt on this client's local dataset.

        Intended for use in later milestones (membership inference,
        global prompt validation per client).

        Parameters
        ----------
        prompt : System prompt string to evaluate.

        Returns
        -------
        Accuracy score in [0, 1].
        """
        return score_prompt(self.model, self.tokenizer, prompt, self.dataset)