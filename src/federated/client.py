"""
src/federated/client.py

Federated client for FedGAPrompt.

Each client represents one hospital institution.
It owns a local dataset partition and runs the Genetic Algorithm
independently without knowledge of other clients.

Designed so the communication layer can be replaced by Flower
in Milestone 6B without changing any GA logic.
"""

from datasets import Dataset

from src.ga.genetic_algorithm import GeneticAlgorithm
from src.ga.population import PromptIndividual
from src.evaluation.scorer import score_prompt


class FederatedClient:
    """
    A single federated institution that optimises prompts locally.

    The client never shares its dataset — only its best PromptIndividual
    is returned to the server after local training.

    Parameters
    ----------
    client_id : Human-readable identifier (e.g. "hospital_a").
    dataset   : Local HuggingFace Dataset partition.
    model      : Shared HuggingFace model instance.
    tokenizer  : Corresponding tokenizer.
    ga_kwargs  : Hyperparameters passed to the GeneticAlgorithm.
    """

    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        model,
        tokenizer,
        ga_kwargs: dict,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer

        self._ga = GeneticAlgorithm(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            **ga_kwargs,
        )

    def train(
        self,
        baseline_prompts: list[str],
    ) -> PromptIndividual:
        """
        Run the local Genetic Algorithm and return the best prompt.

        Parameters
        ----------
        baseline_prompts : Seed prompts received from the server.

        Returns
        -------
        Best PromptIndividual discovered on this client.
        """
        return self._ga.run(
            baseline_prompts=baseline_prompts,
        )

    def evaluate(
        self,
        prompt: str,
    ) -> float:
        """
        Evaluate a system prompt on this client's local dataset.

        Parameters
        ----------
        prompt : System prompt string.

        Returns
        -------
        Accuracy score in [0, 1].
        """
        return score_prompt(
            self.model,
            self.tokenizer,
            prompt,
            self.dataset,
        )