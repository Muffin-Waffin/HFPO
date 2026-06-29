"""
src/federated/strategy.py

Aggregation strategies for FedGAPrompt federated simulation.

Currently implements best-prompt selection — the simplest possible
aggregation: pick the globally highest-fitness prompt from all clients.

Intentionally simple for Milestone 6A. More sophisticated strategies
(prompt averaging, weighted selection, ensemble) can be added here
in later milestones without touching the server or client code.
"""

from src.ga.population import PromptIndividual


def aggregate_best_prompt(
    client_results: list[PromptIndividual],
) -> PromptIndividual:
    """
    Select the globally best prompt from all client results.

    No averaging. No merging. Simply returns the PromptIndividual
    with the highest fitness across all clients.

    Parameters
    ----------
    client_results : List of best PromptIndividuals, one per client.

    Returns
    -------
    The single PromptIndividual with the highest fitness.

    Raises
    ------
    ValueError : If client_results is empty.
    """
    if not client_results:
        raise ValueError("client_results must not be empty.")

    return max(client_results, key=lambda ind: ind.fitness)