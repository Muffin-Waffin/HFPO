"""Executable entry point for one complete HFPO evolutionary optimization
experiment.

Running ``python -m experiments.run_evolution`` loads the reasoning LLM
and the medical datasets, constructs the full HFPO dependency graph
(hospitals, federated server, prompt generator, tournament selector,
elitism, output manager), builds the generation-0 population from
``src.core.seed_prompts.SEED_PROMPTS``, constructs an ``EvolutionEngine``,
and runs it to completion.

This module contains no genetic algorithm logic, no evaluation logic, no
aggregation logic, and no fitness computation of its own; every one of
those responsibilities is delegated to the existing HFPO components. Its
only original code is the small amount of "driver" glue that the rest of
the codebase does not already provide:

    * ``_QwenReasoningLLM``, an adapter satisfying the ``ReasoningLLM``
      protocol on top of the raw ``(model, tokenizer)`` pair returned by
      ``src.llms.loader.load_model()``.
    * ``_build_seed_population()``, which converts
      ``SEED_PROMPTS`` into generation-0 ``PromptCandidate`` objects.

Experiment-level settings that are not present in ``configs/config.py``
(the target datasets, elite count, evaluation-cache version, output
directory, and task description) are defined as module-level constants
below rather than by editing ``configs/config.py``.
"""

from __future__ import annotations

import argparse
import uuid
from typing import Any

from configs import config
from src.core.lineage_tracker import LineageTracker
from src.core.evaluation_cache import EvaluationCache
from src.core.population import Population
from src.core.prompt_candidate import PromptCandidate
from src.core.seed_prompts import SEED_PROMPTS
from src.data.loader import load_dataset
from src.evaluation.qwen_evaluator import QwenEvaluator
from src.evolution.elitism import Elitism
from src.evolution.evolution_engine import EvolutionEngine
from src.evolution.output_manager import OutputManager
from src.evolution.prompt_generator.cleaner import PromptCleaner
from src.evolution.prompt_generator.generator import PromptGenerator
from src.evolution.prompt_generator.templates import PromptTemplateBuilder
from src.evolution.prompt_generator.validator import PromptValidator
from src.evolution.tournament_selector import TournamentSelector
from src.federated.aggregator import Aggregator
from src.federated.federated_server import FederatedServer
from src.federated.hospital_client import HospitalClient
from src.llms.loader import load_model
import random


# ---------------------------------------------------------------------------
# Experiment-level configuration.
#
# configs/config.py defines model and GA hyperparameters (MODEL_NAME,
# MAX_NEW_TOKENS, TEMPERATURE, GA_POPULATION_SIZE, GA_GENERATIONS,
# GA_TOURNAMENT_SIZE). It does not define which datasets back each
# hospital, how many hospitals participate, the elite count, the
# evaluation-cache version label, the output directory, a fixed
# tournament-selection random seed, or the task description handed to
# every PromptGenerationRequest -- those are experiment-run choices
# rather than model/GA hyperparameters, so they are defined here instead
# of by editing configs/config.py.
#
# HFPO simulates one hospital per available medical dataset
# (src.data.loader.load_dataset supports exactly "medqa", "medmcqa", and
# "pubmedqa"), which is what makes the federation heterogeneous: each
# hospital evaluates candidate prompts against a distinct question
# distribution rather than a shard of one shared dataset.
# ---------------------------------------------------------------------------
HOSPITAL_DATASET_NAMES: tuple[str, ...] = ("medqa", "pubmedqa", "medmcqa")
DATASET_SPLIT: str = "train"
FEDERATION_DATASET_LABEL: str = "+".join(HOSPITAL_DATASET_NAMES)
ELITE_COUNT: int = 2
EVALUATION_VERSION: str = "v1"
OUTPUT_DIRECTORY: str = "results/hfpo_run"
TOURNAMENT_RANDOM_SEED: int = 42
TASK_DESCRIPTION: str = (
    "Answer multiple-choice medical licensing exam questions correctly, "
    "choosing exactly one option."
)


class _QwenReasoningLLM:
    """Adapts a loaded Qwen model and tokenizer to the ReasoningLLM
    protocol.

    ``src.llms.loader.load_model()`` returns a raw ``(model, tokenizer)``
    pair; it does not itself expose a ``generate(prompt, temperature) ->
    str`` method, which is what ``PromptGenerator``'s ``ReasoningLLM``
    collaborator requires. This adapter performs exactly the
    tokenize/generate/decode steps needed to bridge the two, and nothing
    else: no prompt cleaning, no validation, no candidate construction.

    Attributes:
        _model: The loaded causal language model.
        _tokenizer: The tokenizer paired with ``_model``.
        _max_new_tokens: Maximum number of new tokens to generate per
            call.
    """

    __slots__ = ("_model", "_tokenizer", "_max_new_tokens")

    def __init__(self, model: Any, tokenizer: Any, max_new_tokens: int) -> None:
        """Initializes the adapter with a model, tokenizer, and generation
        length.

        Args:
            model: The loaded causal language model.
            tokenizer: The tokenizer paired with ``model``.
            max_new_tokens: Maximum number of new tokens to generate per
                call.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens

        print("=" * 80)
        print("HFPO Evolution")
        print("=" * 80)
        print(f"Model: {config.MODEL_NAME}")
        print(f"Population Size: {config.GA_POPULATION_SIZE}")
        print(f"Generations: {config.GA_GENERATIONS}")
        print(f"Tournament Size: {config.GA_TOURNAMENT_SIZE}")
        print(f"Mutation Rate: {config.GA_MUTATION_RATE}")
        print(f"Crossover Rate: {config.GA_CROSSOVER_RATE}")
        print("=" * 80)

        # print(f"Loaded {dataset_name}: {len(dataset)} samples")

    def generate(self, prompt: str, temperature: float) -> str:
        """Generates text from a single-turn chat-formatted prompt.

        Applies the tokenizer's chat template with
        ``enable_thinking=False`` to suppress Qwen3's reasoning-trace
        blocks, since ``PromptGenerator`` expects the raw output to
        already be the candidate prompt text rather than a
        ``<think>...</think>`` block.

        Args:
            prompt: The fully formatted instruction to present to the
                model.
            temperature: Sampling temperature. A value of 0 disables
                sampling.

        Returns:
            The newly generated text, decoded with special tokens
            removed.
        """
        messages = [{"role": "user", "content": prompt}]
        chat_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(chat_text, return_tensors="pt").to(
            self._model.device
        )

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature

        output_ids = self._model.generate(**inputs, **generate_kwargs)
        new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_token_ids, skip_special_tokens=True)


def _build_seed_population(max_population_size: int) -> Population:
    """Converts ``SEED_PROMPTS`` into the generation-0 Population.

    Each seed prompt becomes one generation-0 ``PromptCandidate`` with
    ``origin="seed"``, empty ``parent_ids``, empty ``ancestry_ids``, a
    freshly generated UUID, and ``fitness=None``.

    Args:
        max_population_size: The maximum population size to configure the
            resulting Population with.

    Returns:
        A new Population containing one PromptCandidate per entry in
        SEED_PROMPTS.
    """
    # seed_candidates = [
    #     PromptCandidate(
    #         id=str(uuid.uuid4()),
    #         text=prompt_text,
    #         generation=0,
    #         parent_ids=[],
    #         ancestry_ids=[],
    #         origin="seed",
    #         fitness=None,
    #     )
    #     for prompt_text in SEED_PROMPTS
    # ]

    seed_candidates = [
    PromptCandidate(
        id=str(uuid.uuid4()),
        text=prompt_text,
        generation=0,
        parent_ids=[],
        ancestry_ids=[],
        origin="seed",
        fitness=None,
    )
    for prompt_text in SEED_PROMPTS
    ]

    return Population(
        prompts=seed_candidates,
        generation=0,
        max_population_size=max_population_size,
    )


def main() -> None:
    """Runs one complete HFPO evolutionary optimization experiment."""
    parser = argparse.ArgumentParser(
        description="HFPO Evolutionary Prompt Optimization"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint instead of starting fresh.",
    )
    args = parser.parse_args()

    model, tokenizer = load_model()

    llm = _QwenReasoningLLM(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.MAX_NEW_TOKENS,
    )

    import random
    hospitals = []

    for dataset_name in HOSPITAL_DATASET_NAMES:
        dataset = load_dataset(dataset_name, split=DATASET_SPLIT)

        if len(dataset) > config.EVALUATION_SUBSET_SIZE:
            random.seed(config.RANDOM_SEED)

            indices = random.sample(
                range(len(dataset)),
                config.EVALUATION_SUBSET_SIZE,
            )

            if hasattr(dataset, "select"):
                dataset = dataset.select(indices)
            else:
                dataset = [dataset[i] for i in indices]

        print(f"{dataset_name}: using {len(dataset)} evaluation samples")

        hospitals.append(
            HospitalClient(
                hospital_id=dataset_name,
                dataset=dataset,
                evaluator=QwenEvaluator(
                    model=model,
                    tokenizer=tokenizer,
                ),
            )
        )

    aggregator = Aggregator()
    evaluation_cache = EvaluationCache()
    lineage_tracker = LineageTracker()

    federated_server = FederatedServer(
        hospitals=hospitals,
        aggregator=aggregator,
        evaluation_cache=evaluation_cache,
        lineage_tracker=lineage_tracker,
        model_name=config.MODEL_NAME,
        dataset_name=FEDERATION_DATASET_LABEL,
        evaluation_version=EVALUATION_VERSION,
    )

    output_manager = OutputManager(output_directory=OUTPUT_DIRECTORY)

    start_generation = 0

    if args.resume:
        latest_gen = output_manager.find_latest_population_checkpoint()
        if latest_gen is None:
            print("No checkpoint found. Starting from scratch.")
        else:
            print(f"Found checkpoint for generation {latest_gen}.")
            population = output_manager.load_population_checkpoint(latest_gen)
            print(
                f"Loaded population: {len(population)} candidates, "
                f"generation {population.generation}"
            )

            for candidate in population:
                lineage_tracker.register(candidate)

            for candidate in population:
                if candidate.fitness is not None:
                    evaluation_cache.set(
                        candidate.text,
                        config.MODEL_NAME,
                        FEDERATION_DATASET_LABEL,
                        EVALUATION_VERSION,
                        candidate.fitness,
                    )

            start_generation = population.generation
            print(f"Resuming from generation {start_generation + 1}.")
    else:
        population = _build_seed_population(config.GA_POPULATION_SIZE)
        for candidate in population:
            lineage_tracker.register(candidate)

    prompt_generator = PromptGenerator(
        llm=llm,
        template_builder=PromptTemplateBuilder(),
        cleaner=PromptCleaner(),
        validator=PromptValidator(),
        lineage_tracker=lineage_tracker,
    )

    tournament_selector = TournamentSelector(
        tournament_size=config.GA_TOURNAMENT_SIZE,
        ranking_strategy="average",
        random_seed=TOURNAMENT_RANDOM_SEED,
    )

    elitism = Elitism(
        elite_count=ELITE_COUNT,
        ranking_strategy="average",
    )

    engine = EvolutionEngine(
        population=population,
        federated_server=federated_server,
        tournament_selector=tournament_selector,
        elitism=elitism,
        prompt_generator=prompt_generator,
        output_manager=output_manager,
        num_generations=config.GA_GENERATIONS,
        task_description=TASK_DESCRIPTION,
        temperature=config.TEMPERATURE,
        mutation_rate=config.GA_MUTATION_RATE,
        crossover_rate=config.GA_CROSSOVER_RATE,
        start_generation=start_generation,
    )

    final_population = engine.run()

    evaluated_candidates = [
        candidate for candidate in final_population if candidate.fitness is not None
    ]

    print("HFPO evolutionary run complete.")
    print(f"Generations run: {config.GA_GENERATIONS}")
    print(f"Final population size: {len(final_population)}")

    if evaluated_candidates:
        best_candidate = max(
            evaluated_candidates,
            key=lambda candidate: candidate.fitness.average(),
        )
        print(
            "Best known candidate (an elite carried into the final "
            f"population): {best_candidate.id}"
        )
        print(
            f"Best known average fitness: {best_candidate.fitness.average():.4f}"
        )
        print("Best known prompt:")
        print(best_candidate.text)
    else:
        print(
            "No evaluated candidates remain in the final population; see "
            "per-generation snapshots and best-prompt reports under "
            f"'{OUTPUT_DIRECTORY}' for the full evolutionary history."
        )

    print(
        "Full per-generation snapshots and best-prompt reports were "
        f"written under '{OUTPUT_DIRECTORY}'."
    )


if __name__ == "__main__":
    main()
