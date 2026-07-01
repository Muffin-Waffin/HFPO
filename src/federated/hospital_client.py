"""Isolated hospital evaluation node for HFPO (Heterogeneous Federated
Prompt Optimization).

This module defines :class:`HospitalClient`, which represents one hospital
participating in HFPO. Each `HospitalClient` owns its own private dataset
and a private `Evaluator`, and uses them to evaluate candidate prompts
entirely on its own. Hospitals never communicate with each other, never
see each other's data, and never perform any genetic algorithm operation
such as selection, mutation, crossover, ranking, or aggregation. A
`HospitalClient` receives prompt text from the Federated Server, scores it
locally by delegating to its `Evaluator` for every sample in its dataset,
and returns only an `EvaluationRecord`. Raw patient data, model internals,
and per-question predictions never leave the `HospitalClient`. This models
the privacy constraints of real federated healthcare systems, where each
institution's data must remain local while still contributing to a shared
optimization process.

`HospitalClient` deliberately knows nothing about how scoring is performed.
It depends only on the `Evaluator` abstraction: an object that exposes a
`score_sample(prompt_text, sample) -> int` method, where the returned
value is 1 if the sample was answered correctly and 0 otherwise. Any
model, tokenizer, or prompting strategy is encapsulated entirely within
the `Evaluator` implementation supplied to the hospital. This keeps
`HospitalClient` stable as new models or evaluation pipelines are
introduced, since only the `Evaluator` implementation changes, never
`HospitalClient` itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any,Protocol

from src.evaluation.evaluator import Evaluator

from src.core.evaluation_record import EvaluationRecord
from src.core.prompt_candidate import PromptCandidate


class Evaluator(Protocol):
    """Interface for scoring a single prompt against a single dataset sample.

    `HospitalClient` depends only on this interface, not on any concrete
    model, tokenizer, or prompting implementation. Any object that
    implements `score_sample` with this signature can be supplied to a
    `HospitalClient`. Concrete evaluators (for BioMistral, Qwen, Meditron,
    GPT-OSS, Llama, or any future model) encapsulate their own model,
    tokenizer, and prompt-building logic behind this single method.
    """

    def score_sample(self, prompt_text: str, sample: Any) -> int:
        """Score one dataset sample against the given prompt text.

        Args:
            prompt_text: The system prompt text to evaluate.
            sample: A single sample from the hospital's dataset.

        Returns:
            1 if the sample was answered correctly, 0 otherwise.
        """
        ...


class HospitalClient:
    """Represents one isolated hospital participating in federated evaluation.

    Each `HospitalClient` owns its private dataset and a private
    `Evaluator`, and is solely responsible for evaluating candidate
    prompts against its own local data. Hospitals are completely isolated
    from one another: they never exchange information, and the only
    artifact that ever leaves a `HospitalClient` is an `EvaluationRecord`,
    which contains nothing but an aggregate score and count, never raw
    patient data or individual predictions. A `HospitalClient` performs no
    selection, mutation, crossover, ranking, or aggregation; it only
    orchestrates iteration over its local dataset, delegating the actual
    scoring of each sample to its `Evaluator`.

    `HospitalClient` knows nothing about models, tokenizers, or prompt
    construction; all of that is encapsulated inside the `Evaluator`
    implementation it is given at construction time.

    Attributes:
        hospital_id: Unique identifier for this hospital (e.g.
            "hospital_1").
        dataset: The hospital's private local dataset. Never exposed
            outside this class.
        evaluator: The `Evaluator` used to score prompts against samples
            from `dataset`. Never exposed outside this class.

    Raises:
        ValueError: If `hospital_id` is empty, or if `dataset` or
            `evaluator` is `None`.
    """

    def __init__(
        self,
        hospital_id: str,
        dataset: Any,
        evaluator: Evaluator,
    ) -> None:
        """Initialize a HospitalClient with its private evaluation resources.

        Args:
            hospital_id: Unique identifier for this hospital.
            dataset: The hospital's private local dataset.
            evaluator: The `Evaluator` used to score prompts against
                samples from `dataset`.

        Raises:
            ValueError: If `hospital_id` is empty or whitespace, or if
                `dataset` or `evaluator` is `None`.
        """
        if not hospital_id.strip():
            raise ValueError("HospitalClient hospital_id must not be empty.")

        if dataset is None:
            raise ValueError("HospitalClient dataset must not be None.")

        if evaluator is None:
            raise ValueError("HospitalClient evaluator must not be None.")

        self.hospital_id = hospital_id
        self.dataset = dataset
        self.evaluator = evaluator

    def evaluate_prompt(self, prompt: PromptCandidate) -> EvaluationRecord:
        """Evaluate one prompt against this hospital's local dataset.

        Iterates over every sample in the hospital's private dataset,
        delegating the scoring of each sample to this hospital's
        `Evaluator`, and accumulates the number of correct answers and the
        total number of samples evaluated. `HospitalClient` itself has no
        knowledge of how scoring is performed; it only orchestrates the
        iteration and bookkeeping.

        Args:
            prompt: The `PromptCandidate` to evaluate. This object is not
                modified.

        Returns:
            An `EvaluationRecord` describing this hospital's evaluation of
            the prompt.
        """
        num_correct = 0
        num_total = 0

        for sample in self.dataset:
            num_correct += self.evaluator.score_sample(prompt.text, sample)
            num_total += 1

        evaluation_score = num_correct / num_total if num_total > 0 else 0.0

        return EvaluationRecord(
            prompt_id=prompt.id,
            hospital_id=self.hospital_id,
            generation=prompt.generation,
            score=evaluation_score,
            num_correct=num_correct,
            num_total=num_total,
        )

    def evaluate_population(
        self, population: list[PromptCandidate]
    ) -> list[EvaluationRecord]:
        """Evaluate every prompt in a population against the local dataset.

        Args:
            population: The list of `PromptCandidate` objects to evaluate.

        Returns:
            A list of `EvaluationRecord` objects, one per prompt, in the
            same order as `population`.

        Raises:
            ValueError: If `population` is empty.
        """
        if not population:
            raise ValueError(
                "HospitalClient.evaluate_population received an empty "
                "population; an empty federated round usually indicates "
                "a bug upstream."
            )

        return [self.evaluate_prompt(prompt) for prompt in population]

    def dataset_size(self) -> int:
        """Return the number of samples in the hospital's local dataset.

        Returns:
            `len(self.dataset)`.
        """
        return len(self.dataset)

    def summary(self) -> dict[str, object]:
        """Return a logging-friendly summary of this hospital.

        Returns:
            A dictionary containing the hospital's ID and dataset size.
        """
        return {
            "hospital_id": self.hospital_id,
            "dataset_size": self.dataset_size(),
        }

    def __str__(self) -> str:
        """Return a concise, human-readable summary of this hospital.

        Returns:
            A string of the form:
            'HospitalClient(id="hospital_1", dataset_size=1200)'.
        """
        return (
            f'HospitalClient(\n'
            f'    id="{self.hospital_id}",\n'
            f'    dataset_size={self.dataset_size()}\n'
            f')'
        )