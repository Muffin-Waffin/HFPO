"""Core data structures for the Answering Engine.

This module defines the data classes used throughout the Answering Engine
subsystem. These types are intentionally independent from the evolutionary
pipeline's data structures (``PromptCandidate``, ``FitnessVector``,
``EvaluationRecord``). Where the evolutionary pipeline's types model
federated optimization, these types model post-evolution benchmarking and
paper-ready evaluation results.

Classes:
    Prompt: A loaded prompt ready for evaluation.
    GenerationResult: The raw output of a single model generation call.
    AnsweringEvaluationRecord: Per-question evaluation result.
    AnsweringEvaluationSummary: Aggregate metrics for a (dataset, model,
        prompt) combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Prompt:
    """A loaded prompt ready for evaluation by the Answering Engine.

    Represents a single system prompt that has been loaded from any
    supported source (text file, JSON, JSONL, ``PromptCandidate``, or
    raw string). This is the canonical prompt representation consumed by
    all downstream Answering Engine components.

    Attributes:
        id: Unique identifier for this prompt. May originate from an
            evolutionary ``PromptCandidate`` ID, a filename, or a
            user-supplied label.
        text: The actual system prompt text to be used during evaluation.
        source: Human-readable description of where this prompt was loaded
            from (e.g., ``"text_file"``, ``"json"``, ``"jsonl"``,
            ``"candidate"``, ``"string"``).
        metadata: Arbitrary metadata carried from the source. Defaults to
            an empty dictionary.

    Raises:
        ValueError: If any field fails validation in ``__post_init__``.
    """

    id: str
    text: str
    source: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate field values for this prompt.

        Raises:
            ValueError: If ``id`` is empty or whitespace, ``text`` is
                empty or whitespace, or ``source`` is empty or whitespace.
        """
        if not self.id.strip():
            raise ValueError("Prompt id must not be empty or whitespace.")
        if not self.text.strip():
            raise ValueError("Prompt text must not be empty or whitespace.")
        if not self.source.strip():
            raise ValueError("Prompt source must not be empty or whitespace.")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            A dictionary containing all fields of this prompt.
        """
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Prompt":
        """Reconstruct a Prompt from its dictionary representation.

        Args:
            data: A dictionary as produced by ``as_dict()``.

        Returns:
            A new Prompt instance.

        Raises:
            TypeError: If ``data`` is not a dict.
            ValueError: If required fields are missing or invalid
                (propagated from the constructor).
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"data must be a dict, got {type(data).__name__}."
            )
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            source=str(data.get("source", "dict")),
            metadata=dict(data.get("metadata", {})),
        )

    def __str__(self) -> str:
        """Return a concise, human-readable summary of this prompt.

        Returns:
            A string showing the prompt ID, source, and a truncated
            preview of the text.
        """
        preview = self.text[:80] + "..." if len(self.text) > 80 else self.text
        return (
            f'Prompt(\n'
            f'    id="{self.id}",\n'
            f'    source="{self.source}",\n'
            f'    text="{preview}"\n'
            f')'
        )


@dataclass(slots=True)
class GenerationResult:
    """The raw output of a single model generation call.

    Captures the model's response along with performance metadata
    (latency and generated token count) needed for paper-ready
    benchmarking. This object is produced by ``ResponseGenerator``
    and consumed by ``PredictionExtractor`` and the evaluation
    recording logic.

    Attributes:
        raw_response: The full decoded model output before any
            post-processing (e.g., before stripping ``<think>`` tags).
        clean_response: The model output after removing reasoning
            blocks (e.g., ``<think>...</think>`` tags).
        latency_seconds: Wall-clock time in seconds for the generation
            call, measured from tokenization to decode completion.
        generated_tokens: Number of new tokens generated by the model
            (excluding the input prompt tokens).

    Raises:
        ValueError: If any field fails validation in ``__post_init__``.
    """

    raw_response: str
    clean_response: str
    latency_seconds: float
    generated_tokens: int

    def __post_init__(self) -> None:
        """Validate field values for this generation result.

        Raises:
            ValueError: If ``latency_seconds`` is negative or
                ``generated_tokens`` is negative.
        """
        if self.latency_seconds < 0.0:
            raise ValueError(
                f"latency_seconds must be >= 0, got {self.latency_seconds}."
            )
        if self.generated_tokens < 0:
            raise ValueError(
                f"generated_tokens must be >= 0, got {self.generated_tokens}."
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            A dictionary containing all fields of this result.
        """
        return {
            "raw_response": self.raw_response,
            "clean_response": self.clean_response,
            "latency_seconds": self.latency_seconds,
            "generated_tokens": self.generated_tokens,
        }


@dataclass(slots=True)
class AnsweringEvaluationRecord:
    """Per-question evaluation result produced by the Answering Engine.

    Each record captures the outcome of evaluating one dataset sample
    against one prompt using one model. Records are the atomic unit of
    evaluation data; they are aggregated by ``MetricsCalculator`` into
    ``AnsweringEvaluationSummary`` objects and exported by
    ``ExportManager`` for paper-ready CSV/JSON output.

    Attributes:
        dataset: Name of the dataset this sample belongs to (e.g.,
            ``"medqa"``, ``"pubmedqa"``, ``"medmcqa"``).
        sample_id: Zero-based index of the sample within the dataset.
        prompt_id: Identifier of the prompt used for this evaluation.
        model: Model registry key used for generation (e.g.,
            ``"qwen3-8b"``).
        prediction: The extracted predicted answer. May be an ``int``
            (for multiple-choice) or a ``str`` (for yes/no/maybe), or
            ``None`` if the model response could not be parsed.
        ground_truth: The correct answer from the dataset. An ``int``
            for MedQA/MedMCQA, or a ``str`` for PubMedQA.
        correct: Whether the prediction matches the ground truth.
        latency_seconds: Wall-clock time in seconds for generation.
        generated_tokens: Number of tokens generated by the model.
        raw_response: The full raw model response text.

    Raises:
        ValueError: If any field fails validation in ``__post_init__``.
    """

    dataset: str
    sample_id: int
    prompt_id: str
    model: str
    prediction: object
    ground_truth: object
    correct: bool
    latency_seconds: float
    generated_tokens: int
    raw_response: str

    def __post_init__(self) -> None:
        """Validate field values for this evaluation record.

        Raises:
            ValueError: If ``dataset`` is empty, ``sample_id`` is
                negative, ``prompt_id`` is empty, ``model`` is empty,
                ``latency_seconds`` is negative, or ``generated_tokens``
                is negative.
        """
        if not self.dataset.strip():
            raise ValueError(
                "AnsweringEvaluationRecord dataset must not be empty."
            )
        if self.sample_id < 0:
            raise ValueError(
                f"AnsweringEvaluationRecord sample_id must be >= 0, "
                f"got {self.sample_id}."
            )
        if not self.prompt_id.strip():
            raise ValueError(
                "AnsweringEvaluationRecord prompt_id must not be empty."
            )
        if not self.model.strip():
            raise ValueError(
                "AnsweringEvaluationRecord model must not be empty."
            )
        if self.latency_seconds < 0.0:
            raise ValueError(
                f"AnsweringEvaluationRecord latency_seconds must be >= 0, "
                f"got {self.latency_seconds}."
            )
        if self.generated_tokens < 0:
            raise ValueError(
                f"AnsweringEvaluationRecord generated_tokens must be >= 0, "
                f"got {self.generated_tokens}."
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            A dictionary containing all fields of this record.
        """
        return {
            "dataset": self.dataset,
            "sample_id": self.sample_id,
            "prompt_id": self.prompt_id,
            "model": self.model,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "correct": self.correct,
            "latency_seconds": self.latency_seconds,
            "generated_tokens": self.generated_tokens,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AnsweringEvaluationRecord":
        """Reconstruct an AnsweringEvaluationRecord from its dictionary.

        Args:
            data: A dictionary as produced by ``as_dict()``.

        Returns:
            A new AnsweringEvaluationRecord instance.

        Raises:
            TypeError: If ``data`` is not a dict.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"data must be a dict, got {type(data).__name__}."
            )
        return cls(
            dataset=str(data["dataset"]),
            sample_id=int(data["sample_id"]),
            prompt_id=str(data["prompt_id"]),
            model=str(data["model"]),
            prediction=data["prediction"],
            ground_truth=data["ground_truth"],
            correct=bool(data["correct"]),
            latency_seconds=float(data["latency_seconds"]),
            generated_tokens=int(data["generated_tokens"]),
            raw_response=str(data["raw_response"]),
        )

    def __str__(self) -> str:
        """Return a concise, human-readable summary of this record.

        Returns:
            A string showing key fields of this evaluation record.
        """
        return (
            f'AnsweringEvaluationRecord(\n'
            f'    dataset="{self.dataset}",\n'
            f'    sample_id={self.sample_id},\n'
            f'    model="{self.model}",\n'
            f'    correct={self.correct}\n'
            f')'
        )


@dataclass(slots=True)
class AnsweringEvaluationSummary:
    """Aggregate metrics for one (dataset, model, prompt) combination.

    Produced by ``MetricsCalculator`` from a collection of
    ``AnsweringEvaluationRecord`` objects, this summary contains the
    paper-ready metrics for a single evaluation configuration.

    Attributes:
        dataset: Name of the evaluated dataset.
        model: Model registry key used for generation.
        prompt_id: Identifier of the prompt used.
        accuracy: Fraction of correctly answered questions in [0.0, 1.0].
        precision: Macro-averaged precision across all answer classes.
        recall: Macro-averaged recall across all answer classes.
        macro_f1: Macro-averaged F1 score across all answer classes.
        average_latency: Mean wall-clock generation time in seconds.
        average_tokens: Mean number of generated tokens per question.
        total_questions: Total number of questions evaluated.

    Raises:
        ValueError: If any field fails validation in ``__post_init__``.
    """

    dataset: str
    model: str
    prompt_id: str
    accuracy: float
    precision: float
    recall: float
    macro_f1: float
    average_latency: float
    average_tokens: float
    total_questions: int

    def __post_init__(self) -> None:
        """Validate field values for this evaluation summary.

        Raises:
            ValueError: If ``dataset`` is empty, ``model`` is empty,
                ``prompt_id`` is empty, any metric is negative, or
                ``total_questions`` is not positive.
        """
        if not self.dataset.strip():
            raise ValueError(
                "AnsweringEvaluationSummary dataset must not be empty."
            )
        if not self.model.strip():
            raise ValueError(
                "AnsweringEvaluationSummary model must not be empty."
            )
        if not self.prompt_id.strip():
            raise ValueError(
                "AnsweringEvaluationSummary prompt_id must not be empty."
            )
        if self.total_questions <= 0:
            raise ValueError(
                f"AnsweringEvaluationSummary total_questions must be > 0, "
                f"got {self.total_questions}."
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            A dictionary containing all fields of this summary.
        """
        return {
            "dataset": self.dataset,
            "model": self.model,
            "prompt_id": self.prompt_id,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "macro_f1": self.macro_f1,
            "average_latency": self.average_latency,
            "average_tokens": self.average_tokens,
            "total_questions": self.total_questions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AnsweringEvaluationSummary":
        """Reconstruct an AnsweringEvaluationSummary from its dictionary.

        Args:
            data: A dictionary as produced by ``as_dict()``.

        Returns:
            A new AnsweringEvaluationSummary instance.

        Raises:
            TypeError: If ``data`` is not a dict.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"data must be a dict, got {type(data).__name__}."
            )
        return cls(
            dataset=str(data["dataset"]),
            model=str(data["model"]),
            prompt_id=str(data["prompt_id"]),
            accuracy=float(data["accuracy"]),
            precision=float(data["precision"]),
            recall=float(data["recall"]),
            macro_f1=float(data["macro_f1"]),
            average_latency=float(data["average_latency"]),
            average_tokens=float(data["average_tokens"]),
            total_questions=int(data["total_questions"]),
        )

    def __str__(self) -> str:
        """Return a concise, human-readable summary of these metrics.

        Returns:
            A string showing the key aggregate metrics.
        """
        return (
            f'AnsweringEvaluationSummary(\n'
            f'    dataset="{self.dataset}",\n'
            f'    model="{self.model}",\n'
            f'    prompt_id="{self.prompt_id}",\n'
            f'    accuracy={self.accuracy:.4f},\n'
            f'    macro_f1={self.macro_f1:.4f},\n'
            f'    total_questions={self.total_questions}\n'
            f')'
        )
