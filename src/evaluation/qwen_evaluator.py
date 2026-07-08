"""Qwen-backed concrete implementation of the Evaluator Protocol.

This module defines :class:`QwenEvaluator`, the first concrete
evaluation backend for HFPO. It adapts an already-loaded Qwen model
and tokenizer to the ``Evaluator`` Protocol declared in
``src.evaluation.evaluator``.

QwenEvaluator's sole responsibility is scoring a single dataset
sample against a candidate prompt. It performs no federated
evaluation, prompt evolution, mutation, crossover, tournament
selection, population management, caching, logging, or checkpointing;
those responsibilities belong to other components. It also performs
no prompt building, tokenization, generation, decoding, parsing, or
scoring logic of its own -- ``src.evaluation.scorer.score_sample()``
already assembles the prompt builder, the model/tokenizer, the
parser, and the metrics module into a single evaluation pipeline, so
QwenEvaluator delegates to it entirely rather than duplicating any
part of that pipeline.
"""

from __future__ import annotations

from typing import Any

from src.evaluation import scorer


class QwenEvaluator:
    """Evaluates a single dataset sample against a candidate prompt.

    QwenEvaluator is a thin adapter that satisfies the ``Evaluator``
    Protocol by delegating entirely to
    ``src.evaluation.scorer.score_sample()``, which already wires
    together the prompt builder, the Qwen model/tokenizer, the
    parser, and the metrics module. QwenEvaluator itself builds no
    prompts, tokenizes nothing, generates nothing, parses nothing,
    and scores nothing directly; it only holds a model and tokenizer
    and forwards evaluation requests to the existing scoring
    pipeline.

    Attributes:
        _model: The loaded Qwen model used for generation.
        _tokenizer: The tokenizer paired with ``_model``.
    """

    __slots__ = ("_model", "_tokenizer")

    def __init__(self, model: Any, tokenizer: Any) -> None:
        """Initializes the evaluator with a model and tokenizer.

        Args:
            model: The loaded Qwen model to use for generation. Must
                not be ``None``.
            tokenizer: The tokenizer paired with ``model``. Must not
                be ``None``.

        Raises:
            ValueError: If ``model`` or ``tokenizer`` is ``None``.
        """
        if model is None:
            raise ValueError("model must not be None.")
        if tokenizer is None:
            raise ValueError("tokenizer must not be None.")

        self._model = model
        self._tokenizer = tokenizer

    def score_sample(self, prompt_text: str, sample: Any) -> int:
        """Scores a single dataset sample against a candidate prompt.

        Delegates the entire build-prompt / tokenize / generate /
        decode / parse / score pipeline to
        ``scorer.score_sample()``, passing ``prompt_text`` as that
        function's system prompt. Only the resulting integer score is
        returned; the parsed prediction and the raw decoded response
        produced by the pipeline are discarded.

        Args:
            prompt_text: The candidate prompt to evaluate, used as the
                system prompt for generation. Must not be empty or
                whitespace.
            sample: The dataset sample to evaluate against, in the
                standardized format expected by
                ``src.prompts.builder.build_prompt()`` and
                ``src.evaluation.scorer.score_sample()``.

        Returns:
            1 if the model's prediction for ``sample`` is correct,
            0 otherwise. Never a probability, raw model text, or
            parsed answer.

        Raises:
            ValueError: If ``prompt_text`` is empty or whitespace, or
                if the underlying scoring pipeline produces a score
                that is not exactly 0 or 1.
        """
        if not prompt_text.strip():
            raise ValueError("prompt_text must not be empty.")

        score, _prediction, _clean_response = self.score_sample_details(
            prompt_text,
            sample,
        )

        return score

    def score_sample_details(self, prompt_text: str, sample: Any) -> tuple[int, object, str]:
        """Scores one sample and returns score, parsed prediction, and response."""
        if not prompt_text.strip():
            raise ValueError("prompt_text must not be empty.")

        score, prediction, clean_response = scorer.score_sample(
            self._model,
            self._tokenizer,
            prompt_text,
            sample,
        )

        if score not in (0, 1):
            raise ValueError(
                "scorer.score_sample() must return exactly 0 or 1."
            )

        return score, prediction, clean_response
    
