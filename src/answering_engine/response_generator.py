"""Response generation for the Answering Engine.

This module defines :class:`ResponseGenerator`, responsible for querying
an answer model with a prompt and a dataset question and returning the
raw response together with performance metadata (latency and generated
token count). It wraps the existing model loading infrastructure from
``src.llms.llm`` and replicates the generation pipeline from
``src.evaluation.scorer`` with added instrumentation, without modifying
either module.

The generator performs no evaluation logic: it produces a
:class:`GenerationResult` containing the raw and cleaned response text
along with timing and token metrics. Downstream components
(:class:`PredictionExtractor`, :class:`MetricsCalculator`) handle
answer extraction and scoring.
"""

from __future__ import annotations

import re
import time
from typing import Any

import torch

from configs.config import MAX_NEW_TOKENS
from src.answering_engine.models import GenerationResult
from src.llms.llm import load_llm, get_model_config
from src.prompts.builder import build_prompt


def _build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """Build chat messages for the model.

    Constructs the standard system/user message pair expected by
    chat-template-based models.

    Args:
        system_prompt: The system prompt (evolved or baseline).
        user_prompt: The formatted question prompt produced by
            ``build_prompt()``.

    Returns:
        A list of two message dictionaries with ``"role"`` and
        ``"content"`` keys.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from model output.

    Some models (e.g., Qwen3) produce internal reasoning wrapped in
    ``<think>`` tags. This function strips those blocks to extract the
    final answer text.

    Args:
        text: The raw decoded model output.

    Returns:
        The text with all ``<think>...</think>`` blocks removed and
        surrounding whitespace stripped.
    """
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


class ResponseGenerator:
    """Queries an answer model and returns instrumented responses.

    ``ResponseGenerator`` holds a loaded model and tokenizer and
    provides a single ``generate()`` method that, given a system
    prompt and a dataset sample:

    1. Builds the user prompt via ``src.prompts.builder.build_prompt()``.
    2. Applies the model's chat template.
    3. Runs deterministic generation (``do_sample=False``).
    4. Measures wall-clock latency.
    5. Counts generated tokens.
    6. Strips reasoning blocks (``<think>`` tags).

    The generator performs **no evaluation logic** — it does not parse
    answers, compute scores, or access any evolutionary state.

    Attributes:
        _model: The loaded language model.
        _tokenizer: The tokenizer paired with ``_model``.
        _model_key: The model registry key used to load this model.
        _max_new_tokens: Maximum number of tokens to generate.
    """

    __slots__ = ("_model", "_tokenizer", "_model_key", "_max_new_tokens")

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_key: str,
        max_new_tokens: int | None = None,
    ) -> None:
        """Initialize the generator with a loaded model and tokenizer.

        Args:
            model: A loaded language model (e.g., from
                ``src.llms.llm.load_llm()``). Must not be ``None``.
            tokenizer: The tokenizer paired with ``model``. Must not
                be ``None``.
            model_key: The model registry key identifying this model
                (e.g., ``"qwen3-8b"``).
            max_new_tokens: Maximum new tokens to generate. If ``None``,
                the value from the model's registry config is used; if
                that is also unavailable, the global ``MAX_NEW_TOKENS``
                default is used.

        Raises:
            ValueError: If ``model`` or ``tokenizer`` is ``None``, or
                if ``model_key`` is empty.
        """
        if model is None:
            raise ValueError("model must not be None.")
        if tokenizer is None:
            raise ValueError("tokenizer must not be None.")
        if not model_key.strip():
            raise ValueError("model_key must not be empty.")

        self._model = model
        self._tokenizer = tokenizer
        self._model_key = model_key

        if max_new_tokens is not None:
            self._max_new_tokens = max_new_tokens
        else:
            try:
                config = get_model_config(model_key)
                self._max_new_tokens = config.get(
                    "max_new_tokens", MAX_NEW_TOKENS
                )
            except ValueError:
                self._max_new_tokens = MAX_NEW_TOKENS

    @classmethod
    def from_model_key(
        cls,
        model_key: str,
        max_new_tokens: int | None = None,
    ) -> "ResponseGenerator":
        """Create a ResponseGenerator by loading a model from the registry.

        This is the recommended factory method. It loads the model and
        tokenizer via ``src.llms.llm.load_llm()`` and wraps them in a
        new ``ResponseGenerator``.

        Args:
            model_key: A key from ``configs.config.MODEL_REGISTRY``
                (e.g., ``"qwen3-8b"``, ``"phi-4"``).
            max_new_tokens: Optional override for the maximum number of
                tokens to generate.

        Returns:
            A new ``ResponseGenerator`` instance with the loaded model.

        Raises:
            ValueError: If ``model_key`` is not found in the registry.
        """
        print(f"[ResponseGenerator] Loading model: {model_key}")
        model, tokenizer = load_llm(model_key)
        print(f"[ResponseGenerator] Model loaded: {model_key}")

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_key=model_key,
            max_new_tokens=max_new_tokens,
        )

    @property
    def model_key(self) -> str:
        """Return the model registry key for this generator."""
        return self._model_key

    def generate(
        self,
        system_prompt: str,
        sample: dict[str, Any],
    ) -> GenerationResult:
        """Generate a response for a single dataset sample.

        Runs the full generation pipeline: prompt building, chat
        template application, model inference, decoding, and
        post-processing. Measures wall-clock latency and counts
        generated tokens.

        Args:
            system_prompt: The system prompt text (evolved or baseline).
            sample: A dataset sample in the standardized format
                (question, context, choices, answer).

        Returns:
            A :class:`GenerationResult` containing the raw response,
            cleaned response, latency in seconds, and generated token
            count.
        """
        # Build the user prompt from the dataset sample.
        user_prompt = build_prompt(sample)

        # Build chat messages.
        messages = _build_messages(system_prompt, user_prompt)

        # Apply chat template with graceful fallback for models that
        # do not support the enable_thinking parameter.
        try:
            inputs = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            inputs = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            inputs = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )

        if isinstance(inputs, torch.Tensor):
            input_length = inputs.shape[-1]
            inputs_dict = {"input_ids": inputs.to(self._model.device)}
            if self._tokenizer.pad_token_id is not None:
                inputs_dict["attention_mask"] = (inputs != self._tokenizer.pad_token_id).long().to(self._model.device)
            else:
                inputs_dict["attention_mask"] = torch.ones_like(inputs, device=self._model.device)
            inputs = inputs_dict
        else:
            input_length = inputs["input_ids"].shape[-1]
            inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Resolve all possible stop token IDs for this model
        # Models like Llama-3 (`<|eot_id|>`), Phi-4 (`<|im_end|>`), or Qwen
        # may have multiple stop tokens configured in generation_config.
        eos_token_ids: list[int] | int | None = None
        gen_config_eos = getattr(self._model.generation_config, "eos_token_id", None) if hasattr(self._model, "generation_config") else None
        if gen_config_eos is not None:
            if isinstance(gen_config_eos, list):
                eos_token_ids = list(gen_config_eos)
                if self._tokenizer.eos_token_id is not None and self._tokenizer.eos_token_id not in eos_token_ids:
                    eos_token_ids.append(self._tokenizer.eos_token_id)
            else:
                eos_token_ids = [gen_config_eos]
                if self._tokenizer.eos_token_id is not None and self._tokenizer.eos_token_id != gen_config_eos:
                    eos_token_ids.append(self._tokenizer.eos_token_id)
        elif self._tokenizer.eos_token_id is not None:
            eos_token_ids = self._tokenizer.eos_token_id

        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            if isinstance(eos_token_ids, list) and eos_token_ids:
                pad_token_id = eos_token_ids[0]
            elif isinstance(eos_token_ids, int):
                pad_token_id = eos_token_ids
            else:
                pad_token_id = self._tokenizer.eos_token_id

        gen_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": False,
        }
        if eos_token_ids is not None:
            gen_kwargs["eos_token_id"] = eos_token_ids
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id

        # Generate with latency measurement.
        start_time = time.perf_counter()

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                **gen_kwargs,
            )

        end_time = time.perf_counter()
        latency_seconds = end_time - start_time

        # Extract generated tokens (exclude input tokens).
        generated_ids = output_ids[0][input_length:]
        generated_tokens = len(generated_ids)

        # Decode the generated tokens.
        raw_response = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Strip reasoning blocks.
        clean_response = _strip_think_tags(raw_response)

        return GenerationResult(
            raw_response=raw_response,
            clean_response=clean_response,
            latency_seconds=latency_seconds,
            generated_tokens=generated_tokens,
        )
