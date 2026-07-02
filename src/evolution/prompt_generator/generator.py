"""Prompt generation orchestration for the Prompt Generator
subsystem.

This module defines PromptGenerator, the orchestration component
responsible for creating exactly one new PromptCandidate from a
PromptGenerationRequest. PromptGenerator coordinates the complete
generation pipeline -- template building, LLM invocation, cleaning,
validation, and candidate assembly -- but delegates all specialized
work to injected components. It performs no population management,
no tournament selection, no elitism, no federated evaluation, no
fitness computation, and no evolutionary loop.
"""

import time
import uuid

from src.core.prompt_candidate import PromptCandidate
from src.core.lineage_tracker import LineageTracker

from .cleaner import PromptCleaner
from .interfaces import ReasoningLLM
from .models import (
    GenerationMetadata,
    PromptGenerationRequest,
    PromptGenerationResult,
)
from .templates import PromptTemplateBuilder
from .validator import PromptValidator


class PromptGenerator:
    """Orchestrates the creation of one new PromptCandidate.

    PromptGenerator coordinates a fixed pipeline -- building the LLM
    instruction, invoking the reasoning LLM, cleaning the raw output,
    validating the cleaned output, determining lineage, and assembling
    the resulting PromptCandidate -- by delegating each step to an
    injected collaborator. It performs no population management, no
    tournament selection, no elitism, no federated evaluation, no
    fitness computation, and no evolutionary loop of its own; it
    produces exactly one PromptGenerationResult per call.
    """

    __slots__ = (
        "_llm",
        "_template_builder",
        "_cleaner",
        "_validator",
        "_lineage_tracker",
    )

    def __init__(
        self,
        llm: ReasoningLLM,
        template_builder: PromptTemplateBuilder,
        cleaner: PromptCleaner,
        validator: PromptValidator,
        lineage_tracker: LineageTracker,
    ) -> None:
        """Initializes the generator with its required collaborators.

        Args:
            llm: The reasoning LLM used to generate raw prompt text.
                Must expose ``generate(prompt: str, temperature:
                float) -> str``.
            template_builder: Builds the LLM instruction string from a
                PromptGenerationRequest.
            cleaner: Cleans raw LLM output into normalized prompt
                text.
            validator: Validates cleaned prompt text against
                population acceptance rules.
            lineage_tracker: Builds ancestry identifiers from parent
                identifiers.

        Raises:
            ValueError: If any of the provided collaborators is None.
        """
        if llm is None:
            raise ValueError("llm must not be None.")
        if template_builder is None:
            raise ValueError("template_builder must not be None.")
        if cleaner is None:
            raise ValueError("cleaner must not be None.")
        if validator is None:
            raise ValueError("validator must not be None.")
        if lineage_tracker is None:
            raise ValueError("lineage_tracker must not be None.")

        self._llm = llm
        self._template_builder = template_builder
        self._cleaner = cleaner
        self._validator = validator
        self._lineage_tracker = lineage_tracker

    def generate(
        self,
        request: PromptGenerationRequest,
    ) -> PromptGenerationResult:
        """Generates exactly one new PromptCandidate from a request.

        Runs the full generation pipeline: builds the LLM instruction,
        invokes the reasoning LLM, cleans and validates the output,
        determines origin and lineage, and assembles the resulting
        PromptCandidate together with its generation metadata.

        Args:
            request: The generation request describing the parent(s),
                task, and generation context to generate from.

        Returns:
            A PromptGenerationResult containing the newly generated
            PromptCandidate and the GenerationMetadata describing how
            it was produced.

        Raises:
            TypeError: If request is not a PromptGenerationRequest.
            Exception: Any exception raised by the injected LLM,
                cleaner, validator, or lineage tracker propagates
                unmodified; this method performs no retries and
                catches nothing.
        """
        if not isinstance(request, PromptGenerationRequest):
            raise TypeError(
                "request must be a PromptGenerationRequest, got "
                f"{type(request).__name__}."
            )

        start_time = time.monotonic()

        mutation_operator, instruction = self._template_builder.build(request)
        raw_output = self._llm.generate(
            prompt=instruction,
            temperature=request.temperature,
        )
        clean_text = self._cleaner.clean(raw_output)
        self._validator.validate(clean_text, request)

        # candidate_id = self._generate_candidate_id()

        # is_crossover = request.is_crossover()
        # origin = "crossover" if is_crossover else "mutation"

        # if is_crossover:
        #     parent_ids = [request.parent_a.id, request.parent_b.id]
        # else:
        #     parent_ids = [request.parent_a.id]

        candidate_id = self._generate_candidate_id()

        origin = "mutation"

        parent_ids = [request.parent_a.id]

        # ancestry_ids = self._lineage_tracker.build_ancestry(parent_ids)
        ancestry_ids = self._lineage_tracker.build_ancestry(parent_ids)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        metadata = self._create_metadata(
            origin,
            request.temperature,
            elapsed_ms,
            mutation_operator,
        )
        candidate = self._create_candidate(
            candidate_id=candidate_id,
            text=clean_text,
            request=request,
            parent_ids=parent_ids,
            ancestry_ids=ancestry_ids,
            origin=origin,
            metadata=metadata,
        )
        self._lineage_tracker.register(candidate)
        return PromptGenerationResult(
            candidate=candidate,
            metadata=metadata,
        )

    def _generate_candidate_id(self) -> str:
        """Generate a unique identifier for a new prompt candidate.

        Returns:
            A UUID string suitable for use as a PromptCandidate ID.
        """
        return str(uuid.uuid4())
    def _create_metadata(
        self,
        origin: str,
        temperature: float,
        elapsed_ms: float,
        mutation_operator: str | None,
    ) -> GenerationMetadata:
        """Creates the GenerationMetadata describing this generation.

        Args:
            origin: The origin of the generation, either "mutation" or
                "crossover".
            temperature: The sampling temperature requested for this
                generation, taken from the originating request.
            elapsed_ms: The measured elapsed generation time, in
                milliseconds.

        Returns:
            A GenerationMetadata instance recording the generator
            name, sampling temperature, attempt count, template used,
            and elapsed generation time.
        """
        return GenerationMetadata(
            generator=type(self._llm).__name__,
            temperature=temperature,
            attempts=1,
            template_used=origin,
            mutation_operator=mutation_operator,
            generation_time_ms=elapsed_ms,
        )

    def _create_candidate(
        self,
        candidate_id: str,
        text: str,
        request: PromptGenerationRequest,
        parent_ids: list[str],
        ancestry_ids: list[str],
        origin: str,
        metadata: GenerationMetadata,
    ) -> PromptCandidate:
        """Assembles the newly generated PromptCandidate.

        Args:
            candidate_id: The freshly generated unique identifier for
                this candidate.
            text: The cleaned and validated prompt text.
            request: The generation request this candidate was
                produced from, providing the target generation index.
            parent_ids: The identifiers of the candidate's direct
                parent(s).
            ancestry_ids: The full ancestry identifiers produced by
                the lineage tracker.
            origin: The origin of the generation, either "mutation" or
                "crossover".
            metadata: The GenerationMetadata describing how this
                candidate was generated.

        Returns:
            A new PromptCandidate with fitness left unset (None),
            ready for downstream evaluation.
        """
        return PromptCandidate(
            id=candidate_id,
            text=text,
            generation=request.generation,
            parent_ids=parent_ids,
            ancestry_ids=ancestry_ids,
            origin=origin,
            metadata=metadata.to_dict(),
            fitness=None,
        )