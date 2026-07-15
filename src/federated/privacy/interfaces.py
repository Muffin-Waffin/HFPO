"""Privacy layer interfaces for the HFPO federated evaluation pipeline.

This module defines the abstract interface implemented by every privacy
mechanism in the federated layer. By depending on a Protocol rather than a
concrete implementation, ``FederatedServer`` remains independent of which
privacy mode (none, differential privacy, secure aggregation) is active.

It contains NO privacy logic itself — no noise generation, no masking,
no aggregation. It only defines the interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.evaluation_record import EvaluationRecord


@runtime_checkable
class PrivacyMechanism(Protocol):
    """Protocol for privacy mechanisms applied to federated evaluation records.

    A ``PrivacyMechanism`` receives the full list of ``EvaluationRecord``
    objects collected for one evaluation round (across all hospitals, for the
    uncached prompts in that round) and returns a new list of the same
    length and order with scores (and optional metadata) transformed
    according to the privacy policy (e.g., Gaussian noise addition, score
    clipping, secure‑aggregation masking).

    Implementations must not mutate the input list or its ``EvaluationRecord``
    elements in place; new instances should be constructed (e.g., via
    ``dataclasses.replace`` or explicit construction).
    """

    def apply(
        self,
        records: Sequence[EvaluationRecord],
    ) -> Sequence[EvaluationRecord]:
        """Apply the privacy mechanism to a batch of evaluation records.

        Args:
            records: Immutable sequence of ``EvaluationRecord`` objects,
                one per hospital, all belonging to the same prompt
                generation.

        Returns:
            A new sequence of ``EvaluationRecord`` objects of the same
            length, with scores (and any metadata) transformed according
            to the privacy policy. The original ``records`` must not be
            mutated.
        """
        ...


__all__ = [
    "PrivacyMechanism",
]