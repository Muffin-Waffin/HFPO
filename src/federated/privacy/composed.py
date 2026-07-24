"""Composed privacy mechanism for chaining multiple privacy layers."""

from __future__ import annotations

from typing import Sequence

from src.core.evaluation_record import EvaluationRecord
from src.federated.privacy.interfaces import PrivacyMechanism


class ComposedPrivacyMechanism(PrivacyMechanism):
    """Chains multiple privacy mechanisms in sequence.

    Applies mechanisms in order: first_mechanism.apply() → second_mechanism.apply() → ...

    For server-side unmasking, calls unmask() in reverse order (last to first).
    """

    def __init__(self, mechanisms: Sequence[PrivacyMechanism]) -> None:
        """Initialize with a sequence of privacy mechanisms.

        Args:
            mechanisms: Ordered list of PrivacyMechanism instances.
                Applied in sequence for .apply(), reverse sequence for .unmask().
        """
        if not mechanisms:
            raise ValueError("ComposedPrivacyMechanism requires at least one mechanism")
        self._mechanisms = tuple(mechanisms)

    def apply(
        self,
        records: Sequence[EvaluationRecord],
    ) -> Sequence[EvaluationRecord]:
        """Apply all mechanisms in sequence."""
        result = records
        for mechanism in self._mechanisms:
            result = mechanism.apply(result)
        return result

    def unmask(self, records: Sequence[EvaluationRecord]) -> Sequence[EvaluationRecord]:
        """Unmask in reverse order (last mechanism first)."""
        result = records
        for mechanism in reversed(self._mechanisms):
            if hasattr(mechanism, "unmask"):
                result = mechanism.unmask(result)
        return result

    def __str__(self) -> str:
        return "ComposedPrivacyMechanism(" + " → ".join(str(m) for m in self._mechanisms) + ")"

    def __repr__(self) -> str:
        return f"ComposedPrivacyMechanism({list(self._mechanisms)})"