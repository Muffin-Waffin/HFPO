"""No-privacy baseline mechanism (pass-through).

This mechanism applies no transformation to evaluation records,
serving as the default when privacy is disabled.
"""

from __future__ import annotations

from typing import Sequence

from src.core.evaluation_record import EvaluationRecord
from src.federated.privacy.interfaces import PrivacyMechanism


class NoPrivacyMechanism(PrivacyMechanism):
    """Identity mechanism — returns records unchanged."""

    def apply(
        self,
        records: Sequence[EvaluationRecord],
    ) -> Sequence[EvaluationRecord]:
        """Return records unmodified."""
        return tuple(records)

    def __str__(self) -> str:
        return "NoPrivacyMechanism()"

    def __repr__(self) -> str:
        return "NoPrivacyMechanism()"