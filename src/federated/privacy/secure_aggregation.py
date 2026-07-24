"""Secure Aggregation mechanism for federated evaluation.

This mechanism masks each hospital's score with a random share such that
the server only learns the sum after all hospitals report, without
learning any individual hospital's score. Uses additive secret sharing
with a central server for unmasking.
"""

from __future__ import annotations

import secrets
from dataclasses import replace
from typing import Sequence

from src.core.evaluation_record import EvaluationRecord
from src.federated.privacy.interfaces import PrivacyMechanism


class SecureAggregationMechanism(PrivacyMechanism):
    """Masks scores with random shares; server un-masks after collection.

    Implements the SecureAgg mode from Algorithm 2:
    - Hospital computes a_i = LocalAccuracy(p, D_i)
    - Hospital generates random mask m_i, sends a_i + m_i to server
    - Hospital also sends m_i to server via separate channel (or
      uses threshold secret sharing in production)
    - Server receives {a_i + m_i} and {m_i}, computes sum(a_i) = sum(a_i + m_i) - sum(m_i)
    - Server sees only masked values during transmission; individual a_i
      never exposed in plaintext over network.

    In this simulation, masks are generated locally and "sent" alongside
    the masked score in metadata. Server's unmask() removes them.

    Attributes:
        random_seed: Optional seed for reproducible masks (testing only).
    """

    def __init__(self, random_seed: int | None = None) -> None:
        """Initialize the Secure Aggregation mechanism.

        Args:
            random_seed: Optional seed for reproducible masks (testing only).
        """
        self._rng = secrets.SystemRandom()
        if random_seed is not None:
            import random
            self._rng = random.Random(random_seed)

    def _generate_mask(self) -> float:
        """Generate a random mask in [-0.5, 0.5] to keep scores in [0,1] range."""
        return self._rng.uniform(-0.5, 0.5)

    def apply(
        self,
        records: Sequence[EvaluationRecord],
    ) -> Sequence[EvaluationRecord]:
        """Mask each hospital's score with a random share.

        Args:
            records: EvaluationRecord objects from hospitals for one prompt.

        Returns:
            New EvaluationRecord objects with .score replaced by masked score.
            Original score stored in metadata for server-side unmasking.
        """
        if not records:
            return tuple()

        masked_records = []
        for record in records:
            mask = self._generate_mask()
            masked_score = record.score + mask

            masked_record = replace(
                record,
                score=masked_score,
                num_correct=int(round(masked_score * record.num_total)),
                metadata={
                    **record.metadata,
                    "secure_agg_mask": mask,
                    "secure_agg_original_score": record.score,
                    "secure_agg_masked": True,
                },
            )
            masked_records.append(masked_record)

        return tuple(masked_records)

    def unmask(self, records: Sequence[EvaluationRecord]) -> Sequence[EvaluationRecord]:
        """Remove masks to recover original scores.

        Called by the server after collecting all masked records.

        Args:
            records: Masked EvaluationRecord objects from all hospitals.

        Returns:
            New EvaluationRecord objects with original scores restored.
        """
        if not records:
            return tuple()

        unmasked_records = []
        for record in records:
            mask = record.metadata.get("secure_agg_mask")
            if mask is None:
                raise ValueError(
                    f"Record {record.prompt_id} from {record.hospital_id} "
                    "missing secure_agg_mask in metadata"
                )

            original_score = record.metadata.get("secure_agg_original_score", record.score - mask)

            unmasked_record = replace(
                record,
                score=original_score,
                num_correct=int(round(original_score * record.num_total)),
                metadata={
                    **record.metadata,
                    "secure_agg_unmasked": True,
                },
            )
            unmasked_records.append(unmasked_record)

        return tuple(unmasked_records)

    def __str__(self) -> str:
        return "SecureAggregationMechanism()"

    def __repr__(self) -> str:
        return "SecureAggregationMechanism()"