"""Local Differential Privacy mechanism for federated evaluation.

This mechanism applies calibrated Laplace noise to each hospital's accuracy
score locally before transmission to the server, providing epsilon-DP
for the per-hospital accuracy scalar.
"""

from __future__ import annotations

import math
import random
import secrets
from dataclasses import replace
from typing import Sequence

from src.core.evaluation_record import EvaluationRecord
from src.federated.privacy.interfaces import PrivacyMechanism


class DifferentialPrivacyMechanism(PrivacyMechanism):
    """Applies local differential privacy via Laplace noise addition.

    Each hospital independently adds noise drawn from Laplace(0, Δ/ε)
    to its accuracy score before sending to the server. The sensitivity
    Δ for accuracy in [0, 1] evaluated on n samples is 1/n (changing
    one sample changes accuracy by at most 1/n).

    This implements the DP(ε) mode from Algorithm 2 in the paper:
    - Hospital computes a_i = LocalAccuracy(p, D_i)
    - Hospital adds noise: a_i ← a_i + Laplace(0, Δ/ε)
    - Noisy a_i sent to server

    Attributes:
        epsilon: Privacy budget ε > 0. Smaller = more private, noisier.
        sensitivity: Global sensitivity Δ of the accuracy query.
            Defaults to 1.0 / evaluation_subset_size (per-sample sensitivity).
        clip_scores: Whether to clip noisy scores to [0, 1] range.
    """

    def __init__(
        self,
        epsilon: float,
        sensitivity: float | None = None,
        clip_scores: bool = True,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the DP mechanism.

        Args:
            epsilon: Privacy budget ε > 0. Smaller values add more noise.
            sensitivity: Sensitivity Δ of the accuracy query.
                If None, computed as 1 / subset_size at apply() time.
            clip_scores: If True, clip noisy scores to valid [0, 1] range.
            random_seed: Optional seed for reproducible noise (for testing).

        Raises:
            ValueError: If epsilon <= 0.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")

        self.epsilon = float(epsilon)
        self._fixed_sensitivity = sensitivity
        self.clip_scores = clip_scores
        self._rng = secrets.SystemRandom()
        if random_seed is not None:
            self._rng = random.Random(random_seed)

    @property
    def sensitivity(self) -> float:
        """Return the sensitivity, computing if needed."""
        return self._fixed_sensitivity if self._fixed_sensitivity is not None else 1.0

    def _compute_sensitivity(self, num_samples: int) -> float:
        """Compute sensitivity Δ = 1/n for accuracy on n samples."""
        if num_samples <= 0:
            return 1.0
        return 1.0 / float(num_samples)

    def _laplace_noise(self, scale: float) -> float:
        """Sample from Laplace(0, scale) using inverse transform."""
        u = self._rng.random() - 0.5  # Uniform in (-0.5, 0.5)
        return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))

    def apply(
        self,
        records: Sequence[EvaluationRecord],
    ) -> Sequence[EvaluationRecord]:
        """Add Laplace noise to each hospital's accuracy score.

        Args:
            records: EvaluationRecord objects from hospitals for one prompt.
                Each record must have .score (accuracy in [0,1]) and
                .num_total (sample count for sensitivity).

        Returns:
            New EvaluationRecord objects with .score replaced by noisy score.
            Other fields (.num_correct, .num_total, .metadata) are preserved.
        """
        if not records:
            return tuple()

        noisy_records = []
        for record in records:
            sensitivity = (
                self._fixed_sensitivity
                if self._fixed_sensitivity is not None
                else self._compute_sensitivity(record.num_total)
            )
            scale = sensitivity / self.epsilon

            noise = self._laplace_noise(scale)
            noisy_score = record.score + noise

            if self.clip_scores:
                noisy_score = max(0.0, min(1.0, noisy_score))

            noisy_record = replace(
                record,
                score=noisy_score,
                num_correct=int(round(noisy_score * record.num_total)),
                metadata={
                    **record.metadata,
                    "dp_noise_scale": scale,
                    "dp_epsilon": self.epsilon,
                    "dp_original_score": record.score,
                },
            )
            noisy_records.append(noisy_record)

        return tuple(noisy_records)

    def __str__(self) -> str:
        return f"DifferentialPrivacyMechanism(epsilon={self.epsilon})"

    def __repr__(self) -> str:
        return (
            f"DifferentialPrivacyMechanism(epsilon={self.epsilon}, "
            f"sensitivity={self._fixed_sensitivity}, "
            f"clip_scores={self.clip_scores})"
        )