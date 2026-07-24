"""Privacy mechanisms for federated evaluation in HFPO.

This package provides modular privacy mechanisms that can be composed
or used independently to protect hospital evaluation scores during
federated aggregation.

Available mechanisms:
    - NoPrivacyMechanism: Baseline pass-through (no protection)
    - DifferentialPrivacyMechanism: Local Laplace noise (DP(ε))
    - SecureAggregationMechanism: Additive masking + server unmasking (SecureAgg)

Usage:
    from src.federated.privacy import NoPrivacyMechanism, DifferentialPrivacyMechanism, SecureAggregationMechanism

    # Or use the factory:
    from src.federated.privacy import create_privacy_mechanism

    mechanism = create_privacy_mechanism("dp", epsilon=1.0)
    mechanism = create_privacy_mechanism("secure_agg")
    mechanism = create_privacy_mechanism("none")
"""

from __future__ import annotations

from src.federated.privacy.composed import ComposedPrivacyMechanism
from src.federated.privacy.differential_privacy import DifferentialPrivacyMechanism
from src.federated.privacy.no_privacy import NoPrivacyMechanism
from src.federated.privacy.secure_aggregation import SecureAggregationMechanism
from src.federated.privacy.interfaces import PrivacyMechanism


def create_privacy_mechanism(mode: str, **kwargs) -> PrivacyMechanism:
    """Factory function to create a privacy mechanism by name.

    Args:
        mode: Privacy mode string. Supports:
            - Single: "none", "dp", "secure_agg"
            - Combined: "dp+secure_agg", "secure_agg+dp", "dp,sa", "sa,dp"
        **kwargs: Mechanism-specific parameters:
            - dp: epsilon (required), sensitivity (optional), clip_scores (bool), random_seed (int)
            - secure_agg: random_seed (optional)
            - none: no parameters

    Returns:
        Configured PrivacyMechanism instance (or ComposedPrivacyMechanism for combined modes).

    Raises:
        ValueError: If mode is unknown or required params missing.
    """
    mode = mode.lower().strip().replace(" ", "")

    # Handle combined modes (e.g., "dp+secure_agg", "dp,sa")
    if "+" in mode or "," in mode:
        separator = "+" if "+" in mode else ","
        parts = [p.strip() for p in mode.split(separator) if p.strip()]
        mechanisms = []
        for part in parts:
            mechanisms.append(_create_single_mechanism(part, **kwargs))
        return ComposedPrivacyMechanism(mechanisms)

    return _create_single_mechanism(mode, **kwargs)


def _create_single_mechanism(mode: str, **kwargs) -> PrivacyMechanism:
    """Create a single privacy mechanism (internal helper)."""
    if mode in ("none", "no_privacy", "baseline"):
        return NoPrivacyMechanism()

    if mode in ("dp", "differential_privacy", "differentialprivacy"):
        epsilon = kwargs.get("epsilon")
        if epsilon is None:
            raise ValueError("DifferentialPrivacyMechanism requires 'epsilon' parameter")
        return DifferentialPrivacyMechanism(
            epsilon=epsilon,
            sensitivity=kwargs.get("sensitivity"),
            clip_scores=kwargs.get("clip_scores", True),
            random_seed=kwargs.get("random_seed"),
        )

    if mode in ("secure_agg", "secure_aggregation", "secureagg", "sa"):
        return SecureAggregationMechanism(
            random_seed=kwargs.get("random_seed"),
        )

    raise ValueError(
        f"Unknown privacy mode: {mode}. "
        f"Valid modes: 'none', 'dp', 'secure_agg', or combinations like 'dp+secure_agg'"
    )


__all__ = [
    "PrivacyMechanism",
    "NoPrivacyMechanism",
    "DifferentialPrivacyMechanism",
    "SecureAggregationMechanism",
    "ComposedPrivacyMechanism",
    "create_privacy_mechanism",
]