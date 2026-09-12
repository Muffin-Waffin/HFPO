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
    mechanism = create_privacy_mechanism("secure_agg", coordinator=coordinator, hospital_id="medqa")
    mechanism = create_privacy_mechanism("none")

Factory helpers:
    create_secure_agg_coordinator(hospital_ids, random_seed=None) -> SecureAggregationCoordinator
        Creates the one coordinator instance every hospital- and
        server-side SecureAggregationMechanism for a run must share.
"""

from __future__ import annotations

from typing import Sequence

from src.federated.privacy.composed import ComposedPrivacyMechanism
from src.federated.privacy.differential_privacy import DifferentialPrivacyMechanism
from src.federated.privacy.no_privacy import NoPrivacyMechanism
from src.federated.privacy.secure_aggregation import SecureAggregationMechanism
from src.federated.privacy.secure_aggregation_coordinator import SecureAggregationCoordinator
from src.federated.privacy.interfaces import PrivacyMechanism


def create_secure_agg_coordinator(
    hospital_ids: Sequence[str],
    random_seed: int | None = None,
) -> SecureAggregationCoordinator:
    """Creates the one coordinator instance every hospital- and
    server-side SecureAggregationMechanism for a run must share.

    Args:
        hospital_ids: The full, fixed set of hospital IDs
            participating in this run.
        random_seed: Optional seed for reproducible mask generation,
            passed through to SecureAggregationCoordinator.

    Returns:
        A new SecureAggregationCoordinator configured for exactly
        these hospitals.
    """
    return SecureAggregationCoordinator(hospital_ids, random_seed=random_seed)


def create_privacy_mechanism(mode: str, **kwargs) -> PrivacyMechanism:
    """Factory function to create a privacy mechanism by name.

    Args:
        mode: Privacy mode string. Supports:
            - Single: "none", "dp", "secure_agg"
            - Combined: "dp+secure_agg", "secure_agg+dp", "dp,sa", "sa,dp"
        **kwargs: Mechanism-specific parameters:
            - dp: epsilon (required), sensitivity (optional), clip_scores (bool), random_seed (int)
            - secure_agg: coordinator (required), hospital_id (optional) for server-side role
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
        coordinator = kwargs.get("coordinator")
        if coordinator is None:
            raise ValueError(
                "SecureAggregationMechanism requires 'coordinator' parameter "
                "(create with create_secure_agg_coordinator)"
            )
        hospital_id = kwargs.get("hospital_id")
        return SecureAggregationMechanism(
            coordinator=coordinator,
            hospital_id=hospital_id,
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
    "SecureAggregationCoordinator",
    "create_privacy_mechanism",
    "create_secure_agg_coordinator",
]