"""Pairwise-masking coordinator for secure aggregation.

Generates random additive masks between every pair of participating
hospitals for each evaluation round (prompt_id).  Each hospital's net
mask is the sum of its signed pairwise masks with every other
hospital.  Because every pair contributes +m to one side and -m to the
other, the net masks cancel exactly when summed across all hospitals,
allowing the server to recover the true aggregate without learning any
individual hospital's score.

Thread-safety
-------------
:class:`SecureAggregationCoordinator` guards all mutable state behind a
single :class:`threading.Lock`.  ``HospitalClient`` objects are
evaluated concurrently via :class:`concurrent.futures.ThreadPoolExecutor`
in the HFPO federated layer, so two hospitals CAN call
:meth:`get_net_mask` for the same round simultaneously.  If both sides
of a pair independently generated their own random value (instead of
sharing one via the lock-guarded lookup-or-create), the masks would
**not** cancel and aggregate results would be silently wrong.
"""

from __future__ import annotations

import random
import threading
from typing import Sequence

__all__ = ["SecureAggregationCoordinator"]


class SecureAggregationCoordinator:
    """Generates and caches pairwise masks that cancel across hospitals.

    One coordinator instance is shared by **all** per-hospital
    :class:`SecureAggregationMechanism` instances AND by the server-side
    mechanism instance.  It is the single source of truth for pairwise
    masks within an evaluation round.

    Attributes:
        _hospital_ids: Deterministically sorted tuple of hospital IDs.
        _hospital_set: Frozen set for O(1) membership checks.
        _pair_masks: ``dict[str, dict[tuple[str, str], float]]`` —
            outer key is ``prompt_id``, inner key is the *sorted* pair
            tuple ``(id_lo, id_hi)``.
        _lock: Threading lock guarding ``_pair_masks`` and ``_rng``.
        _rng: Shared ``random.Random`` instance for mask generation.
    """

    __slots__ = ("_hospital_ids", "_hospital_set", "_pair_masks", "_lock", "_rng")

    def __init__(
        self, hospital_ids: Sequence[str], random_seed: int | None = None
    ) -> None:
        """Initialize the coordinator with a fixed set of hospital IDs.

        Args:
            hospital_ids: The full, fixed set of hospital IDs that will
                participate in every round.  Stored as a
                deterministically sorted tuple — sort order defines
                which side of each pair receives ``+mask`` vs ``-mask``.
            random_seed: Optional seed for reproducible mask generation.
                None (default) uses non-deterministic randomness.

        Raises:
            ValueError: If ``hospital_ids`` has fewer than 2 entries or
                contains duplicates.
        """
        ids_list = list(hospital_ids)
        if len(ids_list) < 2:
            raise ValueError(
                f"SecureAggregationCoordinator requires at least 2 hospital IDs, "
                f"got {len(ids_list)}."
            )
        if len(set(ids_list)) != len(ids_list):
            raise ValueError(
                "SecureAggregationCoordinator received duplicate hospital IDs."
            )

        self._hospital_ids: tuple[str, ...] = tuple(sorted(ids_list))
        self._hospital_set: frozenset[str] = frozenset(self._hospital_ids)

        # Outer key: prompt_id.  Inner key: sorted (id_lo, id_hi) pair.
        self._pair_masks: dict[str, dict[tuple[str, str], float]] = {}

        self._lock = threading.Lock()
        self._rng = random.Random(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_net_mask(self, prompt_id: str, hospital_id: str) -> float:
        """Return this hospital's net mask for the given round.

        For every *other* hospital in the configured set, a single
        shared random mask is generated (or retrieved, if the other
        side already triggered generation for this pair/round).  The
        mask is added with sign ``+1`` if ``hospital_id`` is the
        lexicographically **smaller** member of the pair, and ``-1``
        if it is the **larger** member.  The return value is the sum
        of these signed pairwise masks across all other hospitals.

        This method is **thread-safe**: a single
        :class:`threading.Lock` guards both the existence-check and
        the generate-and-store step for each pair as one atomic
        operation.

        Args:
            prompt_id: Identifier for the current evaluation round
                (typically a prompt UUID).
            hospital_id: The hospital requesting its net mask.

        Returns:
            The net additive mask for ``hospital_id`` in this round.

        Raises:
            ValueError: If ``hospital_id`` is not in the configured set.
        """
        if hospital_id not in self._hospital_set:
            raise ValueError(
                f"Hospital '{hospital_id}' is not in the configured set: "
                f"{self._hospital_ids}"
            )

        net_mask = 0.0
        with self._lock:
            round_masks = self._pair_masks.setdefault(prompt_id, {})
            for other_id in self._hospital_ids:
                if other_id == hospital_id:
                    continue

                # Canonical (sorted) pair key — both sides look up the
                # same entry regardless of who asks first.
                pair_key = (
                    (hospital_id, other_id)
                    if hospital_id < other_id
                    else (other_id, hospital_id)
                )

                if pair_key not in round_masks:
                    round_masks[pair_key] = self._rng.uniform(-1.0, 1.0)

                pair_mask = round_masks[pair_key]

                # +1 for the lexicographically smaller side, -1 for the larger.
                sign = 1.0 if hospital_id < other_id else -1.0
                net_mask += sign * pair_mask

        return net_mask

    def discard_round(self, prompt_id: str) -> None:
        """Drop all cached pairwise masks for a completed round.

        Must be called once a round is fully finalized (typically from
        :meth:`SecureAggregationMechanism.unmask`) to prevent mask
        reuse across rounds, which would leak information over
        repeated queries on the same prompt.

        Safe to call on a ``prompt_id`` with no cached masks (no-op).

        Args:
            prompt_id: The round identifier whose masks should be
                discarded.
        """
        with self._lock:
            self._pair_masks.pop(prompt_id, None)
