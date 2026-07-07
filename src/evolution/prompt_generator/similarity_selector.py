"""Similarity-based candidate selection."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
import numpy as np

__all__ = ["PromptSimilaritySelector"]


class PromptSimilaritySelector:
    """Selects the candidate most novel relative to both the parent and
    the other candidates.

    Combines two signals:
    - distance from the parent prompt (avoids producing copies)
    - average distance from sibling candidates (avoids cluster collapse)

    The combined score ensures the selected candidate is genuinely
    different, not just far from the parent while being identical to
    the other candidates.
    """

    __slots__ = ("_model", "_alpha", "_beta")

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        """Initializes the embedding model and scoring weights.

        Args:
            model_name: Sentence transformer model to use.
            alpha: Weight for parent distance (higher = prefer novelty
                vs parent).
            beta: Weight for inter-candidate distance (higher = prefer
                novelty vs siblings).
        """
        self._model = SentenceTransformer(model_name)
        self._alpha = alpha
        self._beta = beta

    def select(
        self,
        parent_prompt: str,
        candidates: list[str],
    ) -> str:
        """Returns the candidate with the highest combined novelty score.

        Novelty = α × (1 - sim_to_parent) + β × avg(1 - sim_to_siblings)

        For a single candidate, only parent distance is used.

        Args:
            parent_prompt: Original parent prompt.
            candidates: List of candidate prompts.

        Returns:
            The candidate with the highest combined novelty.

        Raises:
            ValueError: If no candidates are supplied.
        """
        if not candidates:
            raise ValueError("candidates must not be empty.")

        # Single candidate: nothing to compare against siblings
        if len(candidates) == 1:
            return candidates[0]

        embeddings = self._model.encode(
            [parent_prompt] + candidates,
            normalize_embeddings=True,
        )

        parent_embedding = embeddings[0]           # shape: (d,)
        candidate_embeddings = embeddings[1:]       # shape: (n, d)

        n = len(candidates)

        # Parent similarity: shape (n,)
        parent_sims = candidate_embeddings @ parent_embedding

        # Inter-candidate similarity: shape (n, n)
        inter_sims = candidate_embeddings @ candidate_embeddings.T
        # Zero diagonal (self-similarity) so it doesn't inflate the average
        np.fill_diagonal(inter_sims, 0.0)
        # Average similarity to other candidates: shape (n,)
        avg_sibling_sims = inter_sims.sum(axis=1) / (n - 1)

        # Combined similarity (lower = more novel)
        combined_sim = self._alpha * parent_sims + self._beta * avg_sibling_sims

        best_index = int(np.argmin(combined_sim))
        return candidates[best_index]

    def similarities(
        self,
        parent_prompt: str,
        candidates: list[str],
    ) -> list[float]:
        """Returns cosine similarities to parent for debugging."""
        if not candidates:
            return []

        embeddings = self._model.encode(
            [parent_prompt] + candidates,
            normalize_embeddings=True,
        )

        parent_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]

        similarities = candidate_embeddings @ parent_embedding

        return similarities.tolist()