"""Similarity-based candidate selection."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
import numpy as np

__all__ = ["PromptSimilaritySelector"]


class PromptSimilaritySelector:
    """Selects the candidate least similar to the parent prompt."""

    __slots__ = ("_model",)

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Initializes the embedding model."""
        self._model = SentenceTransformer(model_name)

    def select(
        self,
        parent_prompt: str,
        candidates: list[str],
    ) -> str:
        """Returns the candidate least similar to the parent.

        Args:
            parent_prompt: Original parent prompt.
            candidates: List of candidate prompts.

        Returns:
            The candidate with the lowest cosine similarity.

        Raises:
            ValueError: If no candidates are supplied.
        """
        if not candidates:
            raise ValueError("candidates must not be empty.")

        embeddings = self._model.encode(
            [parent_prompt] + candidates,
            normalize_embeddings=True,
        )

        parent_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]

        similarities = candidate_embeddings @ parent_embedding

        best_index = int(np.argmin(similarities))

        return candidates[best_index]

    def similarities(
        self,
        parent_prompt: str,
        candidates: list[str],
    ) -> list[float]:
        """Returns cosine similarities for debugging."""
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