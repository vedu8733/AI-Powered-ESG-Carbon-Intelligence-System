"""
Embedding service for ESG & Carbon Intelligence RAG pipeline.

This module wraps a SentenceTransformer model and exposes a small,
type-safe interface for generating embeddings for documents and queries.
"""

from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """
    Service responsible for generating dense vector embeddings.

    The same model instance is reused within the service instance to
    avoid repeated initialization overhead.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the SentenceTransformer model to load.
        """
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of text documents.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("texts must not be empty.")

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        # Convert to plain Python lists for serialization / downstream use.
        return embeddings.tolist()  # type: ignore[return-value]

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Args:
            query: The query text.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return embedding.tolist()  # type: ignore[return-value]


