"""
FAISS-based in-memory vector store for the RAG pipeline.

This component is responsible for indexing and searching document
embeddings using a local FAISS index. Documents themselves are stored
in memory alongside the index.
"""

from __future__ import annotations

from typing import List

import faiss  # type: ignore[import]
import numpy as np


class VectorStore:
    """
    Lightweight FAISS-backed vector store.

    The store keeps document texts in memory and uses an IndexFlatL2
    index for similarity search over dense embeddings.
    """

    def __init__(self, embedding_dim: int) -> None:
        """
        Initialize the vector store.

        Args:
            embedding_dim: Dimensionality of the embedding vectors.

        Raises:
            ValueError: If embedding_dim is not positive.
        """
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        self._embedding_dim = embedding_dim
        self._index = faiss.IndexFlatL2(embedding_dim)
        self._documents: list[str] = []

    @property
    def size(self) -> int:
        """Return the number of documents stored."""
        return len(self._documents)

    def add_documents(self, embeddings: list[list[float]], docs: list[str]) -> None:
        """
        Add documents and their embeddings to the vector store.

        Args:
            embeddings: List of embedding vectors.
            docs: Corresponding list of document texts.

        Raises:
            ValueError: If lists are empty or lengths mismatch, or if dimensions
                        do not match the configured embedding dimension.
        """
        if not embeddings or not docs:
            raise ValueError("embeddings and docs must not be empty.")
        if len(embeddings) != len(docs):
            raise ValueError("embeddings and docs must have the same length.")

        emb_array = np.asarray(embeddings, dtype="float32")
        if emb_array.ndim != 2 or emb_array.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Embeddings must have shape (n, {self._embedding_dim}), "
                f"got {emb_array.shape}."
            )

        self._index.add(emb_array)
        self._documents.extend(docs)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[str]:
        """
        Search for the most similar documents to the query embedding.

        Args:
            query_embedding: Query embedding vector.
            top_k: Maximum number of documents to return.

        Returns:
            List of document texts corresponding to the nearest neighbors.
        """
        if self.size == 0:
            return []

        if top_k <= 0:
            return []

        query_vector = np.asarray(query_embedding, dtype="float32").reshape(1, -1)

        if query_vector.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Query embedding dimension {query_vector.shape[1]} does not "
                f"match index dimension {self._embedding_dim}."
            )

        _, indices = self._index.search(query_vector, min(top_k, self.size))
        results: list[str] = []
        for idx in indices[0]:
            if 0 <= int(idx) < self.size:
                results.append(self._documents[int(idx)])
        return results


