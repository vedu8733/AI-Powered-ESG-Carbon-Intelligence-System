"""
End-to-end Retrieval-Augmented Generation (RAG) pipeline.

This module orchestrates query expansion, embedding, vector search, and
LLM-based answer generation for ESG & Carbon Intelligence use cases.
"""

from __future__ import annotations

from typing import Optional, Set

from app.rag.embedding_service import EmbeddingService
from app.rag.retriever import expand_query
from app.rag.vector_store import VectorStore
from app.services.llm_service import LLMServiceError, generate_text


class RAGPipeline:
    """
    High-level RAG pipeline for ESG regulatory and sustainability queries.

    Responsibilities:
      - Build FAISS index from ESG documents.
      - Expand user queries into multiple search queries.
      - Retrieve relevant documents.
      - Deduplicate context.
      - Generate strictly grounded LLM answer.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            embedding_service: Optional custom embedding service.
        """
        self._embedding_service: EmbeddingService = (
            embedding_service or EmbeddingService()
        )
        self._vector_store: Optional[VectorStore] = None

    def build_index(self, documents: list[str]) -> None:
        """
        Build FAISS index from ESG documents.

        Args:
            documents: List of ESG document texts.

        Raises:
            ValueError: If documents list is empty.
        """
        if not documents:
            raise ValueError("documents must not be empty.")

        embeddings = self._embedding_service.embed_texts(documents)

        if not embeddings:
            raise ValueError("Failed to compute embeddings.")

        embedding_dim = len(embeddings[0])

        store = VectorStore(embedding_dim=embedding_dim)
        store.add_documents(embeddings=embeddings, docs=documents)

        self._vector_store = store

    def query(self, question: str) -> str:
        """
        Query the RAG pipeline with strict grounding.

        Rules:
        - Must answer only using retrieved context.
        - If context is irrelevant or empty, return controlled fallback.
        """

        if not question or not question.strip():
            raise ValueError("question must not be empty.")

        if self._vector_store is None:
            raise RuntimeError("RAG index has not been built. Call build_index() first.")

        # Expand query into multiple search queries
        expanded_queries = expand_query(question)

        retrieved_docs: list[str] = []
        seen_docs: Set[str] = set()

        for q in expanded_queries:
            query_embedding = self._embedding_service.embed_query(q)
            results = self._vector_store.search(query_embedding, top_k=3)

            for doc in results:
                if doc not in seen_docs:
                    seen_docs.add(doc)
                    retrieved_docs.append(doc)

        # STRICT GROUNDED BEHAVIOR
        if not retrieved_docs:
            return "No relevant ESG regulatory information found in indexed documents."

        joined_context = "\n\n".join(retrieved_docs)

        prompt = f"""
You are an ESG compliance assistant.

You MUST answer strictly using ONLY the provided context below.
If the answer is not explicitly contained in the context,
respond exactly with:

"No relevant ESG regulatory information found in indexed documents."

Context:
{joined_context}

Question:
{question}

Provide a professional and compliance-focused response.
"""

        try:
            answer = generate_text(prompt)
        except LLMServiceError as exc:
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

        return answer.strip()