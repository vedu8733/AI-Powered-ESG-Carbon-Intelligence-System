

from __future__ import annotations

from typing import Optional

from app.rag.rag_pipeline import RAGPipeline


class ComplianceAgent:
    """
    High-level ESG compliance agent.

    The agent maintains its own RAG index built from a provided corpus
    of ESG-related documents and exposes a single analyze() method for
    answering user questions.
    """

    def __init__(
        self,
        documents: list[str],
        pipeline: Optional[RAGPipeline] = None,
    ) -> None:
        """
        Initialize the compliance agent.

        Args:
            documents: List of ESG-related documents to build the index from.
            pipeline: Optional preconfigured RAGPipeline instance.

        Raises:
            ValueError: If documents is empty.
        """
        if not documents:
            raise ValueError("documents must not be empty.")

        self._pipeline: RAGPipeline = pipeline or RAGPipeline()
        self._pipeline.build_index(documents)

    def analyze(self, question: str) -> str:
        """
        Analyze a user question using the RAG-backed compliance pipeline.

        Args:
            question: The user question to analyze.

        Returns:
            The generated answer string.
        """
        return self._pipeline.query(question)


