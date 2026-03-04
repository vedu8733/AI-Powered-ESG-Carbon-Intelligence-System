"""
Text chunking utilities for the ESG RAG pipeline.

This module implements sentence-aware, overlapping text chunking for
long ESG documents prior to indexing.
"""

from __future__ import annotations

import re
from typing import Final, List


class TextChunker:
    """
    Split long text into reasonably sized, overlapping chunks.

    The goal is to balance retrieval quality and model context limits by
    approximating sentence-aware segmentation with configurable chunk
    size and overlap.
    """

    _MIN_CHUNK_FRACTION: Final[float] = 0.3

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        """
        Initialize the text chunker.

        Args:
            chunk_size: Target maximum length of each chunk (in characters).
            overlap: Number of overlapping characters between consecutive chunks.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self._chunk_size = chunk_size
        self._overlap = overlap

    def split_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        The algorithm:
          - Splits text into sentences using simple punctuation rules.
          - Greedily groups sentences into chunks up to chunk_size.
          - Avoids very small trailing chunks by merging when needed.
          - Applies character-level overlap between final chunks.

        Args:
            text: Input text to split.

        Returns:
            List of chunk strings.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text.strip())
        if not sentences:
            return [text.strip()]

        base_chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sent = sentence.strip()
            if not sent:
                continue

            sent_len = len(sent) + 1  # plus space/newline
            if current and current_len + sent_len > self._chunk_size:
                base_chunks.append(" ".join(current).strip())
                current = [sent]
                current_len = len(sent)
            else:
                current.append(sent)
                current_len += sent_len

            # Handle pathological single sentence longer than chunk_size
            while current and len(" ".join(current)) > self._chunk_size * 1.5:
                long_text = " ".join(current)
                base_chunks.append(long_text[: self._chunk_size].strip())
                remaining = long_text[self._chunk_size :].strip()
                current = [remaining] if remaining else []
                current_len = len(remaining)

        if current:
            base_chunks.append(" ".join(current).strip())

        # Avoid extremely small last chunk by merging with previous if needed.
        min_len = int(self._chunk_size * self._MIN_CHUNK_FRACTION)
        if len(base_chunks) >= 2 and len(base_chunks[-1]) < min_len:
            base_chunks[-2] = (base_chunks[-2] + " " + base_chunks[-1]).strip()
            base_chunks.pop()

        if self._overlap == 0 or len(base_chunks) <= 1:
            return base_chunks

        chunks_with_overlap: list[str] = []
        for idx, chunk in enumerate(base_chunks):
            if idx == 0:
                chunks_with_overlap.append(chunk)
                continue

            prev = chunks_with_overlap[-1]
            prefix = prev[-self._overlap :]
            combined = f"{prefix}{chunk}"

            if len(combined) > self._chunk_size:
                # Trim prefix to respect max chunk size.
                allowed_prefix_len = max(0, self._chunk_size - len(chunk))
                prefix = prefix[-allowed_prefix_len:]
                combined = f"{prefix}{chunk}"

            chunks_with_overlap.append(combined)

        return chunks_with_overlap

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """
        Naive sentence splitter using punctuation heuristics.

        This is intentionally lightweight and does not require any
        external NLP libraries.
        """
        # Split on ., !, ? followed by whitespace or end-of-string.
        pattern = re.compile(r"(?<=[.!?])\s+")
        sentences = pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]


