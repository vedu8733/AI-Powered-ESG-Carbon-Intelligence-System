"""
PDF loading utilities for the ESG RAG pipeline.

This module provides a small, production-ready wrapper around PyPDF2 for
extracting text from PDF documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from PyPDF2 import PdfReader


class PDFLoader:
    """
    Load and extract text content from PDF files.

    This class is intentionally lightweight and focuses solely on
    filesystem and PDF parsing concerns for RAG ingestion.
    """

    _MIN_TEXT_LENGTH: Final[int] = 10

    def load_pdf(self, file_path: str) -> str:
        """
        Load a PDF file from disk and return its full text content.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Combined text from all pages of the PDF.

        Raises:
            ValueError: If the file does not exist, cannot be read, or
                yields no extractable text.
        """
        path = Path(file_path)
        if not path.is_file():
            raise ValueError(f"PDF file does not exist: {file_path}")

        try:
            with path.open("rb") as fp:
                reader = PdfReader(fp)
                texts: list[str] = []

                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        texts.append(page_text.strip())
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Failed to extract text from PDF: {file_path}") from exc

        combined = "\n\n".join(texts).strip()

        if len(combined) < self._MIN_TEXT_LENGTH:
            raise ValueError(
                f"PDF appears to be empty or contains no extractable text: {file_path}"
            )

        return combined


