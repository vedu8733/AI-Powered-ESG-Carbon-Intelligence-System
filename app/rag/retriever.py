"""
Query expansion utilities for the ESG RAG pipeline.

This module uses the Groq-backed LLM service to generate multiple,
diverse query formulations that improve recall during retrieval.
"""

from __future__ import annotations

from typing import List

from app.services.llm_service import LLMServiceError, generate_text


_EXPANSION_PROMPT_TEMPLATE = """
You are an ESG and sustainability search assistant.

Given the following user question, generate three short, diverse search
queries that could be used to retrieve relevant ESG and sustainability
regulatory documents.

Return exactly three queries, each on its own line, with no numbering,
no bullet points, and no extra commentary.

User question:
{question}
""".strip()


def expand_query(original_query: str) -> list[str]:
    """
    Expand a user query into multiple search formulations.

    The function attempts to obtain three alternative queries from the
    Groq LLM via the existing service layer. If the LLM call fails, a
    deterministic fallback based on the original query is used.

    Args:
        original_query: The original user question.

    Returns:
        A list of three short query strings.

    Raises:
        ValueError: If the original query is empty.
    """
    if not original_query or not original_query.strip():
        raise ValueError("original_query must not be empty.")

    prompt = _EXPANSION_PROMPT_TEMPLATE.format(question=original_query.strip())

    try:
        raw_response = generate_text(prompt)
    except LLMServiceError:
        base = original_query.strip()
        return [
            base,
            f"ESG regulations related to: {base}",
            f"Sustainability and compliance guidance for: {base}",
        ]

    lines = [line.strip(" \t-•") for line in raw_response.splitlines()]
    queries = [line for line in lines if line]

    if len(queries) < 3:
        base = original_query.strip()
        fallbacks = [
            f"{base} ESG regulations",
            f"{base} sustainability reporting requirements",
            f"{base} carbon and climate disclosure rules",
        ]
        for fb in fallbacks:
            if len(queries) >= 3:
                break
            if fb not in queries:
                queries.append(fb)

    return queries[:3]


