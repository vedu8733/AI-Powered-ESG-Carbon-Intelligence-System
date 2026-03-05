

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from groq import APIError, Groq  # type: ignore[attr-defined]


load_dotenv()

LLM_MODEL_NAME = "llama-3.1-8b-instant"
_CLIENT: Optional[Groq] = None


class LLMServiceError(RuntimeError):
    """Errors raised by the LLM service layer."""


def _get_api_key() -> str:
   
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise LLMServiceError(
            "GROQ_API_KEY is not set. Please configure it as an environment "
            "variable before using the LLM service."
        )
    return api_key


def _get_client() -> Groq:
    """
    Return a singleton Groq client instance.

    The client is created lazily and reused across calls to avoid
    reconnect overhead.
    """
    global _CLIENT
    if _CLIENT is None:
        api_key = _get_api_key()
        _CLIENT = Groq(api_key=api_key)
    return _CLIENT


def generate_text(prompt: str) -> str:
    
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string.")

    if not prompt.strip():
        raise ValueError("prompt must not be empty.")

    client = _get_client()

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
    except APIError as exc:  # type: ignore[misc]
        # Known SDK-level error type: wrap to keep callsites simple.
        raise LLMServiceError(f"Groq API error: {exc}") from exc
    except Exception as exc:
        # Catch-all for network or unexpected issues; keep message generic.
        raise LLMServiceError("Unexpected error while calling Groq LLM.") from exc

    try:
        message = completion.choices[0].message
        content = getattr(message, "content", None)
    except (AttributeError, IndexError, KeyError) as exc:
        raise LLMServiceError("Received malformed response from Groq LLM.") from exc

    if not isinstance(content, str):
        raise LLMServiceError("Groq LLM response content is not a string.")

    return content.strip()

