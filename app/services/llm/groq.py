"""
app/services/llm/groq.py

Direct Groq SDK provider — no LangChain wrapper.
Uses llama-3.1-8b-instant by default for ~3-5x faster responses than 70b.

Why direct SDK instead of LangChain ChatGroq?
  - Removes abstraction overhead (~50-150ms per request)
  - Native streaming with stream=True
  - Consistent with RAG_Chatbot_v1's production pattern
"""

from typing import Iterator

from groq import Groq

from app.core.config import settings
from app.core.logger import get_logger
from app.core.prompt import SYSTEM_PROMPT, build_prompt
from .base import BaseLLM

logger = get_logger(__name__)


class GroqProvider(BaseLLM):
    """
    Groq LLM provider using the native groq SDK.
    Model is read from settings.groq_model (default: llama-3.1-8b-instant).
    """

    def __init__(self) -> None:
        logger.info("[LLM] Initializing Groq (%s)...", settings.groq_model)
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model
        logger.info("[LLM] Groq ready.")

    def generate(self, query: str, context: str) -> str:
        """Blocking call — returns full answer string."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(query, context)},
            ],
        )
        return response.choices[0].message.content or ""

    def stream(self, query: str, context: str) -> Iterator[str]:
        """
        True token-by-token streaming via Groq's stream=True.
        Time-to-first-token: ~0.1-0.3s with llama-3.1-8b-instant.
        """
        stream_response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(query, context)},
            ],
        )
        for chunk in stream_response:
            token = chunk.choices[0].delta.content
            if token:
                yield token
