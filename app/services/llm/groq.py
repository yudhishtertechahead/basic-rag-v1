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
from app.core.prompt import SYSTEM_PROMPT, REWRITE_PROMPT, build_prompt
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

    def generate(self, query: str, context: str, history: list[dict[str, str]] | None = None) -> str:
        """Blocking call — returns full answer string."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": build_prompt(query, context)})
        
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def stream(self, query: str, context: str, history: list[dict[str, str]] | None = None) -> Iterator[str]:
        """
        True token-by-token streaming via Groq's stream=True.
        Time-to-first-token: ~0.1-0.3s with llama-3.1-8b-instant.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": build_prompt(query, context)})
        
        stream_response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            stream=True,
            messages=messages,
        )
        for chunk in stream_response:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    def rewrite_query(self, query: str, history: list[dict[str, str]]) -> str:
        """Rewrites the query using a fast Groq call based on the history."""
        # Convert history into a readable string for the prompt
        history_text = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)
        prompt = REWRITE_PROMPT.format(history=history_text, query=query)
        
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,  # Low temperature for rewriting
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or query).strip()
