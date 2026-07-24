"""
app/services/llm/base.py

Abstract base class for all LLM providers.
Matches RAG_Chatbot_v1's base.py interface so providers are interchangeable.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Iterator


class BaseLLM(ABC):
    """
    Every LLM provider must implement generate() for blocking calls
    and may override generate_stream() for token-by-token streaming.
    """

    @abstractmethod
    def generate(self, query: str, context: str, history: list[dict[str, str]] | None = None, prompt_id: str | None = None) -> str:
        """
        Blocking generation.
        Returns the full answer string.
        """
        ...

    def stream(self, query: str, context: str, history: list[dict[str, str]] | None = None, prompt_id: str | None = None) -> Iterator[str]:
        """
        Synchronous streaming — yields tokens one-by-one.
        Default: calls generate() and yields the full response as one chunk.
        Override in subclasses for true token-by-token streaming.
        """
        yield self.generate(query, context, history, prompt_id)
        
    @abstractmethod
    def rewrite_query(self, query: str, history: list[dict[str, str]]) -> str:
        """
        Uses the LLM to rewrite a query based on the conversation history (Approach D).
        """
        ...
