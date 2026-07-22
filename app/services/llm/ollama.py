"""
app/services/llm/ollama.py

Ollama LLM provider via LangChain's ChatOllama.
Requires Ollama running locally: https://ollama.com
  → ollama pull llama3
"""

from typing import Iterator

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.logger import get_logger
from app.core.prompt import SYSTEM_PROMPT, REWRITE_PROMPT, build_prompt
from .base import BaseLLM

logger = get_logger(__name__)


class OllamaProvider(BaseLLM):
    """
    Ollama local LLM provider via LangChain ChatOllama.
    Model is read from settings.ollama_model (default: llama3).
    """

    def __init__(self) -> None:
        logger.info("[LLM] Initializing Ollama (%s @ %s)...", settings.ollama_model, settings.ollama_base_url)
        self._llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
        )
        logger.info("[LLM] Ollama ready.")

    def _messages(self, query: str, context: str, history: list[dict[str, str]] | None = None) -> list:
        from langchain_core.messages import AIMessage
        
        msgs = [SystemMessage(content=SYSTEM_PROMPT)]
        
        if history:
            for msg in history:
                if msg["role"] == "user":
                    msgs.append(HumanMessage(content=msg["content"]))
                else:
                    msgs.append(AIMessage(content=msg["content"]))
                    
        msgs.append(HumanMessage(content=build_prompt(query, context)))
        return msgs

    def generate(self, query: str, context: str, history: list[dict[str, str]] | None = None) -> str:
        response = self._llm.invoke(self._messages(query, context, history))
        return str(response.content)

    def stream(self, query: str, context: str, history: list[dict[str, str]] | None = None) -> Iterator[str]:
        for chunk in self._llm.stream(self._messages(query, context, history)):
            token = chunk.content
            if token:
                yield token

    def rewrite_query(self, query: str, history: list[dict[str, str]]) -> str:
        """Rewrites the query using a fast Ollama call based on the history."""
        history_text = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)
        prompt = REWRITE_PROMPT.format(history=history_text, query=query)
        
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()
