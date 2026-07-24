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
from app.core.prompt import REWRITE_PROMPT, build_prompt, get_system_prompt
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

    def _messages(self, query: str, context: str, history: list[dict[str, str]] | None = None, prompt_id: str | None = None) -> list:
        from langchain_core.messages import AIMessage
        
        msgs = [SystemMessage(content=get_system_prompt(prompt_id))]
        
        if history:
            for msg in history:
                if msg["role"] == "user":
                    msgs.append(HumanMessage(content=msg["content"]))
                else:
                    msgs.append(AIMessage(content=msg["content"]))
                    
        msgs.append(HumanMessage(content=build_prompt(query, context)))
        
        import json
        msgs_repr = [{"role": m.type, "content": m.content} for m in msgs]
        print(f"\n\033[35m[PROMPT TO LLM]\n{json.dumps(msgs_repr, indent=2)}\033[0m\n")
        
        return msgs

    def generate(self, query: str, context: str, history: list[dict[str, str]] | None = None, prompt_id: str | None = None) -> str:
        response = self._llm.invoke(self._messages(query, context, history, prompt_id))
        return str(response.content)

    def stream(self, query: str, context: str, history: list[dict[str, str]] | None = None, prompt_id: str | None = None) -> Iterator[str]:
        for chunk in self._llm.stream(self._messages(query, context, history, prompt_id)):
            token = chunk.content
            if token:
                yield token

    def rewrite_query(self, query: str, history: list[dict[str, str]]) -> list[str]:
        """Rewrites/decomposes the query into a JSON list of sub-queries."""
        import json, re
        history_text = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)
        prompt = REWRITE_PROMPT.format(history=history_text, query=query)
        
        response = self._llm.invoke([HumanMessage(content=prompt)])
        raw = re.sub(r"```[a-z]*\n?", "", str(response.content)).strip().rstrip("`")
        
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return [str(q).strip() for q in result if str(q).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
        
        return [raw] if raw else [query]
