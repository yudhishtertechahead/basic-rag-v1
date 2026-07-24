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
from app.core.prompt import REWRITE_PROMPT, build_prompt, get_system_prompt
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

    def generate(self, query: str, context: str, history: list[dict[str, str]] | None = None, prompt_id: str | None = None) -> str:
        """Blocking call — returns full answer string."""
        messages = [{"role": "system", "content": get_system_prompt(prompt_id)}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": build_prompt(query, context)})
        
        import json
        print(f"\n\033[35m[PROMPT TO LLM]\n{json.dumps(messages, indent=2)}\033[0m\n")
        
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def stream(self, query: str, context: str, history: list[dict[str, str]] | None = None, prompt_id: str | None = None) -> Iterator[str]:
        """
        True token-by-token streaming via Groq's stream=True.
        Time-to-first-token: ~0.1-0.3s with llama-3.1-8b-instant.
        """
        messages = [{"role": "system", "content": get_system_prompt(prompt_id)}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": build_prompt(query, context)})
        
        import json
        print(f"\n\033[35m[PROMPT TO LLM]\n{json.dumps(messages, indent=2)}\033[0m\n")
        
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

    def rewrite_query(self, query: str, history: list[dict[str, str]]) -> list[str]:
        """
        Rewrites/decomposes the query into a JSON list of sub-queries.
        Returns [] for conversational messages (no retrieval needed).
        Returns [single_query] for single-topic questions.
        Returns [q1, q2, ...] for multi-topic questions.
        """
        import json, re
        history_text = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)
        prompt = REWRITE_PROMPT.format(history=history_text, query=query)
        
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response.choices[0].message.content or "").strip()
        
        # Strip markdown code fences if model wraps it
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`")
        
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                sub_queries = [str(q).strip() for q in result if str(q).strip()]
                print(f"\n\033[36m[DECOMPOSE] Raw: {raw!r}\n[DECOMPOSE] Sub-queries ({len(sub_queries)}): {sub_queries}\033[0m\n")
                return sub_queries
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: treat the raw string as a single query
        print(f"\n\033[33m[DECOMPOSE] JSON parse failed — fallback to raw: {raw!r}\033[0m\n")
        if raw:
            return [raw]
        return [query]  # last resort: use original
