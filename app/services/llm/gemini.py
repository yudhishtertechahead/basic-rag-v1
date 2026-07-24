"""
app/services/llm/gemini.py

Google Gemini LLM provider using the official Google GenAI SDK directly.
"""

from typing import Iterator
import json

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logger import get_logger
from app.core.prompt import REWRITE_PROMPT, build_prompt, get_system_prompt
from .base import BaseLLM

logger = get_logger(__name__)


class GeminiProvider(BaseLLM):
    """
    Google Gemini provider via the official 'google-genai' SDK.
    Model is read from settings.google_model.
    """

    def __init__(self) -> None:
        logger.info("[LLM] Initializing Google Gemini (%s) directly...", settings.google_model)
        self.client = genai.Client(api_key=settings.google_api_key)
        self.model_name = settings.google_model
        logger.info("[LLM] Gemini ready.")

    def _build_contents(self, query: str, context: str, history: list[dict[str, str]] | None = None) -> list[types.Content]:
        contents = []
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
                )
        
        # Append the final formatted prompt
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=build_prompt(query, context))])
        )
        return contents

    def _get_config(self, prompt_id: str | None = None) -> types.GenerateContentConfig:
        system_instruction = get_system_prompt(prompt_id)
        
        # Logging the prompt context just like before
        print(f"\n\033[35m[PROMPT TO LLM - GEMINI NATIVE]\nSystem: {system_instruction[:100]}...\033[0m\n")
        
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
        )

    def generate(self, query: str, context: str, history: list[dict[str, str]] | None = None, prompt_id: str | None = None) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=self._build_contents(query, context, history),
            config=self._get_config(prompt_id),
        )
        return response.text or ""

    def stream(self, query: str, context: str, history: list[dict[str, str]] | None = None, prompt_id: str | None = None) -> Iterator[str]:
        response = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=self._build_contents(query, context, history),
            config=self._get_config(prompt_id),
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def rewrite_query(self, query: str, history: list[dict[str, str]]) -> list[str]:
        """Rewrites/decomposes the query into a JSON list of sub-queries."""
        import json, re
        history_text = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)
        prompt = REWRITE_PROMPT.format(history=history_text, query=query)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        raw = (response.text or "").strip()
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`")
        
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                sub_queries = [str(q).strip() for q in result if str(q).strip()]
                print(f"\n\033[36m[DECOMPOSE][Gemini] Raw: {raw!r}\n[DECOMPOSE] Sub-queries ({len(sub_queries)}): {sub_queries}\033[0m\n")
                return sub_queries
        except (json.JSONDecodeError, ValueError):
            pass
        
        print(f"\n\033[33m[DECOMPOSE][Gemini] JSON parse failed — fallback: {raw!r}\033[0m\n")
        return [raw] if raw else [query]
