"""
app/services/llm/gemini.py

Google Gemini LLM provider via LangChain's ChatGoogleGenerativeAI.
Kept as LangChain wrapper since the Gemini SDK integration is mature there.
"""

from typing import Iterator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.logger import get_logger
from app.core.prompt import SYSTEM_PROMPT, REWRITE_PROMPT, build_prompt
from .base import BaseLLM

logger = get_logger(__name__)


class GeminiProvider(BaseLLM):
    """
    Google Gemini provider via LangChain's ChatGoogleGenerativeAI.
    Model is read from settings.google_model.
    """

    def __init__(self) -> None:
        logger.info("[LLM] Initializing Google Gemini (%s)...", settings.google_model)
        self._llm = ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=0.3,
        )
        logger.info("[LLM] Gemini ready.")

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
        raw = response.content
        if isinstance(raw, list):
            return " ".join(
                block["text"] for block in raw
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return str(raw)

    def stream(self, query: str, context: str, history: list[dict[str, str]] | None = None) -> Iterator[str]:
        for chunk in self._llm.stream(self._messages(query, context, history)):
            token = chunk.content
            if token:
                yield token

    def rewrite_query(self, query: str, history: list[dict[str, str]]) -> str:
        """Rewrites the query using a fast Gemini call based on the history."""
        history_text = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)
        prompt = REWRITE_PROMPT.format(history=history_text, query=query)
        
        # We can use invoke directly here.
        response = self._llm.invoke([HumanMessage(content=prompt)])
        raw = response.content
        if isinstance(raw, list):
            res_str = " ".join(
                block["text"] for block in raw
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            res_str = str(raw)
        return res_str.strip()
