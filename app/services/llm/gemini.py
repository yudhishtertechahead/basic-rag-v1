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
from app.core.prompt import SYSTEM_PROMPT, build_prompt
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

    def _messages(self, query: str, context: str) -> list:
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_prompt(query, context)),
        ]

    def generate(self, query: str, context: str) -> str:
        response = self._llm.invoke(self._messages(query, context))
        raw = response.content
        if isinstance(raw, list):
            return " ".join(
                block["text"] for block in raw
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return str(raw)

    def stream(self, query: str, context: str) -> Iterator[str]:
        for chunk in self._llm.stream(self._messages(query, context)):
            token = chunk.content
            if token:
                yield token
