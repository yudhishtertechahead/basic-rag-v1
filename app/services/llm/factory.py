"""
app/services/llm/factory.py

LLM Factory — returns the correct provider based on LLM_PROVIDER setting.

Providers:
  - "groq"   → GroqProvider (direct SDK, llama-3.1-8b-instant — fastest)
  - "google" → GeminiProvider (LangChain wrapper)
  - "ollama" → OllamaProvider (LangChain wrapper, local)

How to switch: set LLM_PROVIDER=groq (or google/ollama) in your .env file.

Module-level cache ensures each provider is initialized at most once,
so HTTP clients and model objects are reused across requests.
"""

from app.core.config import settings
from app.core.logger import get_logger
from .base import BaseLLM

logger = get_logger(__name__)

# ── Module-level provider cache ────────────────────────────────────────────────
# One BaseLLM instance per provider name, created once and reused.
_provider_cache: dict[str, BaseLLM] = {}


def get_llm(provider_override: str | None = None) -> BaseLLM:
    """
    Returns a cached BaseLLM instance for the requested provider.

    First call for a given provider builds and caches the object.
    Every subsequent call returns the same instance instantly (~0ms).

    Args:
        provider_override: Optional provider name ("groq", "google", "ollama").
                           Falls back to LLM_PROVIDER in .env if not given.

    Returns:
        A BaseLLM instance with .generate() and .stream() methods.
    """
    provider = (provider_override or settings.llm_provider).lower()

    if provider not in _provider_cache:
        _provider_cache[provider] = _build_provider(provider)

    return _provider_cache[provider]


def _build_provider(provider: str) -> BaseLLM:
    """Constructs a fresh provider instance."""
    if provider == "groq":
        from .groq import GroqProvider
        return GroqProvider()
    elif provider in ("google", "gemini"):
        from .gemini import GeminiProvider
        return GeminiProvider()
    elif provider == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider()
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            "Must be 'groq', 'google', or 'ollama'. Check your .env file."
        )
