"""
app/llm/model_factory.py

LLM Factory — returns the correct LLM based on the LLM_PROVIDER setting.

This is the "toggle" between:
  - Google Gemini (cloud API, requires GOOGLE_API_KEY)
  - Groq (fast cloud API, requires GROQ_API_KEY)
  - Ollama (local open-source model, requires Ollama running on your machine)

Why a factory function?
  - One place to change the LLM. Rest of the code just calls get_llm() and doesn't care which model it is.
  - Easy to add more providers (OpenAI, Anthropic, etc.) later.
  - Clean separation of concerns.

How to switch:
  Set LLM_PROVIDER=google, LLM_PROVIDER=groq, or LLM_PROVIDER=ollama in your .env file.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings

# ── Module-level cache ─────────────────────────────────────────────────────────
# One LLM instance per provider (google, groq, ollama), created once and reused.
# Without this, ChatGroq() / ChatGoogleGenerativeAI() would be re-initialized on
# EVERY user request, wasting ~50-100ms each time setting up HTTP clients.
_llm_cache: dict = {}


def _build_llm(provider: str) -> BaseChatModel:
    """Constructs a fresh LLM instance for the given provider string."""
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        print(f"  [LLM] Initializing Google Gemini ({settings.google_model})...")
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=0.3,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        print(f"  [LLM] Initializing Groq ({settings.groq_model})...")
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0.3,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        print(f"  [LLM] Initializing Ollama ({settings.ollama_model})...")
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
        )
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            "Must be 'google', 'groq', or 'ollama'. Check your .env file."
        )


def get_llm(provider_override: str | None = None) -> BaseChatModel:
    """
    Returns a cached LLM instance for the requested provider.

    First call for a given provider builds and caches the object.
    Every subsequent call returns the same cached instance instantly.

    Args:
        provider_override: Optional override ("google", "groq", "ollama").
                           Falls back to LLM_PROVIDER in .env if not given.
    """
    provider = (provider_override or settings.llm_provider).lower()
    if provider not in _llm_cache:
        _llm_cache[provider] = _build_llm(provider)
    return _llm_cache[provider]
