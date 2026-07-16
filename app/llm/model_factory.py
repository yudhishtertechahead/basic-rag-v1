"""
app/llm/model_factory.py

LLM Factory — returns the correct LLM based on the LLM_PROVIDER setting.

This is the "toggle" between:
  - Google Gemini (cloud API, requires GOOGLE_API_KEY)
  - Ollama (local open-source model, requires Ollama running on your machine)

Why a factory function?
  - One place to change the LLM. Rest of the code just calls get_llm() and doesn't care which model it is.
  - Easy to add more providers (OpenAI, Anthropic, etc.) later.
  - Clean separation of concerns.

How to switch:
  Set LLM_PROVIDER=google or LLM_PROVIDER=ollama in your .env file.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings


def get_llm() -> BaseChatModel:
    """
    Returns an LLM instance based on the LLM_PROVIDER setting in .env.

    Returns:
        A LangChain-compatible chat model (BaseChatModel).
        This means any code using this LLM calls .invoke() the same way
        regardless of whether it's Gemini or Ollama.

    Raises:
        ValueError: If LLM_PROVIDER is set to an unsupported value.
    """

    provider = settings.llm_provider.lower()

    if provider == "google":
        # ── Google Gemini (API-based) ──────────────────────────────────────────
        # Requires GOOGLE_API_KEY and an internet connection
        # Model is set via GOOGLE_MODEL in .env (default: gemini-3.5-flash)
        #
        # Alternative: You could also use init_chat_model("google_genai:gemini-3.5-flash")
        # from langchain.chat_models — this is a newer, unified approach
        # that supports many providers with a single import.
        from langchain_google_genai import ChatGoogleGenerativeAI

        print(f"  LLM Provider: Google Gemini ({settings.google_model})")
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=0.3,  # Lower = more focused/deterministic answers
                              # Higher = more creative but less accurate
        )

    elif provider == "ollama":
        # ── Ollama (Local open-source model) ──────────────────────────────────
        # Runs entirely on your machine — no API key, no internet required
        # First install Ollama: https://ollama.com/download
        # Then pull a model: ollama pull llama3
        #
        # Alternative models to try with Ollama:
        # - mistral          → fast, good for Q&A
        # - phi3             → small and efficient
        # - llama3:8b        → good balance of speed and quality
        # - deepseek-r1:7b   → good reasoning
        from langchain_ollama import ChatOllama

        print(f"  LLM Provider: Ollama ({settings.ollama_model})")
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{settings.llm_provider}'. "
            "Must be 'google' or 'ollama'. Check your .env file."
        )
