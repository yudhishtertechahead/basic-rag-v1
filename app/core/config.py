"""
app/core/config.py

Central configuration file for the entire application.
All settings are read from the .env file using pydantic-settings.

Why pydantic-settings?
- Type-safe: gives an error immediately if a required variable is missing or wrong type
- Clean: all config in one place, not scattered os.getenv() calls across files
- Alternative: you could use plain os.getenv() everywhere, but that's messy at scale
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Reads all environment variables from .env automatically.
    Each field maps to a variable name in the .env file (case-insensitive).
    """

    # ─── Google Gemini (API-based LLM) ────────────────────────────────────────
    # Used when LLM_PROVIDER=google
    google_api_key: str
    google_model: str = "gemini-3.5-flash"

    # ─── Groq (Fast API-based open-source LLMs) ───────────────────────────────
    # Used when LLM_PROVIDER=groq
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # ─── Ollama (Local open-source LLM) ───────────────────────────────────────
    # Used when LLM_PROVIDER=ollama
    # Install Ollama from https://ollama.com and run: ollama pull llama3
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"

    # ─── LLM Provider Toggle ──────────────────────────────────────────────────
    # Set to "google", "groq", or "ollama" in .env to switch between models
    # Alternative: you could also pass this as a query parameter in the API
    llm_provider: str = "google"

    # ─── Qdrant Cloud ─────────────────────────────────────────────────────────
    # Get these from https://cloud.qdrant.io (free tier)
    qdrant_url: str          # e.g. https://xxxx.cloud.qdrant.io
    qdrant_api_key: str      # from the Qdrant Cloud dashboard

    # The name of the collection (like a "table") where vectors are stored
    qdrant_collection: str = "rag_docs"

    # ─── Embedding Model ──────────────────────────────────────────────────────
    # Google's embedding model — used to convert text to vectors
    # Alternative: "nomic-embed-text" via Ollama for fully local embeddings
    embedding_model: str = "models/text-embedding-004"

    # Tells pydantic-settings to read from a .env file in the project root
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Single shared instance — import this anywhere you need config
# Usage: from app.core.config import settings
settings = Settings()
