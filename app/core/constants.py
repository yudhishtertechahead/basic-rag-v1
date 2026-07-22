"""
app/core/constants.py

All hard-coded application constants in one place.

RULE: This file is for VALUES that are intrinsic to the application logic
and do NOT change between environments (dev/staging/prod).

Separation of concerns:
  - .env / config.py   → secrets & environment toggles (API keys, URLs)
  - constants.py       → algorithm parameters, model names, tuning knobs
  - Never scatter magic numbers across individual service files.

Usage:
    from app.core.constants import RETRIEVAL_TOP_K, CHUNK_SIZE
"""

from typing import Final

# ── Embedding & Vector Store ───────────────────────────────────────────────────
# Must match the output dimension of settings.embedding_model (all-MiniLM-L6-v2)
VECTOR_DIM: Final[int] = 384

# BM25 sparse embedding model (loaded via fastembed)
SPARSE_EMBEDDING_MODEL: Final[str] = "Qdrant/bm25"

# Qdrant named vector keys — must match how the collection was created
DENSE_VECTOR_NAME: Final[str] = "dense-text"
SPARSE_VECTOR_NAME: Final[str] = "sparse-text"

# Fields to retrieve from Qdrant payload (keep narrow for low latency)
QDRANT_PAYLOAD_FIELDS: Final[list] = ["text", "source", "page"]


# ── Retrieval Pipeline ─────────────────────────────────────────────────────────
# Number of final chunks passed to the LLM after re-ranking
RETRIEVAL_TOP_K: Final[int] = 3

# How many chunks to fetch from Qdrant before re-ranking (wider net)
# Total fetched = RETRIEVAL_TOP_K * RETRIEVAL_FETCH_MULTIPLIER = 9
RETRIEVAL_FETCH_MULTIPLIER: Final[int] = 3


# ── Re-ranking ────────────────────────────────────────────────────────────────
# Cross-Encoder model for re-ranking retrieved chunks
# Alternatives: cross-encoder/ms-marco-MiniLM-L-12-v2 (larger, slower, more accurate)
RERANKER_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many chunks to keep after re-ranking (usually same as RETRIEVAL_TOP_K)
RERANKER_TOP_K: Final[int] = 3


# ── Document Ingestion ────────────────────────────────────────────────────────
# Character count per chunk (not token count — tokens ~= chars / 4)
CHUNK_SIZE: Final[int] = 500

# How many characters overlap between adjacent chunks to avoid cutting mid-sentence
CHUNK_OVERLAP: Final[int] = 100

# File types supported by the ingestion pipeline
SUPPORTED_EXTENSIONS: Final[tuple] = (".pdf", ".md", ".txt")


# ── Conversation Memory ───────────────────────────────────────────────────────
# Choose between: "sliding_window" (Approach B) or "query_rewriting" (Approach D)
MEMORY_STRATEGY: Final[str] = "query_rewriting"

# How many recent turns (human/ai pairs) to keep in the context window
MEMORY_WINDOW_SIZE: Final[int] = 3
