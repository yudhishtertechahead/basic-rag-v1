"""
app/api/v1/schemas.py

Pydantic request/response models for the v1 API.

Why separate from router.py?
  - Schemas are the "contract" of the API. Keeping them in their own file
    lets tests, other services, and future SDK generators import them
    without importing the router's FastAPI dependencies.
  - Follows the Single Responsibility Principle: router.py = routing logic,
    schemas.py = data shape validation.

Usage:
    from app.api.v1.schemas import ChatRequest, ChatResponse
"""

from pydantic import BaseModel, Field


# ─── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for POST /api/v1/chat and POST /api/v1/chat/stream"""
    question: str = Field(..., min_length=1, description="The user's question")
    llm_provider: str | None = Field(
        default=None,
        description="Override the default LLM provider: 'groq', 'google', or 'ollama'"
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for tracking conversation history"
    )


class SourceItem(BaseModel):
    """A single source chunk reference returned with a chat answer."""
    source: str
    page: int | None = None
    preview: str


class ChatResponse(BaseModel):
    """Response body for POST /api/v1/chat"""
    answer: str
    sources: list[SourceItem]
    llm_provider: str


# ─── Ingestion ────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response body for POST /api/v1/ingest"""
    message: str
    chunks_stored: int


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /api/v1/health"""
    status: str
    llm_provider: str
    model: str
    qdrant_collection: str
