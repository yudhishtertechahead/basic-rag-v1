"""
app/api/router.py

FastAPI route definitions — renamed from routes.py to match RAG_Chatbot_v1 layout.

Endpoints:
  GET  /health       → Check server status and active LLM model
  POST /ingest       → Trigger document ingestion pipeline
  POST /chat         → Full (non-streaming) RAG answer
  POST /chat/stream  → Streaming RAG answer (token-by-token, default UI path)

Changes from routes.py:
  - /health now reports correct model name for all providers including Groq
  - Removed 600-space print separator noise from /chat
  - Imports from new app/services/ and app/db/ paths
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.logger import get_logger
from app.services.ingestion import ingest_folder
from app.services.rag_service import ask, ask_stream

logger = get_logger(__name__)

router = APIRouter()


# ─── Request / Response Models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """The JSON body expected by POST /chat and POST /chat/stream"""
    question: str
    llm_provider: str | None = None  # Optional override ("groq", "google", "ollama")


class ChatResponse(BaseModel):
    """The JSON body returned by POST /chat"""
    answer: str
    sources: list[dict]
    llm_provider: str


class IngestResponse(BaseModel):
    """The JSON body returned by POST /ingest"""
    message: str
    chunks_stored: int


class HealthResponse(BaseModel):
    """The JSON body returned by GET /health"""
    status: str
    llm_provider: str
    model: str
    qdrant_collection: str


# ─── Helper ───────────────────────────────────────────────────────────────────

def _active_model(provider: str) -> str:
    """Return the active model name for the given provider."""
    p = provider.lower()
    if p == "groq":
        return settings.groq_model
    elif p in ("google", "gemini"):
        return settings.google_model
    elif p == "ollama":
        return settings.ollama_model
    return "unknown"


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Health check endpoint.
    Returns server status and current LLM configuration.
    """
    logger.info("GET /health called")
    provider = settings.llm_provider.lower()
    model = _active_model(provider)
    logger.debug("Active LLM: %s / %s | Qdrant: %s", provider, model, settings.qdrant_collection)

    return HealthResponse(
        status="ok",
        llm_provider=provider,
        model=model,
        qdrant_collection=settings.qdrant_collection,
    )


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_documents():
    """
    Runs the document ingestion pipeline:
      1. Loads all PDFs and Markdown files from docs/
      2. Splits into chunks (500 chars, 100 overlap)
      3. Embeds using all-MiniLM-L6-v2
      4. Upserts into Qdrant (additive — does NOT wipe existing data)

    Run this whenever you add new documents.
    """
    logger.info("POST /ingest called")
    try:
        logger.info("=== Starting Ingestion Pipeline ===")
        total = ingest_folder()
        logger.info("=== Ingestion Complete: %d chunks stored ===", total)

        return IngestResponse(
            message="Documents successfully ingested into Qdrant.",
            chunks_stored=total,
        )

    except FileNotFoundError as e:
        logger.error("Ingestion failed — docs/ issue: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Ingestion failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Main chatbot endpoint (non-streaming).
    Returns the full answer + source chunks in one response.
    """
    active_provider = request.llm_provider or settings.llm_provider
    logger.info("POST /chat | provider=%s | question='%s...'", active_provider, request.question[:60])

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = ask(request.question, llm_provider=request.llm_provider)
        logger.info("POST /chat | answered (%d source chunks)", len(result["sources"]))

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            llm_provider=active_provider,
        )

    except Exception as e:
        logger.error("POST /chat | failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.post("/chat/stream", tags=["Chat"])
def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint — tokens sent to the browser word-by-word.
    Use the /ui page which handles the stream via ReadableStream API.

    Time-to-first-token: ~0.1-0.5s with Groq llama-3.1-8b-instant.
    """
    logger.info("POST /chat/stream | question='%s...'", request.question[:60])

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def token_generator():
        try:
            for token in ask_stream(request.question, llm_provider=request.llm_provider):
                yield token
        except Exception as e:
            logger.error("POST /chat/stream | failed: %s", str(e), exc_info=True)
            yield f"\n\n[Error: {str(e)}]"

    return StreamingResponse(token_generator(), media_type="text/plain")
