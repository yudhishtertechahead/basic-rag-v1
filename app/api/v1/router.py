"""
app/api/v1/router.py

FastAPI route definitions for API version 1.

Endpoints (all prefixed with /api/v1 by main.py):
  GET  /health        → System status and active LLM model
  GET  /prompts       → List available prompt templates
  POST /context       → Retrieve RAG chunks without calling LLM (for UI context panel)
  POST /ingest        → Trigger document ingestion pipeline
  POST /chat          → Full (non-streaming) RAG answer
  POST /chat/stream   → Streaming RAG answer (token-by-token)

Design principles:
  - Routes only handle HTTP concerns: parse request, call service, return response.
  - Business logic lives in app/services/ — never in routers.
  - Request/response shapes live in schemas.py — not here.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.v1.schemas import ChatRequest, ChatResponse, HealthResponse, IngestResponse, SourceItem
from app.core.config import settings
from app.core.logger import get_logger
from app.core.prompt import PROMPT_METADATA
from app.services.ingestion import ingest_folder
from app.services.rag_service import ask, ask_stream, retrieve_with_sources

logger = get_logger(__name__)

router = APIRouter()


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


@router.get("/prompts", tags=["System"])
def list_prompts():
    """
    Returns the list of available prompt templates for the UI dropdown.
    No LLM call involved — pure metadata.
    """
    logger.info("GET /prompts called")
    return {"prompts": PROMPT_METADATA}


@router.post("/context", tags=["Chat"])
def get_context(request: ChatRequest):
    """
    Retrieve relevant document chunks for a question WITHOUT calling the LLM.
    Used by the UI to populate the 'Retrieved Context' panel in parallel
    with the streaming chat request — so the user can see what the LLM sees.

    Returns the same chunks (full text) that will be passed to the LLM.
    """
    logger.info("POST /context | question='%s...'", request.question[:60])

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        chunks_text, sources = retrieve_with_sources(request.question)
        return {
            "chunks": [
                {
                    "index": i + 1,
                    "source": s["filename"],
                    "page": s["page"],
                    "content": chunks_text[i],     # full chunk text (not truncated)
                    "preview": s["excerpt"],
                }
                for i, s in enumerate(sources)
            ]
        }
    except Exception as e:
        logger.error("POST /context | failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {str(e)}")


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_documents():
    """
    Runs the document ingestion pipeline:
      1. Loads all PDFs and Markdown files from docs/
      2. Splits into chunks (500 chars, 100 overlap)
      3. Embeds using all-MiniLM-L6-v2 (dense) + Qdrant/bm25 (sparse)
      4. Upserts into Qdrant (additive — does NOT wipe existing data)

    Run this whenever you add new documents to the docs/ folder.
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
    Returns the full answer + source chunks in one JSON response.
    Use /chat/stream for a faster, token-by-token experience.
    """
    active_provider = request.llm_provider or settings.llm_provider
    logger.info("POST /chat | provider=%s | question='%s...'", active_provider, request.question[:60])

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = ask(
            request.question,
            llm_provider=request.llm_provider,
            session_id=request.session_id,
            prompt_id=request.prompt_id,
        )
        logger.info("POST /chat | answered (%d source chunks) [Session: %s]", len(result["sources"]), request.session_id or "none")

        return ChatResponse(
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
            llm_provider=active_provider,
        )

    except Exception as e:
        logger.error("POST /chat | failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.post("/chat/stream", tags=["Chat"])
def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint — tokens sent to the browser word-by-word.
    Time-to-first-token: ~0.1-0.5s with Groq llama-3.1-8b-instant.
    """
    logger.info("POST /chat/stream | question='%s...'", request.question[:60])

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def token_generator():
        try:
            for token in ask_stream(
                request.question,
                llm_provider=request.llm_provider,
                session_id=request.session_id,
                prompt_id=request.prompt_id,
            ):
                yield token
        except Exception as e:
            logger.error("POST /chat/stream | failed: %s", str(e), exc_info=True)
            yield f"\n\n[Error: {str(e)}]"

    return StreamingResponse(token_generator(), media_type="text/plain")
