"""
app/api/routes.py

FastAPI route definitions — the HTTP interface for the chatbot.

Endpoints:
  GET  /health  → Check if the server is running and see which LLM is active
  POST /ingest  → Trigger the document ingestion pipeline (load → chunk → store)
  POST /chat    → Ask a question and get an answer from the RAG pipeline

Why FastAPI?
  - Automatic Swagger UI at /docs (great for testing without writing curl commands)
  - Built-in request/response validation using Pydantic models
  - Very fast and async-ready
  - Alternative: Flask is simpler but less features; Django REST is heavier
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.logger import get_logger
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_documents
from app.rag.pipeline import ask
from app.vectorstore.qdrant_store import store_chunks

logger = get_logger(__name__)

# APIRouter groups related endpoints together.
# In main.py, this router is attached to the FastAPI app.
# Alternative: you could define all routes directly on the FastAPI app instance.
router = APIRouter()


# ─── Request / Response Models ────────────────────────────────────────────────
# Pydantic models define the shape of JSON request bodies and response bodies.
# FastAPI automatically validates incoming JSON against these models.

class ChatRequest(BaseModel):
    """The JSON body expected by POST /chat"""
    question: str  # The user's question


class ChatResponse(BaseModel):
    """The JSON body returned by POST /chat"""
    answer: str                    # The LLM's answer
    sources: list[dict]            # The source chunks used to generate the answer
    llm_provider: str              # Which LLM was used (google or ollama)


class IngestResponse(BaseModel):
    """The JSON body returned by POST /ingest"""
    message: str       # Status message
    chunks_stored: int # How many chunks were embedded and stored


class HealthResponse(BaseModel):
    """The JSON body returned by GET /health"""
    status: str         # "ok"
    llm_provider: str   # Current LLM (google or ollama)
    model: str          # Exact model name being used
    qdrant_collection: str  # Qdrant collection name


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Health check endpoint.

    Returns the server status and current LLM configuration.
    Useful to verify which model is active without reading .env directly.
    """
    logger.info("GET /health called")
    provider = settings.llm_provider.lower()
    model = settings.google_model if provider == "google" else settings.ollama_model
    logger.debug("Active LLM: %s / %s | Qdrant collection: %s", provider, model, settings.qdrant_collection)

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
      1. Loads all PDFs and Markdown files from the docs/ folder
      2. Splits them into chunks
      3. Embeds each chunk using Google embeddings
      4. Stores all vectors in Qdrant Cloud

    Run this endpoint (or `uv run ingest.py`) whenever you add new documents.

    Alternative: This could also be a background task using FastAPI's BackgroundTasks
    so the API responds immediately while ingestion runs in the background.
    """
    logger.info("POST /ingest called")
    try:
        logger.info("=== Starting Ingestion Pipeline ===")

        logger.info("Step 1: Loading documents...")
        documents = load_documents()

        logger.info("Step 2: Chunking documents...")
        chunks = chunk_documents(documents)

        logger.info("Step 3: Storing in Qdrant...")
        store_chunks(chunks)

        logger.info("=== Ingestion Complete: %d chunks stored ===", len(chunks))

        return IngestResponse(
            message="Documents successfully ingested into Qdrant.",
            chunks_stored=len(chunks),
        )

    except FileNotFoundError as e:
        logger.error("Ingestion failed — docs/ issue: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Ingestion failed with unexpected error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Main chatbot endpoint.

    Receives a question, runs the RAG pipeline, and returns:
      - The LLM's answer (grounded in your documents)
      - The source chunks that were used (for transparency)
      - Which LLM provider was used

    Example request body:
        {"question": "What is the leave policy?"}

    Alternative: This could use WebSockets for streaming responses
    so the user sees the answer token-by-token (like ChatGPT).
    """
    logger.info("POST /chat | question='%s...'", request.question[:60])

    if not request.question.strip():
        logger.warning("Empty question received")
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = ask(request.question)
        logger.info("POST /chat | answered successfully (%d source chunks)", len(result['sources']))

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            llm_provider=settings.llm_provider,
        )

    except Exception as e:
        logger.error("POST /chat | failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
