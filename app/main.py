"""
main.py

FastAPI application entry point.

This file creates the FastAPI app, registers all routes, and configures
startup behavior (warmup). Keep this file minimal — logic lives in app/.

API is versioned under /api/v1/ — see app/api/v1/router.py.

Run the server:
    uv run uvicorn main:app --reload
    (main = this file, app = the FastAPI instance below)
"""

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import router as v1_router  # versioned API
from app.api.router import router as legacy_router  # kept for backward compat
from app.core.config import settings

# ─── Create FastAPI App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Aria — TechAhead HR RAG Chatbot",
    description=(
        "A production-grade Retrieval-Augmented Generation chatbot that answers HR "
        "questions from policy documents using Qdrant Hybrid Search and an LLM of your choice.\n\n"
        "**API v1:** All endpoints are under `/api/v1/`."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS Middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Traceability Middleware ──────────────────────────────────────────────────
import uuid
from fastapi import Request

@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next):
    from app.core.context import request_id_var
    req_id = str(uuid.uuid4())[:8]
    token = request_id_var.set(req_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        request_id_var.reset(token)


# ─── Serve static files (chat UI) ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Register Routes ──────────────────────────────────────────────────────────
# v1 API (canonical, versioned)
app.include_router(v1_router, prefix="/api/v1")

# Legacy routes (unversioned) — kept for backward compatibility with old chat UI
# TODO: Remove once static/chat.html is updated to use /api/v1/
app.include_router(legacy_router)


# ─── Startup Warmup ───────────────────────────────────────────────────────────
# Pre-load the embedding model, open Qdrant connection, and warm the active
# LLM provider at server startup. Without this, the first request pays the
# full cold-start cost (~3-4s for model loading + HTTP client init).
@app.on_event("startup")
async def warmup():
    from app.db.vector_store import _get_embedder, get_qdrant_client, retrieve
    from app.services.llm.factory import get_llm

    print("\n[Warmup] Pre-loading SentenceTransformer model into RAM...")
    await asyncio.to_thread(_get_embedder)
    print("[Warmup] Embedding model ready.")

    print("[Warmup] Connecting to Qdrant and warming up retriever...")
    try:
        await asyncio.to_thread(retrieve, "warmup", 1)
    except Exception:
        pass  # warmup failure is non-fatal — first real query will initialize
    print("[Warmup] Qdrant connected and vectorstore ready.")

    print(f"[Warmup] Initializing {settings.llm_provider} LLM provider...")
    await asyncio.to_thread(get_llm, settings.llm_provider)
    print(f"[Warmup] LLM provider ready. Server is fully warmed up!\n")


# ─── Root Endpoint ────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    """Welcome message at the root URL."""
    return {
        "message": "RAG Chatbot API is running!",
        "chat_ui": "/ui",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/ui", tags=["UI"])
def chat_ui():
    """Serves the HTML chat interface at http://localhost:8000/ui"""
    return FileResponse("static/chat.html")


# ─── Run directly (for debugging only) ────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
