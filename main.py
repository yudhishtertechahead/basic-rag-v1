"""
main.py

FastAPI application entry point.

This file creates the FastAPI app, registers all routes, and configures
startup behavior. It is the file referenced when running the server:
    uv run uvicorn main:app --reload
    (main = this file, app = the FastAPI instance below)

Why keep main.py minimal?
  - Separation of concerns: routes live in app/api/routes.py
  - Easier to add middleware, CORS, auth, etc. later without cluttering routes
  - The FastAPI 'app' object is the single source of truth for the whole API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

# ─── Create FastAPI App ───────────────────────────────────────────────────────
# title and description appear in the Swagger UI at /docs
app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "A Retrieval-Augmented Generation chatbot that answers questions "
        "from your documents using Qdrant vector search and an LLM of your choice."
    ),
    version="1.0.0",
)

# ─── CORS Middleware ──────────────────────────────────────────────────────────
# Allows the chat.html file (opened locally in browser) to call this API.
# Without this, the browser blocks requests from a different "origin" (file://).
# allow_origins=["*"] means any origin is allowed — fine for local development.
# In production, replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (local HTML file, etc.)
    allow_methods=["*"],       # Allow GET, POST, etc.
    allow_headers=["*"],       # Allow all headers
)

# ─── Serve static files (chat UI) ───────────────────────────────────────────
# Mounts the static/ folder so chat.html is served at /static/chat.html
# We then add a /ui shortcut route below for convenience.
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Register Routes ──────────────────────────────────────────────────────────
# All routes defined in app/api/routes.py are added to the app here.
app.include_router(router)


# ─── Startup Warmup ───────────────────────────────────────────────────────────
# Pre-load the Hugging Face model and open the Qdrant connection at server
# startup. Without this, the very first user request has to pay the full
# 3-4 second cold start cost of loading the 90MB PyTorch model from disk.
@app.on_event("startup")
async def warmup():
    import asyncio
    from app.vectorstore.qdrant_store import get_embedding_model, get_qdrant_client, retrieve
    print("\n[Warmup] Pre-loading embedding model into RAM...")
    await asyncio.to_thread(get_embedding_model)
    print("[Warmup] Embedding model ready.")
    print("[Warmup] Connecting to Qdrant and building vectorstore...")
    # One dummy query to initialize and cache the QdrantVectorStore object
    await asyncio.to_thread(retrieve, "warmup", 1)
    print("[Warmup] Vectorstore ready. Server is fully warmed up!\n")



# ─── Root Endpoint ────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    """
    Welcome message at the root URL.
    Visit /ui for the chat interface, /docs for API documentation.
    """
    return {
        "message": "RAG Chatbot API is running!",
        "chat_ui": "/ui",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/ui", tags=["UI"])
def chat_ui():
    """
    Serves the HTML chat interface.
    Open http://localhost:8000/ui in your browser to chat.
    """
    return FileResponse("static/chat.html")


# ─── Run directly (for debugging only) ───────────────────────────────────────
# Prefer using: uv run uvicorn main:app --reload
# This block is only for: uv run main.py (no hot-reload)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
