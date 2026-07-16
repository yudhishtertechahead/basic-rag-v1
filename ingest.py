"""
ingest.py

Standalone script to run the document ingestion pipeline from the command line.

Usage:
    uv run ingest.py

When to run this:
  - First time setup (after adding docs to docs/ folder)
  - Whenever you add/update/remove documents in docs/
  - If you change your Qdrant collection or embedding model

What it does:
  docs/ folder → load → chunk → embed → store in Qdrant Cloud

Alternative: You can also trigger ingestion via the API endpoint:
  POST http://localhost:8000/ingest
  (requires the FastAPI server to be running first)
"""

from app.core.logger import get_logger, log_banner, log_result, log_step, log_success
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_documents
from app.vectorstore.qdrant_store import store_chunks

logger = get_logger(__name__)


def run_ingestion():
    """
    Runs the full document ingestion pipeline.

    Steps:
      1. Load all PDF and Markdown files from docs/
      2. Split them into chunks (500 chars, 100 overlap)
      3. Embed each chunk using Google text-embedding-004
      4. Store all embeddings in Qdrant Cloud
    """
    log_banner("RAG Chatbot — Document Ingestion Pipeline")

    # ── Step 1: Load documents ────────────────────────────────────────────────
    log_step(1, 3, "Loading documents from docs/ folder...")
    documents = load_documents()
    log_success(f"Loaded {len(documents)} document page(s)")

    # ── Step 2: Chunk documents ───────────────────────────────────────────────
    log_step(2, 3, "Splitting documents into chunks...")
    chunks = chunk_documents(documents)
    log_success(f"Created {len(chunks)} chunks")
    log_result("Chunk size", "500 chars | overlap 100 chars")

    # ── Step 3: Embed and store in Qdrant ────────────────────────────────────
    log_step(3, 3, "Embedding and storing chunks in Qdrant Cloud...")
    store_chunks(chunks)
    log_success(f"Stored {len(chunks)} chunks in Qdrant Cloud")

    log_banner("Ingestion Complete!", char="-")
    log_result("Chunks stored", str(len(chunks)))
    log_result("Next step", "uv run uvicorn main:app --reload")


if __name__ == "__main__":
    run_ingestion()
