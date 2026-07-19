"""
ingest.py

Standalone CLI script to run the document ingestion pipeline.

Usage:
    uv run ingest.py

When to run:
  - First time setup (after adding docs to the docs/ folder)
  - Whenever you add, update, or remove documents in docs/
  - If you change the embedding model or Qdrant collection

What it does:
  docs/ → extract text → chunk (500 chars, 100 overlap) → embed (all-MiniLM-L6-v2) → upsert into Qdrant

Note: Re-ingest is ADDITIVE (no force_recreate). Existing chunks from OTHER files
are preserved. Use the API DELETE /documents/{name} to remove a specific document.
"""

from app.core.logger import get_logger, log_banner, log_result, log_step, log_success
from app.services.ingestion import ingest_folder

logger = get_logger(__name__)


def run_ingestion():
    """Runs the full document ingestion pipeline."""
    log_banner("RAG Chatbot — Document Ingestion Pipeline")

    log_step(1, 1, "Loading, chunking, and storing documents from docs/ folder...")
    total = ingest_folder()
    log_success(f"Stored {total} chunks in Qdrant (upsert — existing data preserved)")

    log_banner("Ingestion Complete!", char="-")
    log_result("Chunks stored", str(total))
    log_result("Next step", "uv run uvicorn main:app --reload")


if __name__ == "__main__":
    run_ingestion()
