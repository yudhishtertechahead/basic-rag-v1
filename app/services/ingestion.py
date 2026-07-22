"""
app/services/ingestion.py

Document ingestion service — merged from app/ingestion/loader.py + chunker.py.
Mirrors RAG_Chatbot_v1's app/services/ingestion.py structure.

Pipeline:
  file/folder → extract text → split into chunks → store in Qdrant via vector_store

Supports:
  - PDF files (.pdf) — via pdfplumber (better table extraction than pypdf)
  - Markdown / plain text (.md, .txt) — raw text read
"""

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.constants import CHUNK_OVERLAP, CHUNK_SIZE, SUPPORTED_EXTENSIONS
from app.core.logger import get_logger
from app.db.vector_store import ingest_chunks

logger = get_logger(__name__)

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


def _extract_text(file_path: str) -> str:
    """Extract raw text from a PDF, Markdown, or TXT file."""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            # Fallback to pypdf if pdfplumber not available
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext in (".md", ".txt"):
        with open(file_path, encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _split_text(text: str) -> list[str]:
    """Split raw text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_text(text)


def ingest_file(file_path: str, source_name: str | None = None) -> int:
    """
    Extract text from a single file, chunk it, and store in Qdrant.

    Args:
        file_path:   Absolute path to the file.
        source_name: Display name stored as metadata (defaults to basename).

    Returns:
        Number of chunks ingested.
    """
    name = source_name or os.path.basename(file_path)
    ext = os.path.splitext(file_path)[-1].lower()
    logger.info("[%s] Loading: %s", ext.upper().lstrip("."), name)

    text = _extract_text(file_path)
    chunks = _split_text(text)

    if not chunks:
        logger.warning("No chunks produced from '%s'", name)
        return 0

    count = ingest_chunks(chunks, source=name)
    logger.info("[OK] Ingested %d chunks from '%s'", count, name)
    return count


def ingest_folder(folder_path: str | None = None) -> int:
    """
    Ingest all supported files from a folder (default: docs/).

    Returns:
        Total number of chunks ingested across all files.
    """
    target = Path(folder_path) if folder_path else DOCS_DIR

    if not target.exists():
        raise FileNotFoundError(
            f"The docs/ folder was not found at: {target}\n"
            "Please create it and add your PDF/Markdown files."
        )

    files = [f for f in target.iterdir() if f.is_file()]
    if not files:
        raise ValueError(f"The folder '{target}' is empty. Please add PDF or Markdown files.")

    logger.info("Found %d file(s) in %s", len(files), target)

    total = 0
    for file_path in files:
        ext = file_path.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            try:
                total += ingest_file(str(file_path))
            except Exception as e:
                logger.error("Failed to ingest '%s': %s", file_path.name, str(e))
        else:
            logger.warning("[SKIP] Unsupported file type '%s': %s", ext, file_path.name)

    logger.info("Ingestion complete: %d total chunks from %s", total, target)
    return total
