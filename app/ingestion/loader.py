"""
app/ingestion/loader.py

Responsible for LOADING documents from the docs/ folder.
Supports two file types:
  - PDF files (.pdf) — using PyPDFLoader
  - Markdown files (.md) — using TextLoader

What this step does in the pipeline:
  docs/ folder → raw LangChain Document objects (each with .page_content and .metadata)

Alternative loaders available in LangChain:
  - UnstructuredFileLoader    → handles many file types automatically (Word, HTML, etc.)
  - DirectoryLoader           → load all files in a folder with one call
  - WebBaseLoader             → load content from a URL
  - CSVLoader, JSONLoader     → for structured data
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_core.documents import Document

from app.core.logger import get_logger

logger = get_logger(__name__)


# Path to the folder where user places their PDF/Markdown documents
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


def load_documents() -> list[Document]:
    """
    Scans the docs/ folder and loads all supported files.

    Returns:
        A flat list of LangChain Document objects.
        Each Document has:
          - .page_content: the raw text content
          - .metadata: dict with info like source file path, page number
    """
    documents = []

    logger.debug("Scanning docs directory: %s", DOCS_DIR)

    if not DOCS_DIR.exists():
        logger.error("docs/ folder not found at: %s", DOCS_DIR)
        raise FileNotFoundError(
            f"The 'docs/' folder was not found at: {DOCS_DIR}\n"
            "Please create it and add your PDF/Markdown files."
        )

    files = list(DOCS_DIR.iterdir())
    if not files:
        logger.warning("docs/ folder exists but is empty — add PDF/Markdown files")
        raise ValueError("The 'docs/' folder is empty. Please add some PDF or Markdown files.")

    logger.info("Found %d file(s) in docs/", len(files))

    for file_path in files:
        ext = file_path.suffix.lower()

        # ── PDF files ─────────────────────────────────────────────────────────
        # Decision: .pdf extension → use PDFPlumberLoader (page-aware, extracts metadata, superior table extraction)
        if ext == ".pdf":
            logger.info("[PDF ] Loading: %s", file_path.name)
            # PDFPlumberLoader is significantly better at extracting tables compared to PyPDFLoader,
            # which is critical for documents like Referral Incentive grid tables.
            loader = PDFPlumberLoader(str(file_path))
            loaded = loader.load()
            logger.debug("      -> %d page(s) extracted", len(loaded))
            documents.extend(loaded)

        # ── Markdown / plain-text files ───────────────────────────────────────
        # Decision: .md or .txt extension → use TextLoader (reads as plain text)
        elif ext in (".md", ".txt"):
            logger.info("[MD  ] Loading: %s", file_path.name)
            # TextLoader reads the whole file as one Document
            # Alternative: UnstructuredMarkdownLoader preserves headers/structure better
            loader = TextLoader(str(file_path), encoding="utf-8")
            loaded = loader.load()
            logger.debug("      -> %d document(s) extracted", len(loaded))
            documents.extend(loaded)

        # ── Unsupported file type ─────────────────────────────────────────────
        else:
            logger.warning("[SKIP] Unsupported file type '%s': %s", ext, file_path.name)

    logger.info("Total documents loaded: %d page(s) from docs/", len(documents))
    return documents
