"""
app/ingestion/chunker.py

Responsible for SPLITTING large documents into smaller chunks.

Why chunk documents?
  - LLMs have a limited context window (can't read 100 pages at once)
  - Smaller chunks = more precise vector search results
  - We want to retrieve only the relevant 3-4 paragraphs, not entire documents

What this step does in the pipeline:
  Raw Documents → List of smaller chunk Documents

Settings used:
  - chunk_size=500    → each chunk is at most 500 characters long
  - chunk_overlap=100 → 100 characters overlap between adjacent chunks
                        (prevents cutting a sentence right at the boundary)
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import get_logger

logger = get_logger(__name__)


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits a list of LangChain Documents into smaller chunks.

    Args:
        documents: Raw documents returned by the loader.

    Returns:
        A list of smaller Document chunks, each with preserved metadata
        (source file, page number) from the original document.
    """

    # RecursiveCharacterTextSplitter tries to split on natural boundaries:
    # paragraphs (\n\n) → sentences (\n) → words (' ') → characters ('')
    # This keeps sentences intact as much as possible.
    #
    # Alternative splitters:
    # - CharacterTextSplitter      → splits strictly at a fixed character count (less smart)
    # - MarkdownHeaderTextSplitter → splits Markdown by headers (## Introduction, etc.)
    # - TokenTextSplitter          → splits by tokens (more accurate for LLM context limits)
    # - SemanticChunker            → uses embeddings to find natural topic boundaries (slow but smart)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Maximum characters per chunk
        chunk_overlap=100,   # Overlap to avoid cutting mid-sentence
        length_function=len, # How to measure chunk length (by characters here)
    )

    logger.debug(
        "Splitter config: chunk_size=%d, chunk_overlap=%d",
        500, 100
    )

    chunks = splitter.split_documents(documents)

    logger.info(
        "Chunking complete: %d chunks from %d page(s) — avg %.0f chars/chunk",
        len(chunks),
        len(documents),
        sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0,
    )
    logger.debug("Smallest chunk: %d chars | Largest chunk: %d chars",
        min(len(c.page_content) for c in chunks) if chunks else 0,
        max(len(c.page_content) for c in chunks) if chunks else 0,
    )
    return chunks
