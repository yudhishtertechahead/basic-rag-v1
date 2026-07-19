"""
app/services/rag_service.py

The RAG (Retrieval-Augmented Generation) pipeline — moved from app/rag/pipeline.py
and updated to use the new app/services/llm/ and app/db/vector_store layers.

Two entry points:
  - ask()        → full answer at once (used by POST /chat)
  - ask_stream() → yields tokens one-by-one (used by POST /chat/stream)
"""

import time
from typing import Iterator

from app.core.logger import get_logger
from app.db.vector_store import retrieve, query_with_sources
from app.services.llm.factory import get_llm

logger = get_logger(__name__)


def ask(question: str, llm_provider: str | None = None) -> dict:
    """
    Full (non-streaming) RAG pipeline.
    Returns dict with 'answer' and 'sources'.
    Used by POST /chat.
    """
    t0 = time.perf_counter()
    logger.info("New question: '%s'", question[:80])

    # Step 1: Retrieve top-3 relevant chunks
    logger.info("[Step 1/3] Retrieving top-3 chunks from Qdrant...")
    t1 = time.perf_counter()
    chunks = retrieve(question, top_k=3)
    t2 = time.perf_counter()
    logger.info("[TIMING] Step 1 - Retrieval: %.3fs", t2 - t1)

    if not chunks:
        logger.warning("No chunks retrieved — collection may be empty")
        return {
            "answer": "No relevant documents found. Please make sure documents are ingested first.",
            "sources": [],
        }

    # Step 2: Build context string from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} — Source: {chunk.metadata.get('source', 'Unknown')}]\n{chunk.page_content}"
        for i, chunk in enumerate(chunks)
    )
    t3 = time.perf_counter()
    logger.info("[TIMING] Step 2 - Context build: %.3fs", t3 - t2)

    # Step 3: Call LLM (blocking)
    llm = get_llm(llm_provider)
    answer = llm.generate(question, context)
    t4 = time.perf_counter()
    logger.info("[TIMING] Step 3 - LLM generation (%s): %.3fs", type(llm).__name__, t4 - t3)
    logger.info("[TIMING] Total end-to-end: %.3fs", t4 - t0)

    sources = [
        {
            "source": chunk.metadata.get("source", "Unknown"),
            "page": chunk.metadata.get("page", None),
            "preview": chunk.page_content[:200] + "...",
        }
        for chunk in chunks
    ]

    return {"answer": answer, "sources": sources}


def ask_stream(question: str, llm_provider: str | None = None) -> Iterator[str]:
    """
    Streaming RAG pipeline — yields answer tokens one-by-one as they arrive.

    Makes the UI feel instant: first word appears in ~0.5s instead of
    waiting 4-6s for the full response (or <1s with Groq 8b-instant).

    Used by POST /chat/stream via FastAPI's StreamingResponse.
    """
    logger.info("[stream] New question: '%s'", question[:80])
    t0 = time.perf_counter()

    # Retrieve relevant chunks
    logger.info("[stream] [Step 1/3] Retrieving chunks...")
    t1 = time.perf_counter()
    chunks = retrieve(question, top_k=3)
    t2 = time.perf_counter()
    logger.info("[TIMING] stream - Retrieval: %.3fs", t2 - t1)

    if not chunks:
        yield "No relevant documents found. Please make sure documents are ingested first."
        return

    # Build context string
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} — Source: {chunk.metadata.get('source', 'Unknown')}]\n{chunk.page_content}"
        for i, chunk in enumerate(chunks)
    )

    # Stream tokens from the LLM
    llm = get_llm(llm_provider)
    logger.info("[stream] [Step 3/3] Generating answer...")
    
    first_token_received = False
    for token in llm.stream(question, context):
        if not first_token_received:
            t3 = time.perf_counter()
            logger.info("[TIMING] stream - Time-To-First-Token (TTFT): %.3fs", t3 - t2)
            logger.info("[TIMING] stream - Total Time-To-First-Token: %.3fs", t3 - t0)
            first_token_received = True
        yield token
    
    t_end = time.perf_counter()
    logger.info("[TIMING] stream - Total streaming duration: %.3fs", t_end - t0)


def retrieve_with_sources(question: str) -> tuple[list[str], list[dict]]:
    """
    Retrieve context chunks with structured source metadata.
    Used by router.py for the SSE sources payload.

    Returns:
        (text_chunks, source_metadata_list)
    """
    results = query_with_sources(question)
    chunks_text = [r["text"] for r in results]
    sources = [
        {
            "filename": r.get("source", "Unknown"),
            "page": r.get("page"),
            "excerpt": r["text"][:200],
        }
        for r in results
    ]
    return chunks_text, sources
