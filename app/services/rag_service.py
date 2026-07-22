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

from app.core.constants import RETRIEVAL_FETCH_MULTIPLIER, RETRIEVAL_TOP_K, RERANKER_TOP_K, MEMORY_STRATEGY
from app.core.logger import get_logger
from app.db.vector_store import retrieve
from app.services.llm.factory import get_llm
from app.services.memory import get_history, add_turn
from app.services.reranker import rerank_documents

logger = get_logger(__name__)


def ask(question: str, llm_provider: str | None = None, session_id: str | None = None) -> dict:
    """
    Full (non-streaming) RAG pipeline.
    Returns dict with 'answer' and 'sources'.
    Used by POST /chat.
    """
    t0 = time.perf_counter()
    logger.info("New question: '%s'", question[:80])
    
    # Check history
    history = get_history(session_id)
    llm = get_llm(llm_provider)
    search_query = question
    
    rewrite_time = 0.0
    if history and MEMORY_STRATEGY == "query_rewriting":
        logger.info("[Memory] Rewriting query based on history...")
        t_rewrite = time.perf_counter()
        search_query = llm.rewrite_query(question, history)
        rewrite_time = time.perf_counter() - t_rewrite
        logger.info("[Memory] Rewritten query: '%s' (%.3fs)", search_query, rewrite_time)
    elif history:
        logger.info("[Memory] Using sliding window (no rewrite).")

    # Step 1: Retrieve hybrid chunks
    logger.info("[Step 1/4] Retrieving chunks from Qdrant Hybrid Search...")
    chunks, embed_time, search_time = retrieve(search_query, top_k=RETRIEVAL_TOP_K, fetch_multiplier=RETRIEVAL_FETCH_MULTIPLIER)

    if not chunks:
        logger.warning("No chunks retrieved — collection may be empty")
        answer = "No relevant documents found. Please make sure documents are ingested first."
        if session_id:
            add_turn(session_id, question, answer)
        return {
            "answer": answer,
            "sources": [],
        }

    # Step 2: Re-rank chunks with Cross-Encoder
    logger.info("[Step 2/4] Re-ranking chunks with Cross-Encoder...")
    chunks, rerank_time = rerank_documents(search_query, chunks, top_k=RERANKER_TOP_K)

    # Step 3: Build context string from retrieved chunks
    t_before_context = time.perf_counter()
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} — Source: {chunk.metadata.get('source', 'Unknown')}]\n{chunk.page_content}"
        for i, chunk in enumerate(chunks)
    )
    context_time = time.perf_counter() - t_before_context

    # Step 4: Call LLM (blocking)
    t_before_llm = time.perf_counter()
    answer = llm.generate(question, context, history)
    llm_time = time.perf_counter() - t_before_llm
    
    if session_id:
        add_turn(session_id, question, answer)
    
    total_time = time.perf_counter() - t0

    logger.info("--- TIMING SUMMARY ---")
    if rewrite_time > 0:
        logger.info("[TIMING] Query Rewrite  : %.3fs", rewrite_time)
    logger.info("[TIMING] Embedding      : %.3fs", embed_time)
    logger.info("[TIMING] Qdrant Search  : %.3fs", search_time)
    logger.info("[TIMING] Re-ranking     : %.3fs", rerank_time)
    logger.info("[TIMING] Context Build  : %.3fs", context_time)
    logger.info("[TIMING] LLM Generation : %.3fs", llm_time)
    logger.info("                                             ")
    logger.info("[TIMING] Total Time     : %.3fs", total_time)
    logger.info("----------------------")

    sources = [
        {
            "source": chunk.metadata.get("source", "Unknown"),
            "page": chunk.metadata.get("page", None),
            "preview": chunk.page_content[:200] + "...",
        }
        for chunk in chunks
    ]

    return {"answer": answer, "sources": sources}


def ask_stream(question: str, llm_provider: str | None = None, session_id: str | None = None) -> Iterator[str]:
    """
    Streaming RAG pipeline — yields answer tokens one-by-one as they arrive.
    Used by POST /chat/stream via FastAPI's StreamingResponse.
    """
    logger.info("[stream] New question: '%s'", question[:80])
    t0 = time.perf_counter()
    
    # Check history
    history = get_history(session_id)
    llm = get_llm(llm_provider)
    search_query = question
    
    rewrite_time = 0.0
    if history and MEMORY_STRATEGY == "query_rewriting":
        logger.info("[stream] [Memory] Rewriting query...")
        t_rewrite = time.perf_counter()
        search_query = llm.rewrite_query(question, history)
        rewrite_time = time.perf_counter() - t_rewrite
        logger.info("[stream] [Memory] Rewritten: '%s' (%.3fs)", search_query, rewrite_time)
    elif history:
        logger.info("[stream] [Memory] Using sliding window (no rewrite).")

    # Retrieve relevant chunks
    logger.info("[stream] [Step 1/4] Retrieving chunks...")
    chunks, embed_time, search_time = retrieve(search_query, top_k=RETRIEVAL_TOP_K, fetch_multiplier=RETRIEVAL_FETCH_MULTIPLIER)

    if not chunks:
        answer = "No relevant documents found. Please make sure documents are ingested first."
        if session_id:
            add_turn(session_id, question, answer)
        yield answer
        return

    # Re-rank chunks
    logger.info("[stream] [Step 2/4] Re-ranking chunks with Cross-Encoder...")
    chunks, rerank_time = rerank_documents(search_query, chunks, top_k=RERANKER_TOP_K)

    # Build context string
    logger.info("[stream] [Step 3/4] Building context...")
    t_before_context = time.perf_counter()
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} — Source: {chunk.metadata.get('source', 'Unknown')}]\n{chunk.page_content}"
        for i, chunk in enumerate(chunks)
    )
    context_time = time.perf_counter() - t_before_context

    # Stream tokens from the LLM
    logger.info("[stream] [Step 4/4] Generating answer...")
    
    first_token_received = False
    ttft_time = 0.0
    t_before_llm = time.perf_counter()
    full_answer = []
    
    for token in llm.stream(question, context, history):
        if not first_token_received:
            ttft_time = time.perf_counter() - t_before_llm
            first_token_received = True
        full_answer.append(token)
        yield token
        
    if session_id:
        add_turn(session_id, question, "".join(full_answer))
    
    llm_stream_time = time.perf_counter() - t_before_llm
    total_time = time.perf_counter() - t0

    logger.info("--- TIMING SUMMARY ---")
    if rewrite_time > 0:
        logger.info("[TIMING] Query Rewrite  : %.3fs", rewrite_time)
    logger.info("[TIMING] Embedding      : %.3fs", embed_time)
    logger.info("[TIMING] Qdrant Search  : %.3fs", search_time)
    logger.info("[TIMING] Re-ranking     : %.3fs", rerank_time)
    logger.info("[TIMING] Context Build  : %.3fs", context_time)
    logger.info("[TIMING] TTFT           : %.3fs", ttft_time)
    logger.info("[TIMING] LLM Stream     : %.3fs", llm_stream_time)
    logger.info("[TIMING] Total Time     : %.3fs", total_time)
    logger.info("----------------------")


def retrieve_with_sources(question: str) -> tuple[list[str], list[dict]]:
    """
    Retrieve context chunks with structured source metadata.
    Used by router.py for the SSE sources payload.

    Returns:
        (text_chunks, source_metadata_list)
    """
    t0 = time.perf_counter()
    logger.info("[sources] Retrieving chunks for UI metadata...")
    chunks, embed_time, search_time = retrieve(question, top_k=RETRIEVAL_TOP_K, fetch_multiplier=RETRIEVAL_FETCH_MULTIPLIER)
    chunks, rerank_time = rerank_documents(question, chunks, top_k=RERANKER_TOP_K)
    
    chunks_text = [chunk.page_content for chunk in chunks]
    sources = [
        {
            "filename": chunk.metadata.get("source", "Unknown"),
            "page": chunk.metadata.get("page"),
            "excerpt": chunk.page_content[:200],
        }
        for chunk in chunks
    ]
    t3 = time.perf_counter()
    total_time = t3 - t0

    logger.info("--- TIMING SUMMARY (SSE Sources) ---")
    logger.info("[TIMING] Embedding      : %.3fs", embed_time)
    logger.info("[TIMING] Qdrant Search  : %.3fs", search_time)
    logger.info("[TIMING] Re-ranking     : %.3fs", rerank_time)
    logger.info("[TIMING] Total Prep     : %.3fs", total_time)
    logger.info("------------------------------------")
    
    return chunks_text, sources
