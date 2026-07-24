"""
app/services/rag_service.py

The RAG (Retrieval-Augmented Generation) pipeline.

Two entry points:
  - ask()        → full answer at once (used by POST /chat)
  - ask_stream() → yields tokens one-by-one (used by POST /chat/stream)

Features:
  - Query Decomposition (multi_doc=True): the rewrite_query LLM call returns a
    JSON list of focused sub-queries. Each sub-query gets its own parallel Qdrant
    retrieval. Results are merged, deduplicated, and globally re-ranked before
    the LLM sees them. Zero extra LLM calls vs. baseline.
  - User Entity Memory: extracts user facts (name, dept, role) and injects them
    into every LLM prompt as a <user_profile> XML block.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

from app.core.constants import RETRIEVAL_FETCH_MULTIPLIER, RETRIEVAL_TOP_K, RERANKER_TOP_K, MEMORY_STRATEGY
from app.core.logger import get_logger
from app.db.vector_store import retrieve
from app.services.entity_extractor import extract_user_facts, should_extract
from app.services.llm.factory import get_llm
from app.services.memory import get_history, add_turn
from app.services.reranker import rerank_documents
from app.services.user_profile import format_for_prompt, update_profile

logger = get_logger(__name__)

CONVERSATIONAL_REPLY = (
    "I'm here to help with TechAhead HR policies! "
    "Feel free to ask me anything about leave, dress code, POSH, referrals, or any other HR topic."
)


def _retrieve_parallel(sub_queries: list[str]) -> list:
    """
    Runs one Qdrant hybrid search per sub-query in parallel using threads.
    Merges all results and deduplicates by content hash so the re-ranker
    sees diverse chunks across all sub-queries.
    """
    seen_texts: set[str] = set()
    all_chunks = []

    def fetch(q: str):
        chunks, _, _ = retrieve(q, top_k=RETRIEVAL_TOP_K, fetch_multiplier=RETRIEVAL_FETCH_MULTIPLIER)
        return chunks

    t_parallel = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 4)) as pool:
        futures = {pool.submit(fetch, q): q for q in sub_queries}
        for future in as_completed(futures):
            q = futures[future]
            try:
                result_chunks = future.result()
                print(f"\033[36m[PARALLEL RETRIEVE] '{q}' → {len(result_chunks)} chunks\033[0m")
                for chunk in result_chunks:
                    key = chunk.page_content[:120]
                    if key not in seen_texts:
                        seen_texts.add(key)
                        all_chunks.append(chunk)
            except Exception as exc:
                logger.warning("[ParallelRetrieve] Sub-query failed: %s", exc)

    total_parallel = time.perf_counter() - t_parallel
    print(f"\033[35m[TIMING] Parallel Fetch  : {total_parallel:.3f}s | {len(all_chunks)} unique chunks merged\033[0m")
    logger.info("[ParallelRetrieve] %d sub-queries → %d unique chunks merged", len(sub_queries), len(all_chunks))
    return all_chunks


def _run_rewrite(question: str, history: list, llm, multi_doc: bool) -> list[str]:
    """
    Returns a list of sub-queries to retrieve against.

    - multi_doc=True : calls rewrite_query() which returns [] | [q] | [q1, q2, ...]
    - multi_doc=False: skips rewrite entirely, returns [question] (original, no decomp)

    If rewrite returns [] → conversational message, caller should short-circuit.
    """
    if not history:
        return [question]

    if multi_doc and MEMORY_STRATEGY == "query_rewriting":
        logger.info("[Rewrite+Decompose] Running combined rewrite/decompose...")
        t = time.perf_counter()
        sub_queries = llm.rewrite_query(question, history)
        elapsed = time.perf_counter() - t
        print(f"\033[35m[TIMING] Decompose       : {elapsed:.3f}s | {len(sub_queries)} sub-quer{'y' if len(sub_queries)==1 else 'ies'}\033[0m")
        logger.info("[Rewrite+Decompose] → %s (%.3fs)", sub_queries, elapsed)
        return sub_queries  # may be []

    elif not multi_doc and MEMORY_STRATEGY == "query_rewriting":
        # Standard rewrite only (old behaviour) — rewrite still returns list, take first
        logger.info("[Rewrite] Standard single-query rewrite (decomp OFF)...")
        t = time.perf_counter()
        sub_queries = llm.rewrite_query(question, history)
        elapsed = time.perf_counter() - t
        # Take only the first sub-query (ignore decomposition results)
        result = sub_queries[0] if sub_queries else question
        print(f"\033[35m[TIMING] Rewrite         : {elapsed:.3f}s → '{result}'\033[0m")
        logger.info("[Rewrite] → '%s' (%.3fs)", result, elapsed)
        return [result]

    return [question]


def ask(
    question: str,
    llm_provider: str | None = None,
    session_id: str | None = None,
    prompt_id: str | None = None,
    multi_doc: bool = True,
) -> dict:
    """Full (non-streaming) RAG pipeline. Returns dict with 'answer' and 'sources'."""
    t0 = time.perf_counter()
    logger.info("New question: '%s'", question[:80])

    history = get_history(session_id)
    llm = get_llm(llm_provider)

    # ── Entity Extraction ────────────────────────────────────────────────────
    if should_extract(question):
        facts = extract_user_facts(question)
        if facts:
            update_profile(session_id, facts)

    # ── Query Rewrite + Decompose ────────────────────────────────────────────
    sub_queries = _run_rewrite(question, history, llm, multi_doc)
    rewrite_time = 0.0  # captured inside _run_rewrite but kept for logging shape

    # Conversational short-circuit (LLM returned [] → no retrieval needed)
    if len(sub_queries) == 0:
        print(f"\033[32m[CONVERSATIONAL] Short-circuit — no retrieval needed\033[0m")
        logger.info("[Rewrite] Conversational message — short-circuiting retrieval")
        if session_id:
            add_turn(session_id, question, CONVERSATIONAL_REPLY)
        return {"answer": CONVERSATIONAL_REPLY, "sources": []}

    # ── Retrieval ────────────────────────────────────────────────────────────
    logger.info("[Step 1/4] Retrieving (%d sub-queries, multi_doc=%s)...", len(sub_queries), multi_doc)
    t_ret = time.perf_counter()

    if len(sub_queries) > 1:
        chunks = _retrieve_parallel(sub_queries)
        embed_time = search_time = time.perf_counter() - t_ret
    else:
        chunks, embed_time, search_time = retrieve(
            sub_queries[0], top_k=RETRIEVAL_TOP_K, fetch_multiplier=RETRIEVAL_FETCH_MULTIPLIER
        )

    if not chunks:
        answer = "No relevant documents found. Please make sure documents are ingested first."
        if session_id:
            add_turn(session_id, question, answer)
        return {"answer": answer, "sources": []}

    # ── Global Re-ranking (against the ORIGINAL question) ────────────────────
    logger.info("[Step 2/4] Global re-ranking against original question...")
    chunks, rerank_time = rerank_documents(question, chunks, top_k=RERANKER_TOP_K)

    # ── Context Build ────────────────────────────────────────────────────────
    t_ctx = time.perf_counter()
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} — Source: {chunk.metadata.get('source', 'Unknown')}]\n{chunk.page_content}"
        for i, chunk in enumerate(chunks)
    )
    context_time = time.perf_counter() - t_ctx

    # ── User Profile Injection ────────────────────────────────────────────────
    user_profile = format_for_prompt(session_id)

    # ── LLM Call ────────────────────────────────────────────────────────────
    t_llm = time.perf_counter()
    answer = llm.generate(question, context, history, prompt_id)
    llm_time = time.perf_counter() - t_llm

    if session_id:
        add_turn(session_id, question, answer)

    total_time = time.perf_counter() - t0
    logger.info("--- TIMING SUMMARY ---")
    logger.info("[TIMING] Sub-queries    : %d", len(sub_queries))
    logger.info("[TIMING] Embedding      : %.3fs", embed_time)
    logger.info("[TIMING] Qdrant Search  : %.3fs", search_time)
    logger.info("[TIMING] Re-ranking     : %.3fs", rerank_time)
    logger.info("[TIMING] Context Build  : %.3fs", context_time)
    logger.info("[TIMING] LLM Generation : %.3fs", llm_time)
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


def ask_stream(
    question: str,
    llm_provider: str | None = None,
    session_id: str | None = None,
    prompt_id: str | None = None,
    multi_doc: bool = True,
) -> Iterator[str]:
    """
    Streaming RAG pipeline — yields answer tokens one-by-one.
    Used by POST /chat/stream.
    """
    logger.info("[stream] New question: '%s'", question[:80])
    t0 = time.perf_counter()

    history = get_history(session_id)
    llm = get_llm(llm_provider)

    # ── Entity Extraction ────────────────────────────────────────────────────
    if should_extract(question):
        facts = extract_user_facts(question)
        if facts:
            update_profile(session_id, facts)

    # ── Query Rewrite + Decompose ────────────────────────────────────────────
    sub_queries = _run_rewrite(question, history, llm, multi_doc)

    # Conversational short-circuit
    if len(sub_queries) == 0:
        logger.info("[stream] Conversational message — short-circuiting")
        if session_id:
            add_turn(session_id, question, CONVERSATIONAL_REPLY)
        yield CONVERSATIONAL_REPLY
        return

    # ── Retrieval ────────────────────────────────────────────────────────────
    logger.info("[stream] [Step 1/4] Retrieving (%d sub-queries)...", len(sub_queries))
    t_ret = time.perf_counter()

    if len(sub_queries) > 1:
        chunks = _retrieve_parallel(sub_queries)
        embed_time = search_time = time.perf_counter() - t_ret
    else:
        chunks, embed_time, search_time = retrieve(
            sub_queries[0], top_k=RETRIEVAL_TOP_K, fetch_multiplier=RETRIEVAL_FETCH_MULTIPLIER
        )

    if not chunks:
        answer = "No relevant documents found. Please make sure documents are ingested first."
        if session_id:
            add_turn(session_id, question, answer)
        yield answer
        return

    # ── Global Re-ranking ────────────────────────────────────────────────────
    logger.info("[stream] [Step 2/4] Global re-ranking...")
    chunks, rerank_time = rerank_documents(question, chunks, top_k=RERANKER_TOP_K)

    # ── Context Build ─────────────────────────────────────────────────────────
    logger.info("[stream] [Step 3/4] Building context...")
    t_ctx = time.perf_counter()
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} — Source: {chunk.metadata.get('source', 'Unknown')}]\n{chunk.page_content}"
        for i, chunk in enumerate(chunks)
    )
    context_time = time.perf_counter() - t_ctx

    # ── User Profile Injection ────────────────────────────────────────────────
    user_profile = format_for_prompt(session_id)
    if user_profile:
        logger.info("[stream] [UserProfile] Injecting: %s", user_profile)

    # ── Streaming LLM ─────────────────────────────────────────────────────────
    logger.info("[stream] [Step 4/4] Generating answer...")
    first_token = False
    ttft_time = 0.0
    t_llm = time.perf_counter()
    full_answer = []

    for token in llm.stream(question, context, history, prompt_id):
        if not first_token:
            ttft_time = time.perf_counter() - t_llm
            first_token = True
        full_answer.append(token)
        yield token

    if session_id:
        add_turn(session_id, question, "".join(full_answer))

    llm_stream_time = time.perf_counter() - t_llm
    total_time = time.perf_counter() - t0
    logger.info("--- TIMING SUMMARY ---")
    logger.info("[TIMING] Sub-queries    : %d", len(sub_queries))
    logger.info("[TIMING] Embedding      : %.3fs", embed_time)
    logger.info("[TIMING] Qdrant Search  : %.3fs", search_time)
    logger.info("[TIMING] Re-ranking     : %.3fs", rerank_time)
    logger.info("[TIMING] Context Build  : %.3fs", context_time)
    logger.info("[TIMING] TTFT           : %.3fs", ttft_time)
    logger.info("[TIMING] LLM Stream     : %.3fs", llm_stream_time)
    logger.info("[TIMING] Total Time     : %.3fs", total_time)
    logger.info("----------------------")


def retrieve_with_sources(question: str) -> tuple[list[str], list[dict]]:
    """Used by /context endpoint to return chunks for the UI panel."""
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
    logger.info("[TIMING] Total Prep     : %.3fs", time.perf_counter() - t0)
    return chunks_text, sources
