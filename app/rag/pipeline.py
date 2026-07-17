"""
app/rag/pipeline.py

The RAG (Retrieval-Augmented Generation) pipeline — the core of the chatbot.

This file ties everything together:
  1. RETRIEVE: find the top-3 most relevant chunks from Qdrant for the user's question
  2. PROMPT:   inject those chunks as "context" into the LLM prompt
  3. GENERATE: call the LLM and get a grounded answer

Two entry points are provided:
  - ask()        → returns the full answer at once (used by POST /chat)
  - ask_stream() → yields answer tokens one-by-one as they arrive from the LLM
                   (used by POST /chat/stream — makes the UI feel instant)

Speed optimizations applied (v2):
  - System prompt trimmed from ~800 tokens → ~150 tokens (same behavior, fewer tokens sent)
  - ask_stream() uses LangChain's .stream() method so the first token appears
    in ~0.5s instead of the user waiting 4-6s for the full response.
"""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Iterator

from app.core.logger import get_logger
from app.llm.model_factory import get_llm
from app.vectorstore.qdrant_store import retrieve

logger = get_logger(__name__)


# ─── System Prompt (trimmed v2) ───────────────────────────────────────────────
# Reduced from ~800 tokens to ~150 tokens.
# All the core "Aria" behavior is preserved — just stated more concisely.
# Fewer input tokens = less time billed + slightly faster LLM response.
SYSTEM_PROMPT = """\
You are Aria, TechAhead's HR Assistant. Your job is to help employees with HR questions.

Rules:
- Answer ONLY from the <context> block provided. Never use your own training knowledge for HR/policy questions.
- If the answer IS in context: be direct, lead with the key fact, cite exact numbers/dates, and source it naturally (e.g. "As per the Leave Policy...").
- If the answer is NOT in context: say so honestly and direct the user to HR.
- For greetings or small talk: respond warmly in 1–2 sentences.
- Use bullet points only when listing 3+ distinct items. Otherwise use plain prose.
- Never fabricate, estimate, or guess policy details.
- Never repeat the user's question. Never use headers in your response.
- Tone: warm, professional, and direct — like a trusted HR colleague.
"""


def format_context(chunks: list[Document]) -> str:
    """
    Formats the retrieved chunks into a readable context string for the prompt.
    Each chunk is labeled with its source file for traceability.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "Unknown source")
        page = chunk.metadata.get("page", "")
        page_info = f", page {page + 1}" if page != "" else ""
        context_parts.append(
            f"[Chunk {i} — Source: {source}{page_info}]\n{chunk.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


def _build_messages(question: str, chunks: list[Document]) -> list:
    """Builds the [SystemMessage, HumanMessage] list for the LLM."""
    context = format_context(chunks)
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"<context>\n{context}\n</context>\n\nQUESTION: {question}"),
    ]


def ask(question: str, llm_provider: str | None = None) -> dict:
    """
    Full (non-streaming) RAG pipeline.
    Returns the complete answer + source chunks in one shot.
    Used by POST /chat.
    """
    import time
    t0 = time.perf_counter()

    logger.info("New question received: '%s'", question[:80])

    # Step 1: Retrieve top-3 relevant chunks from Qdrant
    logger.info("[Step 1/3] Retrieving top-3 chunks from Qdrant...")
    t1 = time.perf_counter()
    chunks = retrieve(question, top_k=3)
    t2 = time.perf_counter()
    print(f"  [TIMING] Step 1 - Retrieval: {t2-t1:.3f}s")

    if not chunks:
        logger.warning("No chunks retrieved — collection may be empty")
        return {
            "answer": "No relevant documents found. Please make sure documents are ingested first.",
            "sources": [],
        }

    # Step 2: Build prompt
    messages = _build_messages(question, chunks)
    t3 = time.perf_counter()
    print(f"  [TIMING] Step 2 - Prompt build: {t3-t2:.3f}s")

    # Step 3: Call LLM (blocking — waits for full response)
    llm = get_llm(llm_provider)
    response = llm.invoke(messages)
    t4 = time.perf_counter()
    print(f"  [TIMING] Step 3 - LLM generation ({llm_provider or 'default'}): {t4-t3:.3f}s")

    raw = response.content
    answer = (
        " ".join(block["text"] for block in raw if isinstance(block, dict) and block.get("type") == "text")
        if isinstance(raw, list)
        else str(raw)
    )

    sources = [
        {
            "source": chunk.metadata.get("source", "Unknown"),
            "page": chunk.metadata.get("page", None),
            "preview": chunk.page_content[:200] + "...",
        }
        for chunk in chunks
    ]

    print(f"  [TIMING] Total end-to-end: {time.perf_counter()-t0:.3f}s")
    return {"answer": answer, "sources": sources}


def ask_stream(question: str, llm_provider: str | None = None) -> Iterator[str]:
    """
    Streaming RAG pipeline — yields answer tokens one-by-one as they arrive.

    This makes the UI feel instant: the user sees the first word in ~0.5s
    instead of waiting 4-6s for the full response to be assembled.

    Used by POST /chat/stream via FastAPI's StreamingResponse.

    Args:
        question: The user's question.
        llm_provider: Optional override ("google", "groq", "ollama").

    Yields:
        str — individual text tokens from the LLM as they stream in.
    """
    logger.info("[stream] New question: '%s'", question[:80])

    # Retrieve relevant chunks (same as non-streaming path)
    chunks = retrieve(question, top_k=3)

    if not chunks:
        yield "No relevant documents found. Please make sure documents are ingested first."
        return

    # Build prompt messages
    messages = _build_messages(question, chunks)

    # Stream tokens from the LLM — .stream() yields chunks as they arrive
    llm = get_llm(llm_provider)
    for chunk in llm.stream(messages):
        token = chunk.content
        if token:   # skip empty chunks (some providers send empty delimiters)
            yield token
