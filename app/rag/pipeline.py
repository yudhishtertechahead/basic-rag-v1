"""
app/rag/pipeline.py

The RAG (Retrieval-Augmented Generation) pipeline — the core of the chatbot.

This file ties everything together:
  1. RETRIEVE: find the top-3 most relevant chunks from Qdrant for the user's question
  2. PROMPT:   inject those chunks as "context" into the LLM prompt
  3. GENERATE: call the LLM and get a grounded answer

Why RAG instead of just asking the LLM directly?
  LLMs don't know about YOUR documents. RAG gives the LLM relevant excerpts from
  your documents so it can answer based on your specific content — not just its training data.

Alternative approaches available in LangChain:
  - RetrievalQA chain         → older, simpler, single call
  - create_retrieval_chain()  → newer, composable with LCEL (LangChain Expression Language)
  - ConversationalRetrievalChain → adds chat history / memory
  - create_agent() with a retriever tool → agent decides when to search (most flexible)
"""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import get_logger
from app.llm.model_factory import get_llm
from app.vectorstore.qdrant_store import retrieve

logger = get_logger(__name__)


# ─── System Prompt ────────────────────────────────────────────────────────────
# This instructs the LLM on how to behave.
# Tells it to: only use the provided context, admit when it doesn't know.
SYSTEM_PROMPT = """\
You are Aria, a precise and helpful HR Assistant for TechAhead. \
Your sole purpose is to help employees understand company policies, benefits, and workplace guidelines \
by using ONLY the information provided in the <context> block below.

<instructions>

## CORE RULE — Non-negotiable
Your answers MUST be grounded exclusively in the provided <context>.
Do NOT use your own training knowledge to answer policy, benefits, or HR questions.
If the answer is not in the context, say so honestly and guide the user to HR.

## BEHAVIOR BY QUERY TYPE

### Type 1: Policy / HR Question (context IS relevant)
- Answer directly and concisely — lead with the key fact first.
- Include exact numbers, dates, amounts, and thresholds from the context. Never approximate.
- Cite the source naturally: "As per the HR Manual..." or "The Leave Policy states..."
- Use bullet points ONLY when listing 3 or more distinct items or steps.
- For 1–2 item answers, use plain prose — no forced bullet lists.
- Bold the single most important fact in the answer (amount, date, key rule).

### Type 2: Policy / HR Question (context is NOT relevant or empty)
- Do NOT fabricate, estimate, or guess from general knowledge.
- Be honest but warm: "The provided policies don't cover this specifically."
- Always end with a clear next step: "I'd recommend reaching out to HR at [hr@company.com] or your manager directly."

### Type 3: Conversational / Small Talk ("Hi", "Thank you", "How are you?")
- Respond naturally and briefly, like a friendly colleague.
- No need to search policies. Just be warm and human.
- Keep it to 1–2 sentences max.

## FORMATTING
- Never use headers (##) inside your answer — they feel like a document, not a conversation.
- Never repeat or rephrase the user's question in your reply.
- Never start with "Great question!", "Of course!", "Certainly!", or similar filler.
- Keep total response length appropriate to the question — short questions deserve short answers.

## TONE
Warm, professional, and direct. Like a trusted HR colleague, not a legal document.

</instructions>

IMPORTANT REMINDER: Answer only from the <context>. \
If the context does not contain the answer, say so and direct the user to HR. \
Never invent policy details.
"""


def format_context(chunks: list[Document]) -> str:
    """
    Formats the retrieved chunks into a readable context string for the prompt.

    Each chunk is labeled with its source file for traceability.

    Args:
        chunks: List of Document objects returned by the retriever.

    Returns:
        A formatted string ready to be injected into the prompt.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        # Extract source info from metadata (added by the loader)
        source = chunk.metadata.get("source", "Unknown source")
        page = chunk.metadata.get("page", "")
        page_info = f", page {page + 1}" if page != "" else ""

        context_parts.append(
            f"[Chunk {i} — Source: {source}{page_info}]\n{chunk.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def ask(question: str, llm_provider: str | None = None) -> dict:
    import time
    t0 = time.perf_counter()

    logger.info("New question received: '%s'", question[:80])

    # ── Step 1: Retrieve top-3 relevant chunks from Qdrant ────────────────────
    logger.info("[Step 1/4] Retrieving top-3 chunks from Qdrant...")
    t1 = time.perf_counter()
    chunks = retrieve(question, top_k=3)
    t2 = time.perf_counter()
    print(f"  [TIMING] Step 1 - Retrieval (embed query + Qdrant search): {t2-t1:.3f}s")

    if not chunks:
        logger.warning("No chunks retrieved — collection may be empty or not yet ingested")
        return {
            "answer": "No relevant documents found. Please make sure documents are ingested first.",
            "sources": [],
        }

    logger.info("[Step 1/4] Got %d chunk(s)", len(chunks))

    # ── Step 2: Format chunks into a context string ───────────────────────────
    context = format_context(chunks)

    # ── Step 3: Build the prompt ──────────────────────────────────────────────
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {question}"),
    ]
    t3 = time.perf_counter()
    print(f"  [TIMING] Step 2+3 - Context format + prompt build: {t3-t2:.3f}s")

    # ── Step 4: Call the LLM ──────────────────────────────────────────────────
    logger.debug(
        "=== FINAL PROMPT TO LLM ===\n"
        "SYSTEM PROMPT:\n%s\n\n"
        "USER PROMPT:\nCONTEXT:\n%s\n\nQUESTION: %s\n"
        "===========================",
        SYSTEM_PROMPT, context, question
    )

    llm = get_llm(llm_provider)
    response = llm.invoke(messages)
    t4 = time.perf_counter()
    print(f"  [TIMING] Step 4 - LLM generation ({llm_provider or 'default'}): {t4-t3:.3f}s")

    raw = response.content
    if isinstance(raw, list):
        answer = " ".join(
            block["text"] for block in raw
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        answer = str(raw)

    sources = [
        {
            "source": chunk.metadata.get("source", "Unknown"),
            "page": chunk.metadata.get("page", None),
            "preview": chunk.page_content[:200] + "..."
        }
        for chunk in chunks
    ]

    print(f"  [TIMING] Total end-to-end: {time.perf_counter()-t0:.3f}s")

    return {"answer": answer, "sources": sources}
