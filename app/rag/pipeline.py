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
SYSTEM_PROMPT = """You are a friendly and helpful HR and Compliance Assistant.

Rules:
1. FOR POLICY QUESTIONS: If the user asks about company policies, rules, or HR matters, base your answer ONLY on the provided CONTEXT. Cite the source (e.g., "According to the HR Manual...").
2. MISSING POLICY INFO: If they ask a policy question and the context doesn't contain the answer, say EXACTLY: "I don't have enough information in the provided policies to answer this. Please reach out to HR."
3. CONVERSATIONAL CHAT: If the user makes small talk, asks general everyday questions, or says "Hi", answer them naturally and conversationally just like a normal AI assistant. 
4. KEEP IT SHORT: Keep your answers concise and directly to the point. Avoid long paragraphs.
5. FORMATTING: Keep answers clear. Use bullet points for lists and **bold** key terms.
6. POINTS: Use bullet points 
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


def ask(question: str) -> dict:
    """
    Main RAG pipeline: takes a user question, retrieves relevant context,
    and returns an LLM-generated answer grounded in your documents.

    Args:
        question: The user's question as a plain string.

    Returns:
        A dict with:
          - "answer": the LLM's response as a string
          - "sources": list of source chunk metadata (file, page) for transparency

    Pipeline:
        question
          → retrieve(question)          [Qdrant semantic search]
          → format_context(chunks)      [build context string]
          → build messages list         [system prompt + context + question]
          → llm.invoke(messages)        [call LLM]
          → return answer + sources
    """
    logger.info("New question received: '%s'", question[:80])

    # ── Step 1: Retrieve top-3 relevant chunks from Qdrant ────────────────────
    # Alternative: use a LangChain retriever object instead:
    # vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(question)
    logger.info("[Step 1/4] Retrieving top-3 chunks from Qdrant...")
    chunks = retrieve(question, top_k=3)

    if not chunks:
        logger.warning("No chunks retrieved — collection may be empty or not yet ingested")
        return {
            "answer": "No relevant documents found. Please make sure documents are ingested first.",
            "sources": [],
        }

    logger.info("[Step 1/4] ✔ Got %d chunk(s)", len(chunks))

    # ── Step 2: Format chunks into a context string ───────────────────────────
    logger.info("[Step 2/4] Formatting context from retrieved chunks...")
    context = format_context(chunks)
    logger.debug("Context length: %d characters", len(context))

    # ── Step 3: Build the prompt as a list of messages ───────────────────────
    # LangChain uses a list of message objects: SystemMessage, HumanMessage, AIMessage
    # Alternative: use PromptTemplate + ChatPromptTemplate for more structured templating
    logger.info("[Step 3/4] Building prompt (system + context + question)...")
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {question}"),
    ]

    # ── Step 4: Call the LLM (Gemini or Ollama based on .env) ────────────────
    # get_llm() returns whichever model is configured — no change needed here
    logger.info("[Step 4/4] Calling LLM...")
    
    # Console the exact final prompt that goes to the LLM for debugging
    logger.debug(
        "=== FINAL PROMPT TO LLM ===\n"
        "SYSTEM PROMPT:\n%s\n\n"
        "USER PROMPT:\nCONTEXT:\n%s\n\nQUESTION: %s\n"
        "===========================",
        SYSTEM_PROMPT, context, question
    )
    
    llm = get_llm()
    response = llm.invoke(messages)

    # response.content is the LLM's reply.
    # Newer Gemini models return a list of content blocks: [{'type': 'text', 'text': '...'}]
    # Older models / other providers return a plain string.
    # We handle both cases here.
    raw = response.content
    if isinstance(raw, list):
        # Extract text from all text-type blocks and join them
        answer = " ".join(
            block["text"] for block in raw
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        answer = str(raw)  # Already a string

    logger.info("[Step 4/4] LLM response received (%d chars)", len(answer))

    # ── Step 5: Extract source metadata for the response ─────────────────────
    sources = [
        {
            "source": chunk.metadata.get("source", "Unknown"),
            "page": chunk.metadata.get("page", None),
            "preview": chunk.page_content[:200] + "..."  # First 200 chars of chunk
        }
        for chunk in chunks
    ]

    logger.debug("Sources returned: %s", [s["source"] for s in sources])

    return {
        "answer": answer,
        "sources": sources,
    }
