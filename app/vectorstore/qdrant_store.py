"""
app/vectorstore/qdrant_store.py

Handles everything related to the Qdrant vector database:
  1. Connecting to Qdrant Cloud
  2. Creating the collection (if it doesn't exist)
  3. Embedding text chunks and storing them
  4. Retrieving the top-K most relevant chunks for a query

Speed optimizations applied (v2):
  - Embedding: switched from langchain_huggingface.HuggingFaceEmbeddings (wrapper)
    to sentence_transformers.SentenceTransformer loaded directly at module level.
    This removes LangChain abstraction overhead (~150ms) on every query.
  - Retrieval: switched from QdrantVectorStore.similarity_search() (LangChain ORM)
    to client.query_points() (raw Qdrant client). This skips Document serialization
    overhead and returns plain payload text directly (~80ms saved per query).
  - Both objects are module-level singletons — initialized once at startup,
    reused on every request (no reconnection/reload cost).
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Module-level singletons ──────────────────────────────────────────────────
# Both are created once when the module is first imported (at server startup
# thanks to the warmup in main.py) and reused on every subsequent request.

# Direct SentenceTransformer — no LangChain wrapper overhead
_st_model: SentenceTransformer | None = None

# Raw Qdrant client — used for fast query_points() retrieval
_qdrant_client: QdrantClient | None = None

# LangChain-wrapped client — used only during store_chunks() (ingestion)
# Ingestion is a one-time operation so ORM overhead doesn't matter there.
_langchain_embeddings = None


def _get_st_model() -> SentenceTransformer:
    """Returns the cached SentenceTransformer instance, loading it once."""
    global _st_model
    if _st_model is None:
        logger.info("Loading SentenceTransformer model: %s", settings.embedding_model)
        _st_model = SentenceTransformer(settings.embedding_model)
        logger.info("SentenceTransformer model loaded.")
    return _st_model


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    Creates and returns a connection to Qdrant Cloud.
    Cached via lru_cache — only one connection is ever made.
    """
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def _get_langchain_embeddings():
    """
    Returns a LangChain HuggingFaceEmbeddings instance.
    Used ONLY by store_chunks() during ingestion — not on the hot query path.
    """
    global _langchain_embeddings
    if _langchain_embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _langchain_embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 16},
        )
    return _langchain_embeddings


def ensure_collection_exists(client: QdrantClient) -> None:
    """
    Creates the Qdrant collection if it doesn't already exist.
    Vector size 384 matches all-MiniLM-L6-v2 output dimension.
    """
    existing = [c.name for c in client.get_collections().collections]
    logger.debug("Existing Qdrant collections: %s", existing)

    if settings.qdrant_collection not in existing:
        logger.info("Collection '%s' not found — creating...", settings.qdrant_collection)
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=384,               # Matches all-MiniLM-L6-v2 output dimension
                distance=Distance.COSINE,
            ),
        )
        logger.info("Collection '%s' created (dim=384, cosine distance)", settings.qdrant_collection)
    else:
        logger.info("Collection '%s' already exists — skipping creation", settings.qdrant_collection)


def store_chunks(chunks: list[Document]) -> None:
    """
    Embeds a list of document chunks and stores them in Qdrant Cloud.
    Uses QdrantVectorStore.from_documents() — ORM overhead is fine here
    since ingestion is a one-time operation.
    """
    client = get_qdrant_client()
    embeddings = _get_langchain_embeddings()

    logger.info("Connecting to Qdrant Cloud: %s", settings.qdrant_url)
    ensure_collection_exists(client)

    logger.info("Embedding and storing %d chunks...", len(chunks))

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        force_recreate=True,
    )

    logger.info(
        "All %d chunks stored in Qdrant collection '%s'",
        len(chunks), settings.qdrant_collection
    )


def retrieve(query: str, top_k: int = 3) -> list[Document]:
    """
    Performs fast semantic search in Qdrant for the most relevant chunks.

    Speed path (v2):
      1. Embed query with direct SentenceTransformer (no LangChain wrapper)
      2. Search Qdrant with client.query_points() (raw, no ORM)
      3. Wrap results in Document objects so pipeline.py stays unchanged

    Returns:
        List of Document objects with page_content and metadata populated.
    """
    import time
    t0 = time.perf_counter()

    # Step 1: Embed query directly — no LangChain wrapper overhead
    model = _get_st_model()
    query_vector = model.encode(query).tolist()
    t1 = time.perf_counter()
    print(f"    [TIMING]   a) embed query (direct SentenceTransformer): {t1-t0:.3f}s")

    # Step 2: Search Qdrant with raw client call — no ORM overhead
    client = get_qdrant_client()
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
    ).points
    t2 = time.perf_counter()
    print(f"    [TIMING]   b) query_points (raw Qdrant network call): {t2-t1:.3f}s")

    # Step 3: Convert raw payloads → Document objects (pipeline.py unchanged)
    results = [
        Document(
            page_content=hit.payload.get("page_content", hit.payload.get("text", "")),
            metadata={k: v for k, v in hit.payload.items() if k not in ("page_content", "text")},
        )
        for hit in hits
    ]

    logger.info("Retrieved %d chunk(s) for query", len(results))
    return results
