"""
app/db/vector_store.py

Unified vector database layer — moved from app/vectorstore/qdrant_store.py
and aligned with RAG_Chatbot_v1's app/db/vector_store.py.

Key improvements over the old qdrant_store.py:
  - Single _embedder singleton used for BOTH ingest AND query (no dual models)
  - Upsert-based ingest (no force_recreate — re-ingest is additive, not destructive)
  - Added list_documents() and delete_by_source() from RAG_Chatbot_v1
  - query_with_sources() returns structured metadata alongside text
  - Zero LangChain dependency in this module
"""

import os
import uuid

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Module-level singletons ──────────────────────────────────────────────────
# Created once at import/startup, reused on every request.
_embedder: SentenceTransformer | None = None
_qdrant_client: QdrantClient | None = None

VECTOR_DIM = 384  # Matches all-MiniLM-L6-v2 output dimension


def _get_embedder() -> SentenceTransformer:
    """Returns the cached SentenceTransformer instance, loading it once."""
    global _embedder
    if _embedder is None:
        logger.info("Loading SentenceTransformer model: %s", settings.embedding_model)
        _embedder = SentenceTransformer(settings.embedding_model)
        logger.info("SentenceTransformer loaded (dim=%d)", VECTOR_DIM)
    return _embedder


def _embed(texts) -> list:
    """Embed a string or list of strings using the singleton embedder."""
    model = _get_embedder()
    if isinstance(texts, str):
        return model.encode(texts).tolist()
    return model.encode(texts).tolist()


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    Creates and returns a cached connection to Qdrant Cloud.
    lru_cache ensures only one connection is ever made.
    """
    logger.info("Connecting to Qdrant: %s", settings.qdrant_url)
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def ensure_collection() -> None:
    """
    Creates the Qdrant collection if it doesn't already exist.
    Does NOT force_recreate — safe to call before every ingest.
    """
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        logger.info("Collection '%s' not found — creating...", settings.qdrant_collection)
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info("Collection '%s' created (dim=%d, cosine)", settings.qdrant_collection, VECTOR_DIM)
    else:
        logger.info("Collection '%s' already exists — skipping creation", settings.qdrant_collection)


def has_documents() -> bool:
    """Check if the collection exists and already contains points."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        return False
    info = client.get_collection(settings.qdrant_collection)
    return (info.points_count or 0) > 0


def ingest_chunks(chunks: list, source: str = "unknown") -> int:
    """
    Embed a list of text chunks and upsert them into Qdrant.
    Uses client.upsert() — re-ingest is additive, does NOT wipe the collection.

    Args:
        chunks: List of text strings to embed and store.
        source: Source filename to attach as metadata.

    Returns:
        Number of chunks stored.
    """
    ensure_collection()
    client = get_qdrant_client()

    logger.info("Embedding %d chunks from '%s'...", len(chunks), source)
    embeddings = _embed(chunks)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i] if isinstance(embeddings[0], list) else [embeddings[i]],
            payload={"text": chunks[i], "source": source},
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info("Upserted %d points into '%s'", len(chunks), settings.qdrant_collection)
    return len(chunks)


def retrieve(query: str, top_k: int = 3) -> list:
    """
    Fast semantic search in Qdrant.

    Returns a list of LangChain-compatible Document-like objects (dicts with
    page_content + metadata) so that rag_service.py stays unchanged.

    Speed path:
      1. Embed with direct SentenceTransformer (no LangChain wrapper)
      2. client.query_points() raw call (no ORM)
    """
    import time
    from langchain_core.documents import Document

    t0 = time.perf_counter()

    query_vector = _embed(query)
    t1 = time.perf_counter()
    logger.debug("[TIMING] embed query: %.3fs", t1 - t0)

    client = get_qdrant_client()
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    t2 = time.perf_counter()
    logger.debug("[TIMING] query_points: %.3fs", t2 - t1)

    logger.debug("--- DEBUG LOGGING: RETRIEVAL ---")
    logger.debug("Query: %s", query)
    logger.debug("Chunks Returned: %d", len(hits))

    results = []
    for i, hit in enumerate(hits):
        score = hit.score if hasattr(hit, 'score') else 0.0
        payload = hit.payload or {}
        text = payload.get("text", payload.get("page_content", ""))
        logger.debug("\n--- Chunk %d (Score: %.4f) ---\n%s\n", i+1, score, text.strip())
        
        results.append(
            Document(
                page_content=text,
                metadata={k: v for k, v in payload.items() if k not in ("text", "page_content")},
            )
        )

    logger.info("Retrieved %d chunk(s) for query", len(results))
    return results


def query_with_sources(question: str, top_k: int | None = None) -> list[dict]:
    """
    Embed question → search Qdrant → return top-K chunks with metadata.
    Used by rag_service.py for the /chat/stream sources payload.
    """
    k = top_k or 3
    query_vector = _embed(question)
    client = get_qdrant_client()
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=k,
        with_payload=True,
    ).points

    logger.debug("--- DEBUG LOGGING: QUERY WITH SOURCES ---")
    logger.debug("Query: %s", question)
    logger.debug("Chunks Returned: %d", len(hits))

    results = []
    for i, hit in enumerate(hits):
        score = hit.score if hasattr(hit, 'score') else 0.0
        payload = hit.payload or {}
        text = payload.get("text", payload.get("page_content", ""))
        meta = payload.get("metadata", {})
        
        logger.debug("\n--- Chunk %d (Score: %.4f) ---\n%s\n", i+1, score, text.strip())

        results.append({
            "text": text,
            "source": payload.get("source", meta.get("source", "Unknown")),
            "page": payload.get("page", meta.get("page")),
        })
    return results


def list_documents() -> list[dict]:
    """Return unique document sources stored in Qdrant with chunk counts."""
    ensure_collection()
    client = get_qdrant_client()
    results, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        with_payload=True,
        limit=1000,
    )
    source_counts: dict[str, int] = {}
    for point in results:
        payload = point.payload or {}
        meta = payload.get("metadata", {})
        source = payload.get("source", meta.get("source", "Unknown"))
        name = source.split("\\")[-1].split("/")[-1]
        source_counts[name] = source_counts.get(name, 0) + 1
    return [
        {"filename": name, "chunks": count}
        for name, count in sorted(source_counts.items())
    ]


def delete_by_source(source_name: str) -> bool:
    """Delete all Qdrant points whose source filename matches source_name."""
    ensure_collection()
    client = get_qdrant_client()
    results, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        with_payload=True,
        limit=1000,
    )
    ids_to_delete = []
    for point in results:
        payload = point.payload or {}
        meta = payload.get("metadata", {})
        source = payload.get("source", meta.get("source", ""))
        name = source.split("\\")[-1].split("/")[-1]
        if name == source_name:
            ids_to_delete.append(point.id)
    if not ids_to_delete:
        return False
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=ids_to_delete,
    )
    logger.info("Deleted %d point(s) for source '%s'", len(ids_to_delete), source_name)
    return True


# ── Backward-compat aliases (used by old qdrant_store.py shim) ──────────────
def store_chunks(chunks) -> None:
    """
    Backward-compat wrapper for ingest_chunks().
    Accepts LangChain Document objects (with .page_content / .metadata).
    """
    texts = []
    for chunk in chunks:
        if hasattr(chunk, "page_content"):
            texts.append(chunk.page_content)
        else:
            texts.append(str(chunk))

    source = "unknown"
    if chunks and hasattr(chunks[0], "metadata"):
        source = chunks[0].metadata.get("source", "unknown")

    ingest_chunks(texts, source=source)
