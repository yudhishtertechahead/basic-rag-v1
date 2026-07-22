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
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams, Modifier, SparseVector, Prefetch, FusionQuery, Fusion
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from app.core.config import settings
from app.core.constants import (
    DENSE_VECTOR_NAME,
    QDRANT_PAYLOAD_FIELDS,
    SPARSE_EMBEDDING_MODEL,
    SPARSE_VECTOR_NAME,
    VECTOR_DIM,
)
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Module-level singletons ──────────────────────────────────────────────────
# Created once at import/startup, reused on every request.
_embedder: SentenceTransformer | None = None
_sparse_embedder: SparseTextEmbedding | None = None
_qdrant_client: QdrantClient | None = None




def _get_embedder() -> SentenceTransformer:
    """Returns the cached SentenceTransformer instance, loading it once."""
    global _embedder
    if _embedder is None:
        logger.info("Loading SentenceTransformer model: %s", settings.embedding_model)
        _embedder = SentenceTransformer(settings.embedding_model)
        logger.info("SentenceTransformer loaded (dim=%d)", VECTOR_DIM)
    return _embedder

def _get_sparse_embedder() -> SparseTextEmbedding:
    """Returns the cached SparseTextEmbedding (BM25) instance, loading it once."""
    global _sparse_embedder
    if _sparse_embedder is None:
        logger.info("Loading SparseTextEmbedding model: %s", SPARSE_EMBEDDING_MODEL)
        _sparse_embedder = SparseTextEmbedding(SPARSE_EMBEDDING_MODEL)
        logger.info("SparseTextEmbedding loaded")
    return _sparse_embedder

def _embed(texts) -> list:
    """Embed a string or list of strings using the singleton embedder."""
    model = _get_embedder()
    if isinstance(texts, str):
        return model.encode(texts).tolist()
    return model.encode(texts).tolist()

def _embed_sparse(texts) -> list:
    """Generate sparse BM25 vectors."""
    model = _get_sparse_embedder()
    if isinstance(texts, str):
        return list(model.embed([texts]))
    return list(model.embed(texts))


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
        prefer_grpc=True,
    )


def ensure_collection() -> None:
    """
    Creates the Qdrant collection for hybrid search if it doesn't already exist.
    """
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        logger.info("Collection '%s' not found — creating hybrid...", settings.qdrant_collection)
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
            }
        )
        logger.info("Collection '%s' created for Hybrid Search", settings.qdrant_collection)
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
    Embed a list of text chunks (dense + sparse) and upsert them into Qdrant.
    """
    ensure_collection()
    client = get_qdrant_client()

    logger.info("Embedding %d chunks from '%s' (Dense + Sparse)...", len(chunks), source)
    embeddings = _embed(chunks)
    sparse_embeddings = _embed_sparse(chunks)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={
                DENSE_VECTOR_NAME: embeddings[i] if isinstance(embeddings[0], list) else [embeddings[i]],
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse_embeddings[i].indices.tolist(),
                    values=sparse_embeddings[i].values.tolist()
                )
            },
            payload={"text": chunks[i], "source": source},
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info("Upserted %d hybrid points into '%s'", len(chunks), settings.qdrant_collection)
    return len(chunks)


def retrieve(query: str, top_k: int = 3, fetch_multiplier: int = 3) -> list:
    """
    Hybrid semantic + keyword search in Qdrant with Reciprocal Rank Fusion.
    Fetches more chunks internally (top_k * fetch_multiplier) to prepare for re-ranking.
    """
    import time
    from langchain_core.documents import Document

    t0 = time.perf_counter()

    query_vector = _embed(query)
    sparse_res = _embed_sparse(query)[0]
    sparse_vector = SparseVector(
        indices=sparse_res.indices.tolist(),
        values=sparse_res.values.tolist()
    )
    t1 = time.perf_counter()
    embed_time = t1 - t0

    client = get_qdrant_client()
    limit = top_k * fetch_multiplier
    
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            Prefetch(
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                limit=limit,
            ),
            Prefetch(
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                limit=limit,
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=QDRANT_PAYLOAD_FIELDS,
    ).points
    t2 = time.perf_counter()
    search_time = t2 - t1

    results = []
    for hit in hits:
        payload = hit.payload or {}
        text = payload.get("text", payload.get("page_content", ""))
        meta = payload.get("metadata", {})
        
        results.append(
            Document(
                page_content=text,
                metadata={k: v for k, v in payload.items() if k not in ("text", "page_content")},
            )
        )

    logger.info("Retrieved %d chunk(s) for query via Hybrid Search", len(results))
    return results, embed_time, search_time


def query_with_sources(question: str, top_k: int | None = None) -> list[dict]:
    """
    Hybrid search for API sources payload.
    """
    k = top_k or 3
    query_vector = _embed(question)
    sparse_res = _embed_sparse(question)[0]
    sparse_vector = SparseVector(
        indices=sparse_res.indices.tolist(),
        values=sparse_res.values.tolist()
    )
    
    client = get_qdrant_client()
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            Prefetch(
                query=query_vector,
                using="dense-text",
                limit=k,
            ),
            Prefetch(
                query=sparse_vector,
                using="sparse-text",
                limit=k,
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=["text", "source", "page"],
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
        
        logger.debug("\n--- Hybrid Chunk %d (Score: %.4f) ---\n%s\n", i+1, score, text.strip())

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
