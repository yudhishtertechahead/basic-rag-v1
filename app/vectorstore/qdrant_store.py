"""
app/vectorstore/qdrant_store.py

Handles everything related to the Qdrant vector database:
  1. Connecting to Qdrant Cloud
  2. Creating the collection (if it doesn't exist)
  3. Embedding text chunks and storing them
  4. Retrieving the top-K most relevant chunks for a query

What is a vector database?
  Instead of searching by exact keywords, we convert text to "vectors" (arrays of numbers
  that represent meaning). Similar texts get similar vectors. Qdrant stores these vectors
  and lets us find the closest ones to a query — this is "semantic search".

What is a collection?
  In Qdrant, a "collection" is like a table in a SQL database.
  All your document chunks and their vectors are stored in one collection.

Pipeline position:
  Chunks → [this file: embed + store] → Qdrant Cloud
  Query  → [this file: embed + search] → top-3 relevant chunks
"""

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns the embedding model used to convert text into vectors.

    We use Hugging Face's all-MiniLM-L6-v2 model (runs locally, 100% free, no rate limits).
    Embedding dimension: 384 (each text becomes a list of 384 numbers)
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'batch_size': 16}
    )


def get_qdrant_client() -> QdrantClient:
    """
    Creates and returns a connection to Qdrant Cloud.

    Reads QDRANT_URL and QDRANT_API_KEY from .env.

    Alternative: For local Qdrant (Docker), use:
        QdrantClient(host="localhost", port=6333)
    For in-memory Qdrant (testing only, data lost on restart):
        QdrantClient(":memory:")
    """
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def ensure_collection_exists(client: QdrantClient) -> None:
    """
    Creates the Qdrant collection if it doesn't already exist.

    A collection must be created with the correct vector size before storing anything.
    text-embedding-004 produces 768-dimensional vectors.

    Distance.COSINE: measures similarity by the angle between vectors (most common for text)
    Alternative distance metrics:
    - Distance.DOT   → dot product (faster but requires normalized vectors)
    - Distance.EUCLID → Euclidean distance (good for image embeddings)
    """
    existing = [c.name for c in client.get_collections().collections]
    logger.debug("Existing Qdrant collections: %s", existing)

    if settings.qdrant_collection not in existing:
        logger.info("Collection '%s' not found -- creating...", settings.qdrant_collection)
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=384,             # Matches all-MiniLM-L6-v2 output dimension
                distance=Distance.COSINE,
            ),
        )
        logger.info("Collection '%s' created (dim=384, cosine distance)", settings.qdrant_collection)
    else:
        logger.info("Collection '%s' already exists — skipping creation", settings.qdrant_collection)


def store_chunks(chunks: list[Document]) -> None:
    """
    Embeds a list of document chunks and stores them in Qdrant Cloud.

    Steps:
      1. Connect to Qdrant Cloud and ensure the collection exists
      2. Embed and upsert into Qdrant

    Why no batching?
      Since we switched to Hugging Face embeddings (running locally), there are no API
      rate limits! We can embed everything as fast as our CPU allows, and 
      QdrantVectorStore handles the Qdrant upload batching automatically under the hood.
    """
    client = get_qdrant_client()
    embeddings = get_embedding_model()

    logger.info("Connecting to Qdrant Cloud: %s", settings.qdrant_url)
    ensure_collection_exists(client)

    logger.info("Embedding and storing %d chunks using local Hugging Face model...", len(chunks))

    # force_recreate=True wipes any old collection (e.g. the old 3072-dim Gemini one)
    # and rebuilds it with the new 384-dim local embeddings.
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        force_recreate=True,  
    )

    logger.info("All %d chunks stored successfully in Qdrant Cloud collection '%s'", len(chunks), settings.qdrant_collection)


def retrieve(query: str, top_k: int = 3) -> list[Document]:
    """
    Performs semantic search in Qdrant for the most relevant chunks.

    Steps:
      1. Embed the user's query using the same embedding model
      2. Compare query vector against all stored chunk vectors
      3. Return the top_k most similar chunks (by cosine similarity)

    Args:
        query: The user's question as plain text.
        top_k: Number of chunks to retrieve (default: 3, as required by the project spec).

    Returns:
        List of the top_k most relevant Document chunks, each with source metadata.

    Alternative retrieval strategies:
    - MMR (Maximal Marginal Relevance): retrieves diverse results (avoids repetitive chunks)
      Usage: vectorstore.as_retriever(search_type="mmr")
    - Threshold-based: only return chunks above a similarity score
      Usage: vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7})
    """
    client = get_qdrant_client()
    embeddings = get_embedding_model()

    logger.debug("Connecting to Qdrant for retrieval (collection: %s)", settings.qdrant_collection)

    # Connect to the existing collection for querying
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )

    logger.debug("Running similarity search | query='%s...' | top_k=%d", query[:60], top_k)

    # similarity_search embeds the query and finds nearest vectors
    results = vectorstore.similarity_search(query, k=top_k)

    logger.info("Retrieved %d chunk(s) for query", len(results))
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        logger.debug("  Chunk %d: %s — '%s...'", i, src, doc.page_content[:80])

    return results
