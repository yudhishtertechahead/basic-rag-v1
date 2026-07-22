"""
app/services/reranker.py

Local Cross-Encoder reranking service.
Takes a list of candidate documents from Hybrid Search and re-scores them
against the user query for maximum precision.
"""

import time
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from app.core.constants import RERANKER_MODEL
from app.core.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton
_reranker: CrossEncoder | None = None

def _get_reranker() -> CrossEncoder:
    """Returns the cached CrossEncoder instance."""
    global _reranker
    if _reranker is None:
        logger.info("Loading CrossEncoder model: %s", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("CrossEncoder loaded successfully")
    return _reranker

def rerank_documents(query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
    """
    Re-rank a list of LangChain Document objects against the query.
    
    Args:
        query: The user's original question.
        documents: A list of candidate Documents from the vector store.
        top_k: The final number of documents to return after re-ranking.
        
    Returns:
        The sorted top_k Documents.
    """
    if not documents:
        return []
        
    model = _get_reranker()
    
    t0 = time.perf_counter()
    # Format for CrossEncoder: list of pairs (query, text)
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Predict relevance scores
    scores = model.predict(pairs)
    
    # Sort documents based on scores in descending order
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    t1 = time.perf_counter()
    rerank_time = t1 - t0
    
    # Log the new rankings (only the final chunks)
    logger.debug("--- DEBUG LOGGING: RERANKING ---")
    results = []
    for i, (score, doc) in enumerate(scored_docs[:top_k]):
        logger.debug("\n--- Final Reranked Chunk %d (Score: %.4f) ---\n%s\n", i+1, score, doc.page_content.strip())
        results.append(doc)
        
    return results, rerank_time
