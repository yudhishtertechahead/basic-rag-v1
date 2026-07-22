import pytest
from unittest.mock import patch, MagicMock
from app.db.vector_store import _embed, retrieve

@patch("app.db.vector_store._get_embedder")
def test_embed(mock_get_embedder):
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
    mock_get_embedder.return_value = mock_model

    res = _embed("test")
    assert res == [0.1, 0.2, 0.3]

@patch("app.db.vector_store.get_qdrant_client")
@patch("app.db.vector_store._embed")
def test_retrieve(mock_embed, mock_get_client):
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_client = MagicMock()
    mock_point = MagicMock()
    mock_point.payload = {"page_content": "hello", "metadata": {"source": "foo"}}
    mock_point.score = 0.99
    
    mock_query_response = MagicMock()
    mock_query_response.points = [mock_point]
    mock_client.query_points.return_value = mock_query_response
    mock_get_client.return_value = mock_client

    chunks, embed_time, search_time = retrieve("test query", top_k=1)
    assert len(chunks) == 1
    assert chunks[0].page_content == "hello"


from app.db.vector_store import ingest_chunks
@patch("app.db.vector_store.get_qdrant_client")
@patch("app.db.vector_store._embed")
def test_ingest_chunks(mock_embed, mock_client):
    mock_embed.return_value = [0.1, 0.2]
    mock_c = MagicMock()
    mock_client.return_value = mock_c
    
    count = ingest_chunks(["chunk1", "chunk2"])
    assert count == 2



@patch("app.db.vector_store.get_qdrant_client")
@patch("app.db.vector_store._embed")
def test_query_with_sources(mock_embed, mock_client):
    mock_embed.return_value = [0.1, 0.2]
    mock_c = MagicMock()
    mock_point = MagicMock()
    mock_point.payload = {"text": "txt", "source": "src"}
    mock_query_response = MagicMock()
    mock_query_response.points = [mock_point]
    mock_c.query_points.return_value = mock_query_response
    mock_client.return_value = mock_c
    
    from app.db.vector_store import query_with_sources
    res = query_with_sources("q", 1)
    assert res[0]["text"] == "txt"

@patch("app.db.vector_store.ensure_collection")
@patch("app.db.vector_store.get_qdrant_client")
def test_list_documents(mock_client, mock_ensure):
    mock_c = MagicMock()
    mock_point = MagicMock()
    mock_point.payload = {"source": "/a/b.md"}
    mock_c.scroll.return_value = ([mock_point], None)
    mock_client.return_value = mock_c
    
    from app.db.vector_store import list_documents
    res = list_documents()
    assert len(res) == 1
    assert res[0]["filename"] == "b.md"

@patch("app.db.vector_store.ensure_collection")
@patch("app.db.vector_store.get_qdrant_client")
def test_delete_by_source(mock_client, mock_ensure):
    mock_c = MagicMock()
    mock_point = MagicMock()
    mock_point.payload = {"source": "/a/b.md"}
    mock_point.id = 1
    mock_c.scroll.return_value = ([mock_point], None)
    mock_client.return_value = mock_c
    
    from app.db.vector_store import delete_by_source
    res = delete_by_source("b.md")
    assert res is True
    mock_c.delete.assert_called_once()
    
    res = delete_by_source("c.md")
    assert res is False

@patch("app.db.vector_store.ingest_chunks")
def test_store_chunks(mock_ingest):
    from app.db.vector_store import store_chunks
    chunk1 = MagicMock()
    chunk1.page_content = "text1"
    chunk1.metadata = {"source": "f1"}
    chunk2 = "text2"
    store_chunks([chunk1, chunk2])
    mock_ingest.assert_called_once()

