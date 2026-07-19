import os

test_vector_store_add2 = """
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
"""

test_api_add2 = """
@patch("app.api.router.ingest_file")
@patch("builtins.open")
def test_upload_file(mock_open, mock_ingest):
    mock_ingest.return_value = 5
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    response = client.post("/ingest/upload", files={"file": ("test.pdf", b"pdfcontent", "application/pdf")})
    assert response.status_code == 200
    assert response.json()["chunks"] == 5

@patch("app.api.router.list_documents")
def test_get_docs(mock_list):
    mock_list.return_value = [{"filename": "f.md", "chunks": 5}]
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json()["documents"][0]["filename"] == "f.md"

@patch("app.api.router.delete_by_source")
def test_delete_doc(mock_del):
    mock_del.return_value = True
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    response = client.delete("/documents/f.md")
    assert response.status_code == 200
    
    mock_del.return_value = False
    response = client.delete("/documents/none.md")
    assert response.status_code == 404
"""

def append_to_file(path, content):
    with open(path, "a") as f:
        f.write("\n" + content + "\n")

append_to_file("tests/test_vector_store.py", test_vector_store_add2)
append_to_file("tests/test_api.py", test_api_add2)
