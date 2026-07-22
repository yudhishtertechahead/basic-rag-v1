import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "RAG Chatbot API is running!"


def test_chat_ui():
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_empty_question():
    response = client.post("/api/v1/chat", json={"question": "   "})
    assert response.status_code == 400
    assert "Question cannot be empty" in response.json()["detail"]


@patch("app.api.v1.router.ask")
def test_chat_success(mock_ask):
    mock_ask.return_value = {
        "answer": "This is a mock answer",
        "sources": [{"source": "test.pdf", "page": None, "preview": "Source context..."}]
    }
    response = client.post("/api/v1/chat", json={"question": "What is the policy?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "This is a mock answer"
    assert len(response.json()["sources"]) == 1


@patch("app.api.v1.router.ingest_folder")
def test_ingest_success(mock_ingest):
    mock_ingest.return_value = 10
    response = client.post("/api/v1/ingest")
    assert response.status_code == 200
    assert response.json()["chunks_stored"] == 10


@patch("app.api.v1.router.ingest_folder")
def test_ingest_file_not_found(mock_ingest):
    mock_ingest.side_effect = FileNotFoundError("docs missing")
    response = client.post("/api/v1/ingest")
    assert response.status_code == 400


@patch("app.api.v1.router.ingest_folder")
def test_ingest_generic_error(mock_ingest):
    mock_ingest.side_effect = Exception("db error")
    response = client.post("/api/v1/ingest")
    assert response.status_code == 500


@patch("app.api.v1.router.ask")
def test_chat_generic_error(mock_ask):
    mock_ask.side_effect = Exception("llm error")
    response = client.post("/api/v1/chat", json={"question": "hi"})
    assert response.status_code == 500


@patch("app.api.v1.router.ask_stream")
def test_chat_stream_success(mock_ask_stream):
    mock_ask_stream.return_value = iter(["A", "B"])
    response = client.post("/api/v1/chat/stream", json={"question": "hi"})
    assert response.status_code == 200
    assert "A" in response.text


@patch("app.api.v1.router.ask_stream")
def test_chat_stream_error(mock_ask_stream):
    def error_gen(*args, **kwargs):
        yield "A"
        raise Exception("stream broken")
    mock_ask_stream.side_effect = error_gen
    response = client.post("/api/v1/chat/stream", json={"question": "hi"})
    assert response.status_code == 200
    assert "stream broken" in response.text


def test_chat_stream_empty():
    response = client.post("/api/v1/chat/stream", json={"question": " "})
    assert response.status_code == 400
