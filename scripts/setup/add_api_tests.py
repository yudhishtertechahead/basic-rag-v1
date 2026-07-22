import os

test_api_fixes = """
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

@patch("app.api.router.ingest_folder")
def test_ingest_file_not_found(mock_ingest):
    mock_ingest.side_effect = FileNotFoundError("docs missing")
    response = client.post("/ingest")
    assert response.status_code == 400

@patch("app.api.router.ingest_folder")
def test_ingest_generic_error(mock_ingest):
    mock_ingest.side_effect = Exception("db error")
    response = client.post("/ingest")
    assert response.status_code == 500

@patch("app.api.router.ask")
def test_chat_generic_error(mock_ask):
    mock_ask.side_effect = Exception("llm error")
    response = client.post("/chat", json={"question": "hi"})
    assert response.status_code == 500

@patch("app.api.router.ask_stream")
def test_chat_stream_success(mock_ask_stream):
    mock_ask_stream.return_value = ["A", "B"]
    response = client.post("/chat/stream", json={"question": "hi"})
    assert response.status_code == 200
    assert "A" in response.text

@patch("app.api.router.ask_stream")
def test_chat_stream_error(mock_ask_stream):
    def error_gen(*args, **kwargs):
        yield "A"
        raise Exception("stream broken")
    mock_ask_stream.side_effect = error_gen
    response = client.post("/chat/stream", json={"question": "hi"})
    assert response.status_code == 200
    assert "stream broken" in response.text

def test_chat_stream_empty():
    response = client.post("/chat/stream", json={"question": " "})
    assert response.status_code == 400
"""

with open("tests/test_api.py", "a") as f:
    f.write("\n" + test_api_fixes + "\n")
