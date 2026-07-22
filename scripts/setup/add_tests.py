import os

test_llm_add = """
@patch("app.services.llm.groq.Groq")
def test_groq_methods(mock_groq):
    mock_client = MagicMock()
    mock_groq.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Answer"
    mock_client.chat.completions.create.return_value = mock_response
    
    provider = app.services.llm.groq.GroqProvider()
    assert provider.generate("q", "c") == "Answer"
    
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "A"
    mock_client.chat.completions.create.return_value = [mock_chunk]
    assert list(provider.stream("q", "c")) == ["A"]

@patch("app.services.llm.gemini.ChatGoogleGenerativeAI")
def test_gemini_methods(mock_gemini):
    mock_client = MagicMock()
    mock_gemini.return_value = mock_client
    
    mock_resp = MagicMock()
    mock_resp.content = "Answer"
    mock_client.invoke.return_value = mock_resp
    
    provider = app.services.llm.gemini.GeminiProvider()
    assert provider.generate("q", "c") == "Answer"
    
    mock_client.stream.return_value = [mock_resp]
    assert list(provider.stream("q", "c")) == ["Answer"]

@patch("app.services.llm.ollama.ChatOllama")
def test_ollama_methods(mock_ollama):
    mock_client = MagicMock()
    mock_ollama.return_value = mock_client
    
    mock_resp = MagicMock()
    mock_resp.content = "Answer"
    mock_client.invoke.return_value = mock_resp
    
    provider = app.services.llm.ollama.OllamaProvider()
    assert provider.generate("q", "c") == "Answer"
    
    mock_client.stream.return_value = [mock_resp]
    assert list(provider.stream("q", "c")) == ["Answer"]
"""

test_vector_store_add = """
from app.db.vector_store import ingest_chunks
@patch("app.db.vector_store.get_qdrant_client")
@patch("app.db.vector_store._embed")
def test_ingest_chunks(mock_embed, mock_client):
    mock_embed.return_value = [0.1, 0.2]
    mock_c = MagicMock()
    mock_client.return_value = mock_c
    
    count = ingest_chunks(["chunk1", "chunk2"])
    assert count == 2
"""

test_rag_service_add = """
@patch("app.services.rag_service.retrieve")
@patch("app.services.rag_service.get_llm")
def test_ask_stream(mock_get_llm, mock_retrieve):
    mock_retrieve.return_value = [
        MagicMock(page_content="Policy info", metadata={"source": "test.pdf"})
    ]
    mock_llm = MagicMock()
    mock_llm.stream.return_value = ["Ans", "wer"]
    mock_get_llm.return_value = mock_llm

    stream = list(ask_stream("query"))
    assert "wer" in stream[-1]
"""

def append_to_file(path, content):
    with open(path, "a") as f:
        f.write("\n" + content + "\n")

append_to_file("tests/test_llm.py", test_llm_add)
append_to_file("tests/test_vector_store.py", test_vector_store_add)
append_to_file("tests/test_rag_service.py", test_rag_service_add)
