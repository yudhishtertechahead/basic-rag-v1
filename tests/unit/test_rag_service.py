import pytest
from unittest.mock import patch, MagicMock
from app.services.rag_service import ask, ask_stream

@patch("app.services.rag_service.rerank_documents")
@patch("app.services.rag_service.retrieve")
@patch("app.services.rag_service.get_llm")
def test_ask(mock_get_llm, mock_retrieve, mock_rerank):
    chunks = [
        MagicMock(page_content="Policy document info", metadata={"source": "test.pdf"})
    ]
    mock_retrieve.return_value = (chunks, 0.1, 0.2)
    mock_rerank.return_value = (chunks, 0.05)
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Mocked LLM answer"
    mock_get_llm.return_value = mock_llm

    result = ask("What is the policy?", llm_provider="groq")
    
    assert result["answer"] == "Mocked LLM answer"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "test.pdf"
    assert result["sources"][0]["preview"] == "Policy document info..."


@patch("app.services.rag_service.rerank_documents")
@patch("app.services.rag_service.retrieve")
@patch("app.services.rag_service.get_llm")
def test_ask_stream(mock_get_llm, mock_retrieve, mock_rerank):
    chunks = [
        MagicMock(page_content="Policy info", metadata={"source": "test.pdf"})
    ]
    mock_retrieve.return_value = (chunks, 0.1, 0.2)
    mock_rerank.return_value = (chunks, 0.05)
    mock_llm = MagicMock()
    mock_llm.stream.return_value = ["Ans", "wer"]
    mock_get_llm.return_value = mock_llm

    stream = list(ask_stream("query"))
    assert "wer" in stream[-1]

