import pytest
from unittest.mock import patch, MagicMock
from app.services.llm.factory import get_llm, _provider_cache
import app.services.llm.groq
import app.services.llm.gemini
import app.services.llm.ollama

@patch("app.services.llm.groq.GroqProvider")
@patch("app.services.llm.factory.settings")
def test_factory_groq(mock_settings, mock_groq):
    _provider_cache.clear()
    mock_settings.groq_api_key = "test"
    llm = get_llm("groq")
    assert llm is not None

@patch("app.services.llm.gemini.GeminiProvider")
@patch("app.services.llm.factory.settings")
def test_factory_google(mock_settings, mock_google):
    _provider_cache.clear()
    mock_settings.google_api_key = "test"
    llm = get_llm("google")
    assert llm is not None

@patch("app.services.llm.ollama.OllamaProvider")
@patch("app.services.llm.factory.settings")
def test_factory_ollama(mock_settings, mock_ollama):
    _provider_cache.clear()
    llm = get_llm("ollama")
    assert llm is not None

def test_factory_invalid():
    _provider_cache.clear()
    with pytest.raises(ValueError):
        get_llm("invalid")


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

