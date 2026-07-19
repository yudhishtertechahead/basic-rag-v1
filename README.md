# basic-rag-chatbot-v1

A fast RAG (Retrieval-Augmented Generation) chatbot for HR Q&A, powered by Groq's `llama-3.1-8b-instant` and Qdrant vector search.

## Performance Profile

| Step | Time |
|------|------|
| Embedding + retrieval | ~0.2–0.5s |
| First token (Groq 8B instant) | ~0.1–0.3s |
| Full answer | ~1–2s total |

## Quick Start

### 1. Install dependencies
```bash
uv sync
```

### 2. Configure environment
```bash
cp .env.example .env
# Fill in GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

### 3. Add documents
Place your PDF, Markdown, or TXT files in the `docs/` folder.

### 4. Ingest documents
```bash
uv run ingest.py
# or: uv run cli.py ingest
```

### 5. Start the server
```bash
uv run uvicorn main:app --reload
```

### 6. Chat
- **Browser UI:** http://localhost:8000/ui
- **API docs:** http://localhost:8000/docs
- **Terminal CLI:** `uv run cli.py`

## Project Structure

```
basic-rag-chatbot-v1/
├── main.py                    # FastAPI entry point (minimal)
├── ingest.py                  # CLI wrapper for ingestion
├── cli.py                     # Interactive chat + ingest CLI
├── .env.example
├── pyproject.toml
├── README.md
├── static/
│   └── chat.html              # Browser chat UI
├── docs/                      # Place your documents here
├── examples/
│   └── agent_demo.py
└── app/
    ├── api/
    │   └── router.py          # FastAPI routes
    ├── core/
    │   ├── config.py          # Settings (pydantic-settings)
    │   ├── prompt.py          # SYSTEM_PROMPT + build_prompt()
    │   ├── logger.py          # Colored logging
    │   └── logging.py         # Canonical alias for logger.py
    ├── db/
    │   └── vector_store.py    # Qdrant client, embed, retrieve, upsert
    └── services/
        ├── rag_service.py     # ask() + ask_stream() pipeline
        ├── ingestion.py       # ingest_file() + ingest_folder()
        └── llm/
            ├── base.py        # BaseLLM abstract class
            ├── factory.py     # get_llm() provider cache
            ├── groq.py        # Direct Groq SDK (8b-instant, fastest)
            ├── gemini.py      # Google Gemini (LangChain)
            └── ollama.py      # Ollama local (LangChain)
```

## LLM Providers

| Provider | Speed | Setup |
|----------|-------|-------|
| **Groq** (default) | ~1–2s total | `GROQ_API_KEY` from console.groq.com |
| Google Gemini | ~3–5s | `GOOGLE_API_KEY` from aistudio.google.com |
| Ollama (local) | varies | Install ollama, `ollama pull llama3` |

Switch provider: set `LLM_PROVIDER=groq` (or `google`/`ollama`) in `.env`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status + active model |
| `POST` | `/ingest` | Ingest docs/ folder into Qdrant |
| `POST` | `/chat` | Full (non-streaming) answer |
| `POST` | `/chat/stream` | Streaming answer (token-by-token) |
| `GET` | `/ui` | Browser chat interface |
| `GET` | `/docs` | Swagger UI |
