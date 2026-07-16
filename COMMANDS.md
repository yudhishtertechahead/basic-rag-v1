# COMMANDS Reference Guide

> Every command used in this project, explained in plain English.
> Run all commands from the **project root** (`basic-rag-chatbot-v1/`) unless noted otherwise.

---

## 1. Project Setup

```bash
# Install all Python dependencies listed in pyproject.toml
# Run this once when you first clone the project, or whenever pyproject.toml changes
uv sync
```

```bash
# Add a new package to the project (automatically updates pyproject.toml and uv.lock)
# Format: uv add <package-name>
# Example: adding fastapi
uv add fastapi
```

```bash
# Run any Python file using the project's virtual environment (no need to activate venv manually)
# Format: uv run <filename.py>
uv run main.py
```

---

## 2. Running the FastAPI Server

```bash
# Start the FastAPI development server with hot-reload
# Hot-reload means the server restarts automatically when you save a file
# Access the API at: http://localhost:8000
# Access interactive docs (Swagger UI) at: http://localhost:8000/docs
uv run uvicorn main:app --reload
```

```bash
# Run on a custom port (if 8000 is already in use)
uv run uvicorn main:app --reload --port 8001
```

---

## 3. Document Ingestion

```bash
# Run the ingestion pipeline: loads docs/ -> chunks -> embeds -> stores in Qdrant Cloud
# Run this ONCE after adding/changing documents in the docs/ folder
# If you add new documents later, re-run this to update the vector store
uv run ingest.py
```

---

## 4. Testing the API Endpoints

```bash
# Check if the server is healthy and see which LLM is active
curl http://localhost:8000/health
```

```bash
# Trigger ingestion via the API (alternative to running ingest.py directly)
curl -X POST http://localhost:8000/ingest
```

```bash
# Ask a question to the chatbot (replace the question with your own)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the leave policy?"}'
```

```bash
# On Windows PowerShell, use this format instead of curl:
Invoke-RestMethod -Uri "http://localhost:8000/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"question": "What is the leave policy?"}'
```

---

## 5. Switching LLM Providers

The LLM is controlled by the `LLM_PROVIDER` variable in your `.env` file.

```bash
# To use Google Gemini (API-based):
# Open .env and set:
# LLM_PROVIDER=google
# GOOGLE_MODEL=gemini-3.5-flash
# Then restart the server:
uv run uvicorn main:app --reload
```

```bash
# To use Ollama (local open-source model):
# 1. Install Ollama from https://ollama.com/download
# 2. Pull a model (run this in any terminal):
ollama pull llama3

# 3. Start Ollama server (runs in background automatically after install)
ollama serve

# 4. Open .env and set:
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3

# 5. Restart the FastAPI server:
uv run uvicorn main:app --reload
```

---

## 6. Qdrant Cloud Setup (One-Time)

```
1. Go to https://cloud.qdrant.io and sign up for free
2. Create a new cluster (free tier)
3. Copy your:
   - Cluster URL (looks like: https://xxxx.cloud.qdrant.io)
   - API Key (from the Access section)
4. Paste them into your .env file:
   QDRANT_URL=https://xxxx.cloud.qdrant.io
   QDRANT_API_KEY=your_api_key_here
```

---

## 7. Viewing API Documentation

```
# After starting the server, open in browser:
# Swagger UI (interactive, lets you test endpoints in browser)
http://localhost:8000/docs

# ReDoc (cleaner, read-only documentation)
http://localhost:8000/redoc
```

---

## 8. Git Commands (Version Control)

```bash
# Check what files have changed
git status

# Add all changes to staging
git add .

# Commit with a message
git commit -m "Add RAG pipeline with Qdrant"

# Push to GitHub
git push
```

---

## 9. Troubleshooting

```bash
# If uv is not installed:
pip install uv

# Check Python version (need 3.12+)
python --version

# List all installed packages in the venv
uv pip list

# If Ollama is not responding, check it is running:
ollama list

# Re-pull a model:
ollama pull llama3
```

---

## Quick Start Checklist (First Time)

```
[ ] 1. uv sync                           <- install dependencies
[ ] 2. Add docs to docs/ folder          <- your PDF/Markdown files
[ ] 3. Set .env variables                <- API keys, Qdrant URL, LLM_PROVIDER
[ ] 4. uv run ingest.py                  <- load + embed + store docs in Qdrant
[ ] 5. uv run uvicorn main:app --reload  <- start the API server
[ ] 6. Open http://localhost:8000/docs   <- test the chatbot in browser
```
