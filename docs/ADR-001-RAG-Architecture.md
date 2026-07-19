# Architecture Decision Record
## ADR-001: Selection of Core RAG Components

**Date:** 2026-07-19
**Status:** Accepted

### 1. Context
We need to build a Retrieval-Augmented Generation (RAG) backend that allows employees to query internal HR documents. To do this, we require a vector database for semantic search, an embedding model for text vectorization, and a Large Language Model (LLM) for natural language generation. 

### 2. Decision
We have selected the following technology stack for the RAG architecture:
- **Vector Database:** **Qdrant Cloud** (Free Tier).
- **Embedding Model:** **all-MiniLM-L6-v2** (via `SentenceTransformers`).
- **LLM Interface:** **Factory Pattern** supporting Groq (Llama-3), Google Gemini, and Ollama.
- **Framework:** **FastAPI** for API serving.

### 3. Alternatives Considered
* **Vector Database:**
  * *Pinecone:* Ruled out due to free-tier inactivity deletion and strict limitations on index counts.
  * *ChromaDB / FAISS (Local):* Ruled out for production readiness; requires managing persistent state on disk which complicates containerization/deployment compared to a managed cloud solution.
* **Embedding Model:**
  * *OpenAI `text-embedding-3-small`:* Ruled out due to cost per token and API latency. `all-MiniLM-L6-v2` is free, fast, runs locally on CPU, and is highly effective for short HR policy chunks.
* **LLM Provider:**
  * *OpenAI / Anthropic only:* Ruled out to avoid vendor lock-in. We opted for a custom `LLM Factory` that allows seamless swapping between Google Gemini, ultra-fast Groq APIs, and fully local/private Ollama models depending on privacy requirements.

### 4. Consequences
* **Positive:**
  * Using Qdrant Cloud eliminates the need to self-host and manage a vector database.
  * The local embedding model (`all-MiniLM-L6-v2`) ensures that document contents are not sent to a third-party embedding API, improving privacy and reducing costs.
  * The Factory pattern for LLMs allows the company to test different models for cost/performance trade-offs without changing the RAG orchestration logic.
* **Negative/Risks:**
  * Running the embedding model locally means the application server requires a bit more memory/CPU on startup. We mitigated this by warming up the models during application startup to prevent latency on the first request.
