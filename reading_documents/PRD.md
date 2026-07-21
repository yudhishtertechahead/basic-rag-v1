# Product Requirements Document (PRD)
**Product Name:** Aria (TechAhead HR Assistant)
**Date:** 2026-07-19

## 1. Product Overview
**What are we building?**
A Retrieval-Augmented Generation (RAG) chatbot designed to act as an intelligent HR assistant. It ingests company policy documents, stores them in a vector database, and uses an LLM to answer employee questions accurately by retrieving contextually relevant information from those policies.

**For Whom?**
- **TechAhead Employees:** To quickly find answers to HR-related questions (e.g., leave policies, dress codes, referral bonuses) without waiting for human HR staff.
- **TechAhead HR Team:** To reduce the repetitive burden of answering standard policy questions, allowing them to focus on complex employee relations and strategic tasks.

## 2. Success Criteria
What does success look like for this MVP?
1. **Accuracy (Zero Hallucinations):** The assistant must only answer based on the provided company documents. If an answer does not exist in the documents, it must explicitly state that it cannot find the answer rather than guessing.
2. **Speed:** The end-to-end response time (including embedding the query, retrieving chunks, and generating the LLM response) should average under 2.5 seconds.
3. **Usability:** The system must offer both a REST API for potential frontend integration and an interactive CLI for testing and terminal users.
4. **Reliability:** The system must achieve a minimum of 85% test coverage across core RAG, vector store, and API services.

## 3. Core Features
- **Document Ingestion:** Automated loading, chunking, and embedding of PDF/Markdown files from the `docs/` folder into Qdrant Cloud.
- **Semantic Retrieval:** Ability to match user questions to specific document chunks using `all-MiniLM-L6-v2` embeddings via cosine similarity.
- **LLM Agnosticism:** Built-in support to effortlessly toggle between cloud-based LLMs (Google Gemini, Groq) and local LLMs (Ollama) through environment configurations.
- **Traceability:** Full structured debug logging allowing developers to trace individual requests from API entry to database retrieval to final LLM generation.
