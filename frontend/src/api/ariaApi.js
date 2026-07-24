/**
 * ariaApi.js
 * All fetch wrappers for the Aria FastAPI backend.
 * Base URL auto-detects: uses /api/v1 (same-origin) in production,
 * or proxied via Vite during development.
 */

const BASE = '/api/v1';

// ── Health ─────────────────────────────────────────────────────────────────
export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error('Backend unreachable');
  return res.json(); // { status, llm_provider, model, qdrant_collection }
}

// ── Prompt Templates ───────────────────────────────────────────────────────
export async function fetchPrompts() {
  const res = await fetch(`${BASE}/prompts`);
  if (!res.ok) throw new Error('Could not fetch prompts');
  return res.json(); // { prompts: [{id, name, description}] }
}

// ── Retrieved Context (no LLM) ─────────────────────────────────────────────
export async function fetchContext({ question, sessionId }) {
  const res = await fetch(`${BASE}/context`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
  if (!res.ok) return { chunks: [] };
  return res.json(); // { chunks: [{index, source, page, content, preview}] }
}

// ── Streaming Chat ─────────────────────────────────────────────────────────
/**
 * Starts a streaming chat request. Returns a ReadableStream reader.
 * The caller iterates the reader with read() and appends tokens.
 *
 * @param {object} params
 * @param {string} params.question
 * @param {string|null} params.sessionId
 * @param {string|null} params.llmProvider  - 'groq' | 'google' | 'ollama'
 * @param {string|null} params.promptId     - 'default' | 'concise' | 'detailed' | 'strict'
 * @returns {Promise<{reader: ReadableStreamDefaultReader, decoder: TextDecoder}>}
 */
export async function streamChat({ question, sessionId, llmProvider, promptId }) {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      session_id: sessionId,
      llm_provider: llmProvider || null,
      prompt_id: promptId || 'default',
    }),
  });

  if (!res.ok) {
    let detail = `Server error ${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch (_) { }
    throw new Error(detail);
  }

  return {
    reader: res.body.getReader(),
    decoder: new TextDecoder(),
  };
}

// ── Ingest (admin) ────────────────────────────────────────────────────────
export async function triggerIngest() {
  const res = await fetch(`${BASE}/ingest`, { method: 'POST' });
  if (!res.ok) throw new Error('Ingestion failed');
  return res.json();
}
