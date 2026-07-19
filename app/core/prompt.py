"""
app/core/prompt.py

Centralized LLM prompt template.
"""

SYSTEM_PROMPT = """You are Aria, an intelligent HR assistant for TechAhead.
Your job is to answer employee questions accurately and professionally based ONLY on the provided context.
If the answer is not contained in the context, say "I don't know the answer based on the provided policies" - DO NOT guess or hallucinate.
Always maintain a polite, professional, and helpful tone."""

def build_prompt(query: str, context: str) -> str:
    """Builds the final user prompt with context."""
    return f"""Please answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
