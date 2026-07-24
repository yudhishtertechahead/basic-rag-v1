"""
app/services/entity_extractor.py

Extracts user personal facts from a message using a fast Groq LLM call.

Always uses Groq (fastest, cheapest) regardless of the user's selected model.
Only called when the heuristic gate detects a personal disclosure pattern.
"""

import json
import re

from app.core.logger import get_logger

logger = get_logger(__name__)

# Heuristic gate: only try extraction if the message matches these patterns
PERSONAL_DISCLOSURE_PATTERN = re.compile(
    r"\b(my name is|i am called|call me|i'm|i am|i work(?: in| at| as)?|"
    r"my role|my department|my designation|my team|my manager|"
    r"i have \d+|my experience|i've been|years? (of )?experience|"
    r"i'm based|i live in|i'm from|my employee id)\b",
    re.IGNORECASE,
)

EXTRACT_PROMPT = """You are a fact extractor. Extract ONLY explicitly stated personal facts from the user message.

<instructions>
1. Return a JSON object with any of these keys if found: name, department, role, experience_years, location, manager, employee_id.
2. Use the EXACT values the user stated. Do not infer or guess.
3. If no personal facts are present, return: {{}}
4. experience_years should be an integer if stated.
5. Do NOT include HR policy questions or general statements as facts.
</instructions>

<message>
{message}
</message>

Return ONLY valid JSON, nothing else:"""


def should_extract(message: str) -> bool:
    """Fast regex check — returns True if the message looks like a personal disclosure."""
    return bool(PERSONAL_DISCLOSURE_PATTERN.search(message))


def extract_user_facts(message: str) -> dict:
    """
    Calls Groq to extract structured user facts from a message.
    Always uses Groq for speed regardless of the user's selected LLM.
    Returns a dict of facts, or {} if nothing found or on error.
    """
    if not should_extract(message):
        return {}

    try:
        from app.services.llm.factory import get_llm
        groq_llm = get_llm("groq")

        prompt = EXTRACT_PROMPT.format(message=message)

        raw = groq_llm.generate(
            query=prompt,
            context="",
            history=None,
            prompt_id=None,
            _raw_mode=True,
        )

        # Strip markdown code fences if model wraps it
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`")

        facts = json.loads(raw)
        if isinstance(facts, dict) and facts:
            print(f"\033[32m[ENTITY] Extracted user facts: {facts}\033[0m")
            logger.info("[EntityExtractor] Extracted facts: %s", facts)
            return facts
        return {}

    except Exception as e:
        logger.debug("[EntityExtractor] Extraction failed (non-critical): %s", e)
        return {}
