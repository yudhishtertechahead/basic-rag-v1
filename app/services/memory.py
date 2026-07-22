"""
app/services/memory.py

Manages conversation history (state) across REST API requests using a sliding window.
Currently backed by an in-memory dictionary.
In production, this would be swapped with Redis or a database (e.g., SQLite).
"""

from app.core.constants import MEMORY_WINDOW_SIZE
from app.core.logger import get_logger

logger = get_logger(__name__)

# In-memory store: Maps session_id -> list of message dicts
# e.g., "session-uuid": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
_memory_store: dict[str, list[dict[str, str]]] = {}


def get_history(session_id: str | None) -> list[dict[str, str]]:
    """Retrieve the conversation history for a given session ID."""
    if not session_id:
        return []
    
    # Return a copy to prevent accidental mutation of the store
    return list(_memory_store.get(session_id, []))


def add_turn(session_id: str | None, user_message: str, assistant_message: str) -> None:
    """
    Append a new user/assistant pair to the conversation history.
    Enforces the sliding window limit (keeps the last MEMORY_WINDOW_SIZE pairs).
    """
    if not session_id:
        return

    if session_id not in _memory_store:
        _memory_store[session_id] = []

    history = _memory_store[session_id]
    
    # Append the new turn
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})
    
    # Enforce sliding window: Each turn is 2 messages.
    # So max messages = MEMORY_WINDOW_SIZE * 2
    max_messages = MEMORY_WINDOW_SIZE * 2
    if len(history) > max_messages:
        # Keep only the tail
        _memory_store[session_id] = history[-max_messages:]
        logger.debug("Sliding window triggered for session %s (kept last %d messages)", session_id, max_messages)

def clear_history(session_id: str) -> None:
    """Clear history for a session (e.g., when user clicks 'New Chat')."""
    if session_id in _memory_store:
        del _memory_store[session_id]
