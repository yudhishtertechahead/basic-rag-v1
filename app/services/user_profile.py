"""
app/services/user_profile.py

Per-session user entity/fact store.

Stores personal facts that the user has volunteered (name, department, role, etc.)
across the lifetime of a session. Unlike conversation history (which is a rolling
transcript), this is a structured key-value fact store that never ages out.

Backed by an in-memory dict — resets on server restart (intentional).
"""

from app.core.logger import get_logger

logger = get_logger(__name__)

# session_id → dict of known user facts
_profile_store: dict[str, dict] = {}

# Known fact keys and their display labels for the prompt
PROFILE_KEYS = {
    "name":             "Name",
    "department":       "Department",
    "role":             "Role/Designation",
    "experience_years": "Experience",
    "location":         "Location/City",
    "manager":          "Manager",
    "employee_id":      "Employee ID",
}


def get_profile(session_id: str | None) -> dict:
    """Returns the known fact dict for this session. Empty dict if nothing known."""
    if not session_id:
        return {}
    return dict(_profile_store.get(session_id, {}))


def update_profile(session_id: str | None, new_facts: dict) -> None:
    """
    Merges new_facts into the existing profile for this session.
    Existing keys are preserved; new_facts only adds or overwrites specific keys.
    """
    if not session_id or not new_facts:
        return
    if session_id not in _profile_store:
        _profile_store[session_id] = {}
    _profile_store[session_id].update(new_facts)
    logger.info("[UserProfile] Updated profile for session %s: %s", session_id, new_facts)


def clear_profile(session_id: str | None) -> None:
    """Clears the profile when user starts a new chat."""
    if session_id and session_id in _profile_store:
        del _profile_store[session_id]
        logger.debug("[UserProfile] Cleared profile for session %s", session_id)


def format_for_prompt(session_id: str | None) -> str:
    """
    Formats the user profile as a compact key: value string for the LLM.
    Returns empty string if no facts are known (so no block is injected).
    """
    profile = get_profile(session_id)
    if not profile:
        return ""

    lines = []
    for key, label in PROFILE_KEYS.items():
        if key in profile:
            lines.append(f"{label}: {profile[key]}")

    return " | ".join(lines) if lines else ""
