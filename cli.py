"""
cli.py

Unified CLI tool — merged from chat.py + ingest.py interactive flows.

Usage:
    uv run cli.py          # launch the chat interface (default)
    uv run cli.py ingest   # run document ingestion

Commands at the chat prompt:
    /ingest  → re-run ingestion without restarting
    /health  → check which model is active
    quit     → exit

The server must be running:
    uv run uvicorn main:app --reload
"""

import sys
import requests

# FastAPI server URL — must be running
BASE_URL = "http://localhost:8000"
CHAT_STREAM_URL = f"{BASE_URL}/chat/stream"
HEALTH_URL = f"{BASE_URL}/health"

# ── ANSI colors ────────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def strip_markdown(text: str) -> str:
    """Convert LLM markdown to clean plain text for terminal display."""
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'#{1,6}\s*',     '',    text)
    text = re.sub(r'`(.+?)`',       r'\1', text)
    text = re.sub(r'\n{3,}',        '\n\n', text)
    return text.strip()


def chat(question: str, llm_provider: str) -> None:
    """Runs the answer locally directly from rag_service for timing comparison."""
    from app.services.rag_service import ask
    try:
        print(f"{DIM}Thinking locally...{RESET}")
        result = ask(question, llm_provider)
        answer = strip_markdown(result["answer"])
        print(f"\n{BOLD}{CYAN}Aria:{RESET} {answer}\n")
    except Exception as e:
        print(f"\n{RED}[ERROR] {str(e)}{RESET}\n")


def show_health() -> None:
    """Display the active LLM configuration from /health."""
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        r.raise_for_status()
        data = r.json()
        print(f"\n{CYAN}Server status:{RESET}")
        print(f"  Provider : {data.get('llm_provider', 'unknown')}")
        print(f"  Model    : {data.get('model', 'unknown')}")
        print(f"  Qdrant   : {data.get('qdrant_collection', 'unknown')}\n")
    except Exception as e:
        print(f"{RED}[ERROR] Could not reach /health: {e}{RESET}\n")


def run_ingest_from_api() -> None:
    """Trigger POST /ingest on the running server."""
    try:
        print(f"\n{DIM}Starting ingestion via API...{RESET}")
        r = requests.post(f"{BASE_URL}/ingest", timeout=120)
        r.raise_for_status()
        data = r.json()
        print(f"{GREEN}[OK]{RESET} {data.get('message')} ({data.get('chunks_stored')} chunks)\n")
    except Exception as e:
        print(f"{RED}[ERROR] Ingestion failed: {e}{RESET}\n")


def print_banner() -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 54}{RESET}")
    print(f"{BOLD}{CYAN}   Aria — TechAhead HR Assistant{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 54}{RESET}")
    print(f"{DIM}  Type your question and press Enter.{RESET}")
    print(f"{DIM}  Commands: /ingest, /health, quit{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 54}{RESET}\n")


def chat_loop() -> None:
    print_banner()

    print(f"{YELLOW}Choose your LLM provider:{RESET}")
    print(f"  [1] Groq  llama-3.1-8b-instant  (fastest, recommended)")
    print(f"  [2] Google Gemini")
    print(f"  [3] Ollama (local)")
    choice = input(f"{YELLOW}Enter 1, 2, or 3 [1]: {RESET}").strip()

    if choice == "2":
        llm_provider = "google"
        print(f"\n{CYAN}Using Google Gemini...{RESET}\n")
    elif choice == "3":
        llm_provider = "ollama"
        print(f"\n{CYAN}Using Ollama (local)...{RESET}\n")
    else:
        llm_provider = "groq"
        print(f"\n{CYAN}Using Groq llama-3.1-8b-instant...{RESET}\n")

    print(f"{DIM}Warming up models (this may take a few seconds)...{RESET}")
    from app.db.vector_store import _get_embedder, get_qdrant_client
    from app.services.llm.factory import get_llm
    _get_embedder()
    get_qdrant_client()
    get_llm(llm_provider)
    print(f"{GREEN}[OK] Models loaded and ready!{RESET}\n")

    while True:
        try:
            question = input(f"{BOLD}{GREEN}You: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{DIM}Goodbye!{RESET}\n")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "bye", "q"):
            print(f"\n{DIM}Goodbye!{RESET}\n")
            break

        if question.lower() == "/health":
            show_health()
            continue

        if question.lower() == "/ingest":
            run_ingest_from_api()
            continue

        print(f"{DIM}Thinking...{RESET}")
        chat(question, llm_provider)
        print()


def run_ingest_direct() -> None:
    """Run ingestion directly (without needing the server running)."""
    from app.core.logger import log_banner, log_result, log_step, log_success
    from app.services.ingestion import ingest_folder

    log_banner("RAG Chatbot — Document Ingestion Pipeline")
    log_step(1, 1, "Loading, chunking, and storing documents from docs/...")
    total = ingest_folder()
    log_success(f"Stored {total} chunks in Qdrant")
    log_banner("Ingestion Complete!", char="-")
    log_result("Chunks stored", str(total))
    log_result("Next step", "uv run uvicorn main:app --reload")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        run_ingest_direct()
    else:
        chat_loop()
