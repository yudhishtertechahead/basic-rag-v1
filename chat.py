"""
chat.py

Terminal-based interactive chat with the RAG chatbot.
Run this while the FastAPI server is also running.

Usage:
    uv run chat.py

Type your question and press Enter. Type 'quit' or 'exit' to stop.

Alternative: Use chat.html for a browser-based UI instead.
"""

import sys
import requests

# FastAPI server URL — must be running: uv run uvicorn main:app --reload
API_URL = "http://localhost:8000/chat"

GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def chat(question: str) -> dict:
    """Sends the question to the FastAPI /chat endpoint and returns the result."""
    try:
        response = requests.post(API_URL, json={"question": question}, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"\n{RED}[ERROR] Cannot connect to the API server.{RESET}")
        print(f"{DIM}  Make sure the server is running:{RESET}")
        print(f"{YELLOW}  uv run uvicorn main:app --reload{RESET}\n")
        sys.exit(1)
    except requests.exceptions.Timeout:
        return {"answer": "Request timed out. The LLM might be slow — try again.", "sources": []}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}


def print_banner():
    print(f"\n{BOLD}{CYAN}{'=' * 52}{RESET}")
    print(f"{BOLD}{CYAN}   RAG Chatbot -- Ask your documents anything!{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 52}{RESET}")
    print(f"{DIM}  Type your question and press Enter.{RESET}")
    print(f"{DIM}  Type 'quit' or 'exit' to stop.{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 52}{RESET}\n")


def main():
    print_banner()

    while True:
        # ── Get user input ─────────────────────────────────────────────────────
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

        # ── Call the API ───────────────────────────────────────────────────────
        print(f"{DIM}Thinking...{RESET}")
        result = chat(question)

        # ── Print the answer ───────────────────────────────────────────────────
        print(f"\n{BOLD}{CYAN}Bot:{RESET} {result['answer']}")

        # ── Print sources ──────────────────────────────────────────────────────
        sources = result.get("sources", [])
        if sources:
            print(f"\n{DIM}Sources:{RESET}")
            for i, src in enumerate(sources, 1):
                source_name = src.get("source", "Unknown")
                # Show just the filename, not the full path
                source_name = source_name.split("\\")[-1].split("/")[-1]
                page = src.get("page")
                page_str = f" (page {page + 1})" if page is not None else ""
                print(f"  {DIM}[{i}] {source_name}{page_str}{RESET}")

        print()  # blank line between turns


if __name__ == "__main__":
    main()
