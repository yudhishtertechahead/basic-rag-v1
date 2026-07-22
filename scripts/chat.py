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
API_URL = "http://localhost:8000/chat/stream"

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
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **bold** → bold
    text = re.sub(r'\*(.+?)\*',     r'\1', text)   # *italic* → italic
    text = re.sub(r'#{1,6}\s*',     '',    text)   # ### Heading → Heading
    text = re.sub(r'`(.+?)`',       r'\1', text)   # `code` → code
    text = re.sub(r'\n{3,}',        '\n\n', text)  # collapse triple blank lines
    return text.strip()


def chat(question: str, llm_provider: str) -> None:
    """Streams the answer from /chat/stream and prints tokens as they arrive."""
    try:
        with requests.post(
            API_URL,
            json={"question": question, "llm_provider": llm_provider},
            timeout=60,
            stream=True,          # ← keep connection open while tokens arrive
        ) as response:
            response.raise_for_status()
            print(f"\n{BOLD}{CYAN}Bot:{RESET} ", end="", flush=True)
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    print(chunk, end="", flush=True)  # print each token as it arrives
            print()  # newline after answer ends
            return
    except requests.exceptions.ConnectionError:
        print(f"\n{RED}[ERROR] Cannot connect to the API server.{RESET}")
        print(f"{DIM}  Make sure the server is running:{RESET}")
        print(f"{YELLOW}  uv run uvicorn main:app --reload{RESET}\n")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"\n{RED}[ERROR] Request timed out. The LLM might be slow — try again.{RESET}\n")
    except Exception as e:
        print(f"\n{RED}[ERROR] {str(e)}{RESET}\n")


def print_banner():
    print(f"\n{BOLD}{CYAN}{'=' * 52}{RESET}")
    print(f"{BOLD}{CYAN}   RAG Chatbot -- Ask your documents anything!{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 52}{RESET}")
    print(f"{DIM}  Type your question and press Enter.{RESET}")
    print(f"{DIM}  Type 'quit' or 'exit' to stop.{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 52}{RESET}\n")


def main():
    print_banner()

    # ── Ask user for LLM choice ────────────────────────────────────────────
    print(f"{YELLOW}Choose your LLM provider:{RESET}")
    print(f"  [1] Google Gemini (default)")
    print(f"  [2] Groq")
    choice = input(f"{YELLOW}Enter 1 or 2 [1]: {RESET}").strip()
    
    if choice == "2":
        llm_provider = "groq"
        print(f"\n{CYAN}Using Groq...{RESET}\n")
    else:
        llm_provider = "google"
        print(f"\n{CYAN}Using Google Gemini...{RESET}\n")

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

        # ── Call the API (streaming) ────────────────────────────────────────────
        print(f"{DIM}Thinking...{RESET}")
        chat(question, llm_provider)
        print()  # blank line between turns


if __name__ == "__main__":
    main()
