"""
app/core/logger.py

Centralized logging setup for the entire application.

Uses Python's built-in `logging` module with ANSI color codes for
color-coded, timestamped output — no extra dependencies needed.

Color legend in terminal:
  [DEBUG]   → Cyan    — detailed internal state, only shown in DEBUG mode
  [INFO]    → Green   — normal progress messages
  [WARNING] → Yellow  — something unexpected but not fatal
  [ERROR]   → Red     — something broke, needs attention

Usage in any file:
    from app.core.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Starting ingestion...")
    logger.debug("Chunk content: %s", chunk.page_content[:100])
    logger.warning("docs/ folder is empty")
    logger.error("Failed to connect to Qdrant: %s", str(e))
"""

import logging
import sys


# ── ANSI color codes ──────────────────────────────────────────────────────────
# These are terminal escape sequences that colorize text.
# They work on all modern terminals (Windows 10+, macOS, Linux).
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
DIM    = "\033[2m"
BLUE   = "\033[34m"
MAGENTA = "\033[35m"


class ColorFormatter(logging.Formatter):
    """
    Custom log formatter that adds:
      - Colors per log level
      - Timestamp in [HH:MM:SS] format
      - Module name so you know which file logged the message
    
    Output format:
      [HH:MM:SS] [LEVEL   ] module_name  → message
    """

    # Map each log level to its color + label
    LEVEL_STYLES = {
        logging.DEBUG:    (CYAN,    "DEBUG  "),
        logging.INFO:     (GREEN,   "INFO   "),
        logging.WARNING:  (YELLOW,  "WARNING"),
        logging.ERROR:    (RED,     "ERROR  "),
        logging.CRITICAL: (RED + BOLD, "CRITICAL"),
    }

    def format(self, record: logging.LogRecord) -> str:
        color, label = self.LEVEL_STYLES.get(record.levelno, (RESET, "LOG    "))

        # Timestamp: e.g. 16:44:21
        timestamp = self.formatTime(record, datefmt="%H:%M:%S")

        # Shorten module name for readability: app.ingestion.loader → loader
        module = record.name.split(".")[-1] if "." in record.name else record.name
        module_padded = module[:18].ljust(18)  # Pad to align columns

        # The actual log message
        message = record.getMessage()

        # Handle exceptions (if logger.exception() is used)
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return (
            f"{DIM}[{timestamp}]{RESET} "
            f"{color}{BOLD}[{label}]{RESET} "
            f"{DIM}{module_padded}{RESET}  {message}"
        )


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with colored console output.

    Args:
        name: Use __name__ so the logger knows which module it belongs to.
              e.g. get_logger(__name__) in loader.py → name = "app.ingestion.loader"

    Returns:
        A configured Logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Loaded 3 documents")
        logger.warning("No PDF files found")
        logger.error("Qdrant connection failed")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture all levels; handler filters below

    # Console handler — prints to stdout (visible in terminal)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(ColorFormatter())

    logger.addHandler(handler)

    # Prevent log messages from bubbling up to the root logger
    # (avoids duplicate output if root logger is also configured)
    logger.propagate = False

    return logger


# ── Convenience banners for pipeline section headers ─────────────────────────

def log_banner(title: str, char: str = "=", width: int = 52) -> None:
    """
    Prints a bold banner line for major pipeline sections.

    Usage:
        log_banner("Document Ingestion Pipeline")

    Output:
        ====================================================
          Document Ingestion Pipeline
        ====================================================
    """
    border = char * width
    print(f"\n{BOLD}{BLUE}{border}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{border}{RESET}")


def log_step(step: int, total: int, description: str) -> None:
    """
    Prints a formatted step indicator for multi-step pipelines.

    Usage:
        log_step(1, 3, "Loading documents from docs/")

    Output:
        >> [1/3] Loading documents from docs/
    """
    print(f"\n{BOLD}{MAGENTA}>> [{step}/{total}]{RESET} {description}")


def log_success(message: str) -> None:
    """Prints a green checkmark success message."""
    print(f"{GREEN}{BOLD}  [OK] {message}{RESET}")


def log_result(label: str, value: str) -> None:
    """Prints a dimmed key: value result line."""
    print(f"    {DIM}{label}:{RESET} {BOLD}{value}{RESET}")
