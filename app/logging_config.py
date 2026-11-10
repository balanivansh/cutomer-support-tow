"""
app/logging_config.py
Structured, coloured logging for the tow-support-assistant pipeline.

Every log line format:
  HH:MM:SS  LEVEL  [CATEGORY]  stage=<stage>  conv=<id>  |  <message>

Categories: API_ERROR, EMBEDDING_ERROR, RETRIEVAL_ERROR, LLM_ERROR,
            DB_ERROR, CONFIG_ERROR, INFO

Outputs:
  • Terminal  — rich-coloured, human-readable single lines
  • File      — plain-text rotating file in LOG_DIR (10 MB × 5 backups)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path
from typing import Literal, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Category = Literal[
    "API_ERROR",
    "EMBEDDING_ERROR",
    "RETRIEVAL_ERROR",
    "LLM_ERROR",
    "DB_ERROR",
    "CONFIG_ERROR",
    "INFO",
]

# ---------------------------------------------------------------------------
# Secret masking
# ---------------------------------------------------------------------------

# Patterns that look like API keys / tokens (long alphanumeric strings)
_SECRET_PATTERN = re.compile(r"[A-Za-z0-9_\-]{20,}")
_MASK_KEEP_TAIL = 4


def _mask_secrets(text: str) -> str:
    """
    Replace anything that looks like an API key / token with ****<last4>.
    Only masks strings ≥ 20 chars that appear after known key= prefixes.
    """
    def _replace(match: re.Match) -> str:
        val = match.group(0)
        if len(val) >= 20:
            return "*" * (len(val) - _MASK_KEEP_TAIL) + val[-_MASK_KEEP_TAIL:]
        return val

    # Only mask values that follow known secret patterns in the log text
    masked = re.sub(
        r"(?i)(groq_api_key|hf_token|api_key|token|secret|password)\s*[=:]\s*([^\s,\"']+)",
        lambda m: f"{m.group(1)}={_replace_tail(m.group(2))}",
        text,
    )
    return masked


def _replace_tail(val: str) -> str:
    if len(val) <= _MASK_KEEP_TAIL:
        return "****"
    return "*" * (len(val) - _MASK_KEEP_TAIL) + val[-_MASK_KEEP_TAIL:]


# ---------------------------------------------------------------------------
# Custom formatter (plain text — used for the file handler)
# ---------------------------------------------------------------------------

class PipelineFormatter(logging.Formatter):
    """Single-line formatter that embeds category, stage, conv_id."""

    FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
    DATEFMT = "%H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)

    def format(self, record: logging.LogRecord) -> str:
        # Pull pipeline-specific extras injected by our helper functions
        category = getattr(record, "category", "INFO")
        stage = getattr(record, "stage", "")
        conv_id = getattr(record, "conv_id", "")

        parts = [f"[{category}]"]
        if stage:
            parts.append(f"stage={stage}")
        if conv_id:
            parts.append(f"conv={conv_id}")
        parts.append("|")
        parts.append(record.getMessage())

        record.msg = "  ".join(parts)
        record.args = ()

        formatted = super().format(record)
        return _mask_secrets(formatted)


# ---------------------------------------------------------------------------
# Rich console formatter (terminal)
# ---------------------------------------------------------------------------

_CATEGORY_STYLES: dict[str, str] = {
    "API_ERROR": "bold red",
    "EMBEDDING_ERROR": "bold magenta",
    "RETRIEVAL_ERROR": "bold yellow",
    "LLM_ERROR": "bold red",
    "DB_ERROR": "bold red",
    "CONFIG_ERROR": "bold white on red",
    "INFO": "bold cyan",
}

_rich_theme = Theme(
    {
        "logging.level.debug": "dim",
        "logging.level.info": "green",
        "logging.level.warning": "yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
    }
)

_console = Console(theme=_rich_theme, stderr=True)


class RichPipelineHandler(RichHandler):
    """RichHandler that injects category / stage / conv_id into the message."""

    def emit(self, record: logging.LogRecord) -> None:
        category = getattr(record, "category", "INFO")
        stage = getattr(record, "stage", "")
        conv_id = getattr(record, "conv_id", "")
        style = _CATEGORY_STYLES.get(category, "")

        parts: list[str] = []
        if style:
            parts.append(f"[{style}][{category}][/{style}]")
        else:
            parts.append(f"[{category}]")
        if stage:
            parts.append(f"stage={stage}")
        if conv_id:
            parts.append(f"conv={conv_id}")
        parts.append("|")
        parts.append(record.getMessage())

        record.msg = "  ".join(parts)
        record.args = ()
        # Mask before emitting
        record.msg = _mask_secrets(record.msg)
        super().emit(record)


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

_LOGGER_NAME = "tow_assistant"
_initialized = False


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "./logs",
) -> logging.Logger:
    """
    Configure the root pipeline logger.  Call once at application startup.
    Safe to call multiple times (idempotent after first call).
    """
    global _initialized
    if _initialized:
        return logging.getLogger(_LOGGER_NAME)

    level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # ── Terminal handler (rich) ──────────────────────────────────────────────
    rich_handler = RichPipelineHandler(
        console=_console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        log_time_format="[%H:%M:%S]",
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    # ── Rotating file handler (plain text) ───────────────────────────────────
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(PipelineFormatter())
    logger.addHandler(file_handler)

    _initialized = True
    return logger


def get_logger() -> logging.Logger:
    """Return the pipeline logger (auto-initialised with defaults if needed)."""
    if not _initialized:
        # Lazily init with defaults so imports work before explicit setup_logging()
        try:
            from app.config import settings  # noqa: PLC0415
            setup_logging(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)
        except SystemExit:
            # Config failed (empty keys); still return a usable logger
            setup_logging()
    return logging.getLogger(_LOGGER_NAME)


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------

def log_error(
    category: Category,
    *,
    stage: str = "",
    conv_id: str = "",
    msg: str,
    exc: Optional[BaseException] = None,
) -> None:
    """
    Log an error with pipeline metadata.

    Example:
        log_error("EMBEDDING_ERROR", stage="embed_query", conv_id="c123",
                  msg="HF returned 429. Retrying in 2s.", exc=e)
    """
    logger = get_logger()
    extra = {"category": category, "stage": stage, "conv_id": conv_id}
    if exc:
        logger.error(msg, extra=extra, exc_info=exc)
    else:
        logger.error(msg, extra=extra)


def log_info(
    *,
    stage: str = "",
    conv_id: str = "",
    msg: str,
) -> None:
    """
    Log an informational message with pipeline metadata.

    Example:
        log_info(stage="retrieval", conv_id="c123", msg="Fetched 5 chunks.")
    """
    logger = get_logger()
    extra = {"category": "INFO", "stage": stage, "conv_id": conv_id}
    logger.info(msg, extra=extra)


def log_warning(
    category: Category = "INFO",
    *,
    stage: str = "",
    conv_id: str = "",
    msg: str,
) -> None:
    """
    Log a warning with pipeline metadata.
    """
    logger = get_logger()
    extra = {"category": category, "stage": stage, "conv_id": conv_id}
    logger.warning(msg, extra=extra)


def log_debug(
    *,
    stage: str = "",
    conv_id: str = "",
    msg: str,
) -> None:
    """
    Log a debug message with pipeline metadata.
    """
    logger = get_logger()
    extra = {"category": "INFO", "stage": stage, "conv_id": conv_id}
    logger.debug(msg, extra=extra)
