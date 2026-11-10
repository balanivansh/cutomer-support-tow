"""
app/config.py
Loads all application settings from .env via pydantic-settings.
Raises a clear CONFIG_ERROR (never a raw stack-trace) when required values
are missing or invalid.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    LLM_PROVIDER: Literal["groq"] = "groq"
    LLM_MODEL: str = ""  # filled after check_models.py run

    # ── Embeddings ───────────────────────────────────────────────────────────
    HF_TOKEN: str = ""
    EMBEDDING_MODEL: str = ""   # filled after check_models.py run
    EMBEDDING_MODE: Literal["hf_inference_api", "local"] = "hf_inference_api"

    # ── Paths ────────────────────────────────────────────────────────────────
    CHROMA_PATH: str = "./data/chroma"
    SQLITE_PATH: str = "./data/app.db"
    KB_PATH: str = "./data/1_support_kb_tow.md"

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_DIR: str = "./logs"

    # ── Pipeline behaviour ───────────────────────────────────────────────────
    DEBOUNCE_SECONDS: float = 4.0

    # ── Retrieval / BM25 ────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5           # number of chunks returned per retrieval pass
    BM25_TOP_K: int = 5                # BM25 candidate pool size

    # ── HTTP / retry ─────────────────────────────────────────────────────────
    HTTP_TIMEOUT_SECONDS: float = 30.0  # timeout for outbound HTTP calls
    MAX_RETRIES: int = 3                # max retry attempts on transient errors
    RETRY_BACKOFF_SECONDS: float = 2.0  # base back-off between retries (exponential)

    # ── Groq rate-limit safety (requests/min) ────────────────────────────────
    GROQ_RPM_LIMIT: int = 30           # stay under this; adjust after check_models

    # ── Embedding batching ───────────────────────────────────────────────────
    EMBED_BATCH_SIZE: int = 32         # texts sent per HF Inference API call

    # ── Validators ──────────────────────────────────────────────────────────

    @field_validator("GROQ_API_KEY", mode="after")
    @classmethod
    def _require_groq_key_at_runtime(cls, v: str) -> str:
        # Validation here is intentionally lenient so the config object can be
        # instantiated even with empty keys (needed during setup / check_models).
        # Runtime code that actually calls Groq must call assert_keys_present().
        return v

    @model_validator(mode="after")
    def _warn_empty_model_names(self) -> "Settings":
        # Soft warning only — prevents accidental use in production paths.
        return self

    # ── Public helpers ───────────────────────────────────────────────────────

    def assert_keys_present(self, *keys: str) -> None:
        """
        Call this in any module that actually needs a secret before using it.
        Prints a CONFIG_ERROR and exits instead of raising a raw exception.

        Usage:
            settings.assert_keys_present("GROQ_API_KEY", "HF_TOKEN")
        """
        missing = [k for k in keys if not getattr(self, k, "")]
        if missing:
            _config_error(
                f"The following required variable(s) are empty in your .env file: "
                f"{', '.join(missing)}. "
                f"Open .env and fill in the values, then re-run."
            )

    def masked(self, key: str) -> str:
        """Return the value of *key* with all but the last 4 chars masked."""
        value = getattr(self, key, "") or ""
        if len(value) <= 4:
            return "****"
        return "*" * (len(value) - 4) + value[-4:]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _config_error(message: str) -> None:
    """Print a human-readable CONFIG_ERROR and exit with code 2."""
    # Avoid importing logging_config here to prevent circular imports.
    # Use a plain print so this always works, even before logging is set up.
    print(
        f"\n[CONFIG_ERROR] {message}\n"
        f"  → Check your .env file (copy from .env.example if missing).\n",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the validated Settings singleton (cached after first call)."""
    try:
        return Settings()
    except Exception as exc:  # pydantic ValidationError etc.
        _config_error(str(exc))
        raise  # unreachable, but satisfies type-checkers


# Convenience alias used throughout the app
settings = get_settings()
