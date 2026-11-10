"""Shared helpers for the KB stage: env loading, readable logging, local embedder.
Self-contained on purpose: reads .env directly, does not depend on app/config.py."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _env(name, default):
    return os.getenv(name) or default  # empty value in .env falls back to default


def _path(name, default):
    p = Path(_env(name, default))
    return p if p.is_absolute() else (ROOT / p).resolve()


KB_PATH = _path("KB_PATH", "./data/1_support_kb_tow.md")
CHROMA_PATH = _path("CHROMA_PATH", "./data/chroma")
LOG_DIR = _path("LOG_DIR", "./logs")
EMBEDDING_MODE = _env("EMBEDDING_MODE", "local")
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-m3")
KB_COLLECTION = "kb_chunks"

_logger = None


def get_logger():
    global _logger
    if _logger is not None:
        return _logger
    lg = logging.getLogger("tow_kb")
    lg.setLevel(_env("LOG_LEVEL", "INFO").upper())
    lg.propagate = False
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        lg.addHandler(sh)
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(LOG_DIR / "kb_stage.log", maxBytes=1_000_000,
                                     backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            lg.addHandler(fh)
        except OSError:
            pass
    _logger = lg
    return lg


def log_info(stage, msg):
    get_logger().info(f"[INFO] stage={stage} | {msg}")


def log_error(category, stage, msg, exc=None):
    if exc is not None:
        msg = f"{msg} ({type(exc).__name__}: {exc})"
    get_logger().error(f"[{category}] stage={stage} | {msg}")


_model = None


def get_embedder():
    global _model
    if _model is None:
        if EMBEDDING_MODE != "local":
            log_error("CONFIG_ERROR", "load_embedder",
                      f"EMBEDDING_MODE is '{EMBEDDING_MODE}' but this stage needs 'local'. Fix .env.")
            sys.exit(1)
        from sentence_transformers import SentenceTransformer
        log_info("load_embedder", f"Loading {EMBEDDING_MODEL} (first load can take a few seconds)")
        try:
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            log_error("EMBEDDING_ERROR", "load_embedder", f"Could not load {EMBEDDING_MODEL}", e)
            sys.exit(1)
        log_info("load_embedder", f"Loaded on device: {_model.device}")
    return _model


def embed(texts):
    try:
        return get_embedder().encode(texts, normalize_embeddings=True,
                                     batch_size=16, show_progress_bar=False).tolist()
    except Exception as e:
        log_error("EMBEDDING_ERROR", "embed", f"Embedding failed for {len(texts)} text(s)", e)
        raise
