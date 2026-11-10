"""
scripts/check_models.py
=======================
Live model availability report for tow-support-assistant.

Checks:
  A) Groq  — lists all visible models, runs a quick ping + multilingual test
  B) HF Inference API — tests candidate embedding models for feature extraction
     and runs a retrieval sanity (cosine similarity) test
  C) Prints a summary table and a recommended shortlist

USAGE:
  python scripts/check_models.py

Add extra HF candidate models to HF_CANDIDATE_MODELS below.
"""

from __future__ import annotations

import sys
import os
import time
import math
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `app.*` imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging must be set up before anything else uses it
# ---------------------------------------------------------------------------
from app.logging_config import setup_logging, log_error, log_info, log_warning
from app.config import settings

setup_logging(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)

# ---------------------------------------------------------------------------
# Config guard: fail fast with a clear CONFIG_ERROR if keys are missing
# ---------------------------------------------------------------------------
_missing: list[str] = []
if not settings.GROQ_API_KEY:
    _missing.append("GROQ_API_KEY")
if not settings.HF_TOKEN:
    _missing.append("HF_TOKEN")

if _missing:
    log_error(
        "CONFIG_ERROR",
        stage="startup",
        msg=(
            f"Cannot run check_models.py — the following variable(s) are empty "
            f"in your .env: {', '.join(_missing)}. "
            f"Open .env, fill in the values, then re-run this script."
        ),
    )
    sys.exit(2)

# ---------------------------------------------------------------------------
# Third-party imports (all installed in .venv)
# ---------------------------------------------------------------------------
import httpx
import numpy as np
from groq import Groq
from huggingface_hub import InferenceClient
from rich.console import Console
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ── Add extra HF models here if you want to test more candidates ──────────
HF_CANDIDATE_MODELS: list[str] = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/LaBSE",
]

# Groq model categories to skip (not chat-completion models)
GROQ_SKIP_PATTERNS = ("whisper", "guard", "tts", "distil-whisper", "playai")

# Test prompts
GROQ_PING_PROMPT = "Reply with the single word: OK"
GROQ_MULTILINGUAL_PROMPT = (
    "Restate the following message as one clear English sentence:\n"
    "\"का TUG कम हो गया है। कृपया इसकी जाँच कर हमें बताने\""
)

# HF cosine-similarity test texts
HF_TEXTS = {
    "english": "How many free credits do I get?",
    "hinglish": "mujhe kitne free credit milte hain",
    "hindi": "मुझे कितने फ्री क्रेडिट मिलते हैं",
    "unrelated": "How do I reset my router?",
}

console = Console()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _groq_skip(model_id: str) -> bool:
    lower = model_id.lower()
    return any(pat in lower for pat in GROQ_SKIP_PATTERNS)


def _extract_rate_limit_headers(headers: httpx.Headers | dict) -> str:
    """Pull Groq rate-limit headers and format as a short string."""
    keys = [
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "retry-after",
    ]
    parts = []
    for k in keys:
        v = headers.get(k)
        if v:
            parts.append(f"{k.replace('x-ratelimit-', '')}={v}")
    return ", ".join(parts) if parts else "—"


# ---------------------------------------------------------------------------
# Section A: Groq
# ---------------------------------------------------------------------------

def check_groq() -> list[dict]:
    """
    Returns list of result dicts with keys:
      model, works, latency_ms, multilingual_answer, rate_limit_info, error
    """
    console.rule("[bold cyan]A) Groq Model Check[/bold cyan]")
    client = Groq(api_key=settings.GROQ_API_KEY)

    # 1. List all models
    console.print("[dim]Fetching model list from Groq...[/dim]")
    try:
        models_response = client.models.list()
        all_models = [m.id for m in models_response.data]
    except Exception as exc:
        log_error("API_ERROR", stage="groq_list_models", msg=f"Failed to list Groq models: {exc}", exc=exc)
        console.print(f"[bold red]ERROR:[/bold red] Could not list Groq models — {exc}")
        return []

    chat_models = [m for m in all_models if not _groq_skip(m)]
    console.print(f"Found [bold]{len(all_models)}[/bold] total models, "
                  f"[bold]{len(chat_models)}[/bold] chat models to test.\n")

    results: list[dict] = []

    for model_id in chat_models:
        console.print(f"  Testing [cyan]{model_id}[/cyan] ...")
        result: dict = {
            "model": model_id,
            "works": False,
            "latency_ms": None,
            "multilingual_answer": "",
            "rate_limit_info": "—",
            "error": "",
        }

        # ── Ping test ────────────────────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            # Use httpx directly so we can inspect response headers
            ping_response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": GROQ_PING_PROMPT}],
                    "max_tokens": 10,
                    "temperature": 0,
                },
                timeout=settings.HTTP_TIMEOUT_SECONDS,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            result["latency_ms"] = latency_ms
            result["rate_limit_info"] = _extract_rate_limit_headers(ping_response.headers)

            if ping_response.status_code == 200:
                result["works"] = True
                console.print(f"    ✅ Ping OK  ({latency_ms} ms)")
            else:
                err = ping_response.json().get("error", {}).get("message", ping_response.text[:200])
                result["error"] = f"HTTP {ping_response.status_code}: {err}"
                log_error("API_ERROR", stage="groq_ping", msg=f"{model_id}: {result['error']}")
                console.print(f"    ❌ Ping FAIL — {result['error']}")
                results.append(result)
                continue

        except Exception as exc:
            result["error"] = str(exc)
            log_error("API_ERROR", stage="groq_ping", msg=f"{model_id}: {exc}", exc=exc)
            console.print(f"    ❌ Ping ERROR — {exc}")
            results.append(result)
            continue

        # ── Multilingual test ─────────────────────────────────────────────────
        try:
            ml_response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": GROQ_MULTILINGUAL_PROMPT}],
                    "max_tokens": 100,
                    "temperature": 0,
                },
                timeout=settings.HTTP_TIMEOUT_SECONDS,
            )
            if ml_response.status_code == 200:
                answer = ml_response.json()["choices"][0]["message"]["content"].strip()
                result["multilingual_answer"] = answer
                console.print(f"    🌐 Hindi→English: [italic]{answer}[/italic]")
            else:
                err = ml_response.json().get("error", {}).get("message", ml_response.text[:200])
                result["multilingual_answer"] = f"ERROR: {err}"
                log_warning("LLM_ERROR", stage="groq_multilingual", msg=f"{model_id}: {err}")
        except Exception as exc:
            result["multilingual_answer"] = f"ERROR: {exc}"
            log_error("LLM_ERROR", stage="groq_multilingual", msg=f"{model_id}: {exc}", exc=exc)

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Section B: Hugging Face Inference API — Embeddings
# ---------------------------------------------------------------------------

def check_hf_embeddings() -> list[dict]:
    """
    Returns list of result dicts with keys:
      model, works, vector_size, latency_ms, similarity_hinglish,
      similarity_hindi, similarity_unrelated, error
    """
    console.rule("[bold magenta]B) Hugging Face Embedding Check[/bold magenta]")
    hf_client = InferenceClient(token=settings.HF_TOKEN)
    results: list[dict] = []

    for model_id in HF_CANDIDATE_MODELS:
        console.print(f"\n  Testing [magenta]{model_id}[/magenta] ...")
        result: dict = {
            "model": model_id,
            "works": False,
            "vector_size": None,
            "latency_ms": None,
            "similarity_hinglish": None,
            "similarity_hindi": None,
            "similarity_unrelated": None,
            "error": "",
        }

        # ── Feature extraction test ─────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            embedding = hf_client.feature_extraction(
                text=HF_TEXTS["english"],
                model=model_id,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)

            # Normalise: HF may return nested list (batch) or flat list
            vec = embedding
            if isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list):
                vec = vec[0]
            # Some models return [batch, seq, dim]; take mean over seq dim
            if isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list):
                vec = [sum(col) / len(col) for col in zip(*vec)]

            result["works"] = True
            result["vector_size"] = len(vec)
            result["latency_ms"] = latency_ms
            console.print(f"    ✅ OK  dim={len(vec)}  ({latency_ms} ms)")

        except Exception as exc:
            result["error"] = str(exc)
            log_error("EMBEDDING_ERROR", stage="hf_feature_extraction",
                      msg=f"{model_id}: {exc}", exc=exc)
            console.print(f"    ❌ FAIL — {exc}")
            results.append(result)
            continue

        # ── Cosine similarity sanity test ────────────────────────────────────
        console.print("    🔍 Running retrieval sanity test ...")
        try:
            embeddings: dict[str, list[float]] = {}
            for label, text in HF_TEXTS.items():
                raw = hf_client.feature_extraction(text=text, model=model_id)
                vec_i = raw
                if isinstance(vec_i, list) and len(vec_i) > 0 and isinstance(vec_i[0], list):
                    vec_i = vec_i[0]
                if isinstance(vec_i, list) and len(vec_i) > 0 and isinstance(vec_i[0], list):
                    vec_i = [sum(col) / len(col) for col in zip(*vec_i)]
                embeddings[label] = vec_i

            sim_hing = cosine_similarity(embeddings["english"], embeddings["hinglish"])
            sim_hind = cosine_similarity(embeddings["english"], embeddings["hindi"])
            sim_unr = cosine_similarity(embeddings["english"], embeddings["unrelated"])

            result["similarity_hinglish"] = round(sim_hing, 4)
            result["similarity_hindi"] = round(sim_hind, 4)
            result["similarity_unrelated"] = round(sim_unr, 4)

            console.print(
                f"    Cosine sim — Hinglish: [green]{sim_hing:.4f}[/green]  "
                f"Hindi: [green]{sim_hind:.4f}[/green]  "
                f"Unrelated: [yellow]{sim_unr:.4f}[/yellow]"
            )
            if sim_hing > sim_unr and sim_hind > sim_unr:
                console.print("    ✅ Sanity check PASSED (multilingual > unrelated)")
            else:
                log_warning(
                    "RETRIEVAL_ERROR",
                    stage="hf_sanity",
                    msg=f"{model_id}: sanity check FAILED — multilingual similarity not clearly above unrelated.",
                )
                console.print(
                    "    ⚠️  Sanity check FAILED — multilingual similarity is NOT clearly above unrelated"
                )

        except Exception as exc:
            result["error"] = (result["error"] + f" | Sanity test error: {exc}").lstrip(" | ")
            log_error("EMBEDDING_ERROR", stage="hf_sanity", msg=f"{model_id}: {exc}", exc=exc)
            console.print(f"    ❌ Sanity test error — {exc}")

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Section C: Summary table + shortlist
# ---------------------------------------------------------------------------

def print_summary(groq_results: list[dict], hf_results: list[dict]) -> None:
    console.rule("[bold green]C) Summary Report[/bold green]")

    # ── Groq table ────────────────────────────────────────────────────────────
    console.print("\n[bold]Groq LLM Results[/bold]")
    tbl = Table(box=box.ROUNDED, show_lines=True)
    tbl.add_column("Model", style="cyan", no_wrap=True)
    tbl.add_column("Works", justify="center")
    tbl.add_column("Latency (ms)", justify="right")
    tbl.add_column("Rate Limit Info", style="dim")
    tbl.add_column("Hindi→English Translation", max_width=60)
    tbl.add_column("Error", style="red", max_width=40)

    for r in groq_results:
        tbl.add_row(
            r["model"],
            "✅" if r["works"] else "❌",
            str(r["latency_ms"]) if r["latency_ms"] else "—",
            r["rate_limit_info"],
            r["multilingual_answer"] or "—",
            r["error"] or "—",
        )
    console.print(tbl)

    # ── HF table ─────────────────────────────────────────────────────────────
    console.print("\n[bold]Hugging Face Embedding Results[/bold]")
    tbl2 = Table(box=box.ROUNDED, show_lines=True)
    tbl2.add_column("Model", style="magenta", no_wrap=True)
    tbl2.add_column("Works", justify="center")
    tbl2.add_column("Dim", justify="right")
    tbl2.add_column("Latency (ms)", justify="right")
    tbl2.add_column("Sim Hinglish", justify="right")
    tbl2.add_column("Sim Hindi", justify="right")
    tbl2.add_column("Sim Unrelated", justify="right")
    tbl2.add_column("Error", style="red", max_width=40)

    for r in hf_results:
        tbl2.add_row(
            r["model"],
            "✅" if r["works"] else "❌",
            str(r["vector_size"]) if r["vector_size"] else "—",
            str(r["latency_ms"]) if r["latency_ms"] else "—",
            str(r["similarity_hinglish"]) if r["similarity_hinglish"] is not None else "—",
            str(r["similarity_hindi"]) if r["similarity_hindi"] is not None else "—",
            str(r["similarity_unrelated"]) if r["similarity_unrelated"] is not None else "—",
            r["error"] or "—",
        )
    console.print(tbl2)

    # ── Shortlist recommendation ─────────────────────────────────────────────
    console.rule("[bold yellow]Suggested Shortlist[/bold yellow]")
    console.print("\n[bold]Top 2 LLM candidates[/bold] (from your Groq account):")

    working_groq = [r for r in groq_results if r["works"]]
    working_groq_sorted = sorted(working_groq, key=lambda r: r["latency_ms"] or 99999)

    if len(working_groq_sorted) >= 1:
        m = working_groq_sorted[0]
        console.print(
            f"  1. [cyan]{m['model']}[/cyan] — "
            f"Fastest working model ({m['latency_ms']} ms). "
            f"Good balance of speed and capability for draft-reply generation."
        )
    if len(working_groq_sorted) >= 2:
        m = working_groq_sorted[1]
        console.print(
            f"  2. [cyan]{m['model']}[/cyan] — "
            f"Second fastest ({m['latency_ms']} ms). "
            f"Useful as a fallback or if translation quality was noticeably better."
        )
    if not working_groq_sorted:
        console.print("  [red]No working Groq models found — check your GROQ_API_KEY.[/red]")

    console.print("\n[bold]Top 2 Embedding model candidates[/bold] (from HF Inference API):")

    def embed_score(r: dict) -> float:
        """Higher = better: mean of multilingual sims minus unrelated sim."""
        if not r["works"]:
            return -999.0
        sh = r["similarity_hinglish"] or 0.0
        hd = r["similarity_hindi"] or 0.0
        un = r["similarity_unrelated"] or 0.0
        return (sh + hd) / 2 - un

    working_hf = [r for r in hf_results if r["works"]]
    working_hf_sorted = sorted(working_hf, key=embed_score, reverse=True)

    if len(working_hf_sorted) >= 1:
        m = working_hf_sorted[0]
        sc = embed_score(m)
        console.print(
            f"  1. [magenta]{m['model']}[/magenta] — "
            f"Best multilingual-vs-unrelated separation score ({sc:+.4f}), "
            f"dim={m['vector_size']}, latency={m['latency_ms']} ms."
        )
    if len(working_hf_sorted) >= 2:
        m = working_hf_sorted[1]
        sc = embed_score(m)
        console.print(
            f"  2. [magenta]{m['model']}[/magenta] — "
            f"Runner-up (score {sc:+.4f}), "
            f"dim={m['vector_size']}, latency={m['latency_ms']} ms. "
            f"Consider if the top model is slow or rate-limited on your plan."
        )
    if not working_hf_sorted:
        console.print("  [red]No working HF embedding models found — check your HF_TOKEN.[/red]")

    console.print(
        "\n[dim italic]NOTE: These are suggestions only. "
        "Do NOT edit .env yet — share this report and choose together.[/dim italic]\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    console.print(
        "\n[bold green]tow-support-assistant — Model Availability Check[/bold green]\n"
        "[dim]This script tests your API keys live. No keys are printed.[/dim]\n"
    )

    groq_results = check_groq()
    hf_results = check_hf_embeddings()
    print_summary(groq_results, hf_results)

    log_info(stage="check_models", msg="Model check complete.")


if __name__ == "__main__":
    main()
