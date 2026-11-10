"""
scripts/check_llm_final.py
==========================
Final LLM evaluation on Groq — three tests:
  1. Multilingual restatement with reasoning_effort on gpt-oss-120b / gpt-oss-20b
  2. JSON classification of 4 customer messages (3 models)
  3. Grounded-draft / NOT_IN_KB compliance test (2 models)
  4. Summary table

USAGE:
  python scripts/check_llm_final.py

This script does NOT write any choice into .env.
"""

from __future__ import annotations

import sys
import json
import re
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging + config — set up before anything else
# ---------------------------------------------------------------------------
from app.logging_config import setup_logging, log_error, log_info, log_warning
from app.config import settings

setup_logging(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)
settings.assert_keys_present("GROQ_API_KEY")

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import httpx
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT = settings.HTTP_TIMEOUT_SECONDS

REASONING_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
ALL_JSON_MODELS   = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
GROUNDED_MODELS   = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]

CUSTOMER_MESSAGES = [
    "goodmorning",
    "tug credit nahi mil raha please help",
    "मेरा TUG क्रेडिट कम हो गया है, कृपया जाँच करके बताइए",
    "bhai coins match ke baad nahi aaye, mai winner tha",
]

GROUNDED_KB = (
    "Free credits let a broadcaster start a match without spending coins. "
    "Credits are a weekly total set by the Health Heart: Green 3, Yellow 2, Red 1. "
    "Credits are refreshed every Monday."
)

GROUNDED_SYSTEM = (
    "Answer ONLY from the knowledge text given. "
    "Reply in the same language and script as the customer. "
    "If the answer is not in the text, reply exactly: NOT_IN_KB. Never guess."
)

GROUNDED_QUESTIONS = [
    {
        "id": "a",
        "text": "tug credit kam ho gaya hai, kyun?",
        "expect": "grounded_answer",
        "note": "Should answer from text (Hinglish / Roman script)",
    },
    {
        "id": "b",
        "text": "mera account ban kyu hua?",
        "expect": "NOT_IN_KB",
        "note": "Not in KB — must return NOT_IN_KB, never guess",
    },
]

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _groq_post(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    response_format: Optional[dict] = None,
    stage: str = "",
) -> tuple[Optional[dict], float, str]:
    """
    POST to Groq chat completions.
    Returns (response_json_or_None, latency_ms, rate_limit_info_string).
    """
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if response_format:
        payload["response_format"] = response_format

    t0 = time.perf_counter()
    try:
        resp = httpx.post(
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=TIMEOUT,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000)

        rl_keys = [
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-reset-requests",
            "retry-after",
        ]
        rl_parts = [
            f"{k.replace('x-ratelimit-','').replace('x-','')}"
            f"={resp.headers.get(k)}"
            for k in rl_keys
            if resp.headers.get(k)
        ]
        rl_info = ", ".join(rl_parts) or "—"

        if resp.status_code == 200:
            return resp.json(), latency_ms, rl_info
        else:
            err_msg = ""
            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text[:300])
            except Exception:
                err_msg = resp.text[:300]
            log_error(
                "LLM_ERROR",
                stage=stage or "groq_post",
                msg=f"{model}: HTTP {resp.status_code} — {err_msg}",
            )
            return None, latency_ms, rl_info

    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000)
        log_error("LLM_ERROR", stage=stage or "groq_post", msg=f"{model}: {exc}", exc=exc)
        return None, latency_ms, "—"


def _extract_content(response_json: dict, model: str) -> str:
    """
    Pull the assistant message text from a Groq response.
    For reasoning models the API may include a 'reasoning' field inside the
    message; we want only the visible 'content'.
    If content is empty/None but 'reasoning' is present, still return content
    (empty string) so the caller knows the visible answer is blank.
    """
    try:
        msg = response_json["choices"][0]["message"]
        content = msg.get("content") or ""
        return content.strip()
    except (KeyError, IndexError):
        return ""


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks that some models emit before the answer."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Test 1 — Multilingual restatement with reasoning_effort
# ---------------------------------------------------------------------------

MULTILINGUAL_PROMPT = (
    "Restate this message as one clear English sentence: "
    "का TUG कम हो गया है। कृपया इसकी जाँच कर हमें बताने"
)


def test_multilingual_restatement() -> list[dict]:
    console.rule("[bold cyan]Test 1 — Multilingual Restatement (reasoning_effort=low)[/bold cyan]")
    results: list[dict] = []

    for model in REASONING_MODELS:
        console.print(f"\n  Model: [cyan]{model}[/cyan]")
        resp, latency, rl = _groq_post(
            model,
            messages=[{"role": "user", "content": MULTILINGUAL_PROMPT}],
            max_tokens=1024,
            reasoning_effort="low",
            stage="test1_multilingual",
        )
        if resp is None:
            results.append(
                {"model": model, "non_empty": False, "answer": "", "latency_ms": latency, "rl": rl}
            )
            console.print(f"    [red]❌ Request failed[/red]  ({latency} ms)")
            continue

        content = _extract_content(resp, model)
        # Also check if reasoning tokens were returned (informational only)
        try:
            usage = resp.get("usage", {})
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        except Exception:
            reasoning_tokens = 0

        non_empty = bool(content)
        results.append(
            {"model": model, "non_empty": non_empty, "answer": content, "latency_ms": latency, "rl": rl}
        )

        status = "✅" if non_empty else "⚠️  EMPTY"
        console.print(f"    {status}  ({latency} ms)  reasoning_tokens={reasoning_tokens}")
        console.print(f"    Answer: [italic]{content if content else '(empty)'}[/italic]")
        console.print(f"    Rate-limit: [dim]{rl}[/dim]")

    return results


# ---------------------------------------------------------------------------
# Test 2 — JSON classification
# ---------------------------------------------------------------------------

JSON_SYSTEM = (
    "You are a support-message classifier. "
    "For the customer message given, return ONLY a JSON object with these fields:\n"
    "  intent: one of greeting | vague | real_question\n"
    "  english_query: a clear English restatement of what the customer is asking "
    "(or empty string for greetings)\n"
    "  role_guess: one of broadcaster | participant | unknown\n"
    "  language_and_script: e.g. English, Hindi-Devanagari, Hinglish-Roman, mixed\n"
    "Return nothing else — no markdown, no explanation."
)

JSON_SCHEMA = {
    "type": "json_object"
}


def _classify_one(model: str, message: str, idx: int) -> dict:
    """Classify a single message; return result dict."""
    resp, latency, _ = _groq_post(
        model,
        messages=[
            {"role": "system", "content": JSON_SYSTEM},
            {"role": "user", "content": message},
        ],
        max_tokens=300,
        reasoning_effort="low" if model in REASONING_MODELS else None,
        response_format=JSON_SCHEMA,
        stage=f"test2_json_msg{idx}",
    )
    if resp is None:
        return {"valid": False, "json_obj": None, "latency_ms": latency, "error": "request failed"}

    raw = _extract_content(resp, model)
    raw = _strip_think_tags(raw)

    # Try to parse JSON
    try:
        parsed = json.loads(raw)
        return {"valid": True, "json_obj": parsed, "latency_ms": latency, "raw": raw, "error": ""}
    except json.JSONDecodeError as e:
        log_error(
            "LLM_ERROR",
            stage=f"test2_json_parse_msg{idx}",
            msg=f"{model} msg{idx}: JSON parse failed — {e}",
        )
        return {"valid": False, "json_obj": None, "latency_ms": latency, "raw": raw, "error": str(e)}


def test_json_classification() -> dict[str, list[dict]]:
    """Returns {model: [result_per_message]}"""
    console.rule("[bold yellow]Test 2 — JSON Classification (4 messages × 3 models)[/bold yellow]")
    all_results: dict[str, list[dict]] = {}

    for model in ALL_JSON_MODELS:
        console.print(f"\n  Model: [yellow]{model}[/yellow]")
        model_results: list[dict] = []

        for idx, msg in enumerate(CUSTOMER_MESSAGES, start=1):
            result = _classify_one(model, msg, idx)
            model_results.append(result)

            status = "✅ valid JSON" if result["valid"] else "❌ invalid JSON"
            console.print(f"    [{idx}] {status}  ({result['latency_ms']} ms)")
            console.print(f"         Input: [dim]{msg}[/dim]")
            if result["valid"] and result["json_obj"]:
                obj = result["json_obj"]
                console.print(
                    f"         intent=[green]{obj.get('intent','?')}[/green]  "
                    f"role=[cyan]{obj.get('role_guess','?')}[/cyan]  "
                    f"lang=[magenta]{obj.get('language_and_script','?')}[/magenta]"
                )
                console.print(f"         english_query: [italic]{obj.get('english_query','?')}[/italic]")
            else:
                console.print(f"         Raw: [red]{result.get('raw','')[:200]}[/red]")
                if result.get("error"):
                    console.print(f"         Error: [red]{result['error']}[/red]")

            # Small pause to avoid rate-limiting
            time.sleep(0.4)

        all_results[model] = model_results

    return all_results


# ---------------------------------------------------------------------------
# Test 3 — Grounded draft / NOT_IN_KB compliance
# ---------------------------------------------------------------------------

def _grounded_one(model: str, question: str, q_id: str) -> dict:
    resp, latency, _ = _groq_post(
        model,
        messages=[
            {"role": "system", "content": f"{GROUNDED_SYSTEM}\n\nKnowledge text:\n{GROUNDED_KB}"},
            {"role": "user", "content": question},
        ],
        max_tokens=512,
        reasoning_effort="low" if model in REASONING_MODELS else None,
        stage=f"test3_grounded_{q_id}",
    )
    if resp is None:
        return {"answer": "", "latency_ms": latency, "error": "request failed"}

    raw = _extract_content(resp, model)
    raw = _strip_think_tags(raw)
    return {"answer": raw, "latency_ms": latency, "error": ""}


def test_grounded_draft() -> dict[str, list[dict]]:
    """Returns {model: [result_per_question]}"""
    console.rule("[bold magenta]Test 3 — Grounded Draft / NOT_IN_KB[/bold magenta]")
    all_results: dict[str, list[dict]] = {}

    for model in GROUNDED_MODELS:
        console.print(f"\n  Model: [magenta]{model}[/magenta]")
        model_results: list[dict] = []

        for q in GROUNDED_QUESTIONS:
            result = _grounded_one(model, q["text"], q["id"])
            answer = result["answer"]

            if q["expect"] == "NOT_IN_KB":
                correct = "NOT_IN_KB" in answer
                rule_tag = "✅ RETURNED NOT_IN_KB" if correct else "❌ GUESSED / WRONG"
            else:
                # Expect a real grounded answer — non-empty, no "NOT_IN_KB"
                correct = bool(answer) and "NOT_IN_KB" not in answer
                rule_tag = "✅ ANSWERED FROM KB" if correct else "❌ EMPTY / REFUSED"

            result["correct"] = correct
            result["expect"] = q["expect"]
            model_results.append(result)

            console.print(
                f"    [{q['id']}] {rule_tag}  ({result['latency_ms']} ms)  — {q['note']}"
            )
            console.print(f"         Q: [dim]{q['text']}[/dim]")
            console.print(f"         A: [italic]{answer if answer else '(empty)'}[/italic]")

            time.sleep(0.4)

        all_results[model] = model_results

    return all_results


# ---------------------------------------------------------------------------
# Test 4 — Summary table
# ---------------------------------------------------------------------------

def print_summary(
    t1: list[dict],
    t2: dict[str, list[dict]],
    t3: dict[str, list[dict]],
) -> None:
    console.rule("[bold green]Test 4 — Summary Table[/bold green]")

    # Build lookup for t1 (only reasoning models tested there)
    t1_map = {r["model"]: r for r in t1}

    tbl = Table(box=box.ROUNDED, show_lines=True, title="LLM Final Evaluation — Groq")
    tbl.add_column("Model", style="cyan", no_wrap=True)
    tbl.add_column("Multilingual\nAnswer Non-Empty", justify="center")
    tbl.add_column("JSON Valid\n(x of 4)", justify="center")
    tbl.add_column("Grounded\nAnswer ✓", justify="center")
    tbl.add_column("NOT_IN_KB\nReturned ✓", justify="center")
    tbl.add_column("Avg Latency\n(ms)", justify="right")
    tbl.add_column("Notes", style="dim", max_width=36)

    all_models = sorted(set(ALL_JSON_MODELS + GROUNDED_MODELS + REASONING_MODELS))

    for model in all_models:
        # T1
        t1_r = t1_map.get(model)
        multi = "✅" if (t1_r and t1_r["non_empty"]) else ("—" if not t1_r else "❌")

        # T2
        t2_r = t2.get(model, [])
        json_valid = sum(1 for r in t2_r if r["valid"])
        json_str = f"{json_valid}/4" if t2_r else "—"

        # T3
        t3_r = t3.get(model, [])
        grounded_ok = None
        not_in_kb_ok = None
        for r in t3_r:
            if r["expect"] == "grounded_answer":
                grounded_ok = r["correct"]
            elif r["expect"] == "NOT_IN_KB":
                not_in_kb_ok = r["correct"]

        g_str = ("✅" if grounded_ok else ("❌" if grounded_ok is False else "—"))
        n_str = ("✅" if not_in_kb_ok else ("❌" if not_in_kb_ok is False else "—"))

        # Avg latency across all tests this model ran
        all_latencies: list[int] = []
        if t1_r:
            all_latencies.append(t1_r["latency_ms"])
        for r in t2_r:
            all_latencies.append(r["latency_ms"])
        for r in t3_r:
            all_latencies.append(r["latency_ms"])
        avg_lat = round(sum(all_latencies) / len(all_latencies)) if all_latencies else 0

        # Notes
        notes_parts = []
        if model in REASONING_MODELS:
            notes_parts.append("reasoning_effort=low supported")
        if t3_r:
            answers = [r["answer"] for r in t3_r if r["expect"] == "grounded_answer"]
            if answers and answers[0]:
                # Check if reply is in Hinglish/Roman
                # Simple heuristic: no Devanagari chars → Roman script
                has_devanagari = bool(re.search(r"[\u0900-\u097F]", answers[0]))
                notes_parts.append("Roman script ✅" if not has_devanagari else "Devanagari used ⚠️")

        tbl.add_row(
            model,
            multi,
            json_str,
            g_str,
            n_str,
            str(avg_lat),
            "; ".join(notes_parts) or "—",
        )

    console.print(tbl)
    console.print(
        "\n[dim italic]No choice has been written to .env. "
        "Share this report to decide LLM_MODEL together.[/dim italic]\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    console.print(
        "\n[bold green]tow-support-assistant — LLM Final Evaluation[/bold green]\n"
        "[dim]Testing: openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b[/dim]\n"
        "[dim]No keys will be printed. No .env changes will be made.[/dim]\n"
    )

    t1_results = test_multilingual_restatement()
    time.sleep(1)

    t2_results = test_json_classification()
    time.sleep(1)

    t3_results = test_grounded_draft()

    print_summary(t1_results, t2_results, t3_results)
    log_info(stage="check_llm_final", msg="LLM final evaluation complete.")


if __name__ == "__main__":
    main()
