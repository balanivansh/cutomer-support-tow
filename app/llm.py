"""Groq adapter: one function for every LLM call. Handles empty answers, rate limits, bad JSON,
and logs tokens used, so problems show up as readable API_ERROR / LLM_ERROR lines.
Swap providers later by changing only this file."""
import json
import os
import re
import time

from app.kb_common import log_error, log_info  # also loads .env

LLM_MODEL = os.getenv("LLM_MODEL") or "openai/gpt-oss-120b"
QUERY_MODEL = os.getenv("QUERY_MODEL") or LLM_MODEL
DRAFT_MODEL = os.getenv("DRAFT_MODEL") or LLM_MODEL

_client = None


class LLMError(Exception):
    """Raised when the LLM call cannot produce a usable answer after retries."""


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            log_error("CONFIG_ERROR", "llm_client", "GROQ_API_KEY is empty. Fill it in .env.")
            raise LLMError("GROQ_API_KEY missing")
        from groq import Groq
        _client = Groq(api_key=key)
    return _client


def _parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


def _retry_after(exc, default=10):
    try:
        return min(float(exc.response.headers.get("retry-after", default)), 30.0)
    except Exception:
        return default


def chat_json(messages, model, max_tokens, stage, conv_id=None, attempts=3):
    """Returns (data_dict, usage_dict, latency_ms). Raises LLMError if it cannot."""
    import groq
    tag = f"conv={conv_id} " if conv_id else ""
    client = _get_client()
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=0)
    if model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"
    last_problem = "unknown"

    for attempt in range(1, attempts + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
        except groq.RateLimitError as e:
            wait = _retry_after(e)
            last_problem = "rate limit (429)"
            log_error("API_ERROR", stage, f"{tag}Groq rate limit (429) on {model}. Waiting {wait:.0f}s, "
                      f"attempt {attempt}/{attempts}. Free tier: 8,000 tokens/min, 200,000/day.", e)
            time.sleep(wait)
            continue
        except groq.BadRequestError as e:
            if "reasoning_effort" in kwargs and "reasoning" in str(e).lower():
                kwargs.pop("reasoning_effort")
                log_error("API_ERROR", stage, f"{tag}Model {model} rejected reasoning_effort. Retrying without it.", e)
                continue
            log_error("API_ERROR", stage, f"{tag}Groq rejected the request for {model}", e)
            raise LLMError(f"bad request: {e}")
        except (groq.APIConnectionError, groq.APIStatusError) as e:
            last_problem = "API/network error"
            log_error("API_ERROR", stage, f"{tag}Groq call failed, attempt {attempt}/{attempts}", e)
            time.sleep(2)
            continue

        ms = int((time.time() - t0) * 1000)
        usage = {"in": getattr(resp.usage, "prompt_tokens", 0) or 0,
                 "out": getattr(resp.usage, "completion_tokens", 0) or 0}
        content = (resp.choices[0].message.content or "").strip()
        log_info(stage, f"{tag}{model} answered in {ms} ms, tokens in={usage['in']} out={usage['out']}")

        if not content:
            last_problem = "empty answer"
            kwargs["max_tokens"] = int(kwargs["max_tokens"] * 2)
            log_error("LLM_ERROR", stage, f"{tag}Empty answer from {model} (it probably spent its token budget "
                      f"on reasoning). Retrying with max_tokens={kwargs['max_tokens']}.")
            continue
        try:
            return _parse_json(content), usage, ms
        except Exception as e:
            last_problem = "invalid JSON"
            log_error("LLM_ERROR", stage, f"{tag}Answer was not valid JSON: {content[:120]!r}", e)
            continue

    raise LLMError(f"{stage}: gave up after {attempts} attempts ({last_problem})")
