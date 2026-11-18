"""Draft pipeline (LangGraph):
build_query -> [greeting/vague: clarify | anything else: retrieve]
retrieve (reviewed KB only, no role filter) -> [KB similarity low: holding reply | else: draft]
Public function: generate_draft(list_of_customer_messages) -> dict. It never sends anything."""
import json
import os
import re
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.kb_common import embed, log_error, log_info
from app.llm import DRAFT_MODEL, QUERY_MODEL, LLMError, LLMOutputTooLong, chat_json
from app.prompts import build_clarify_messages, build_draft_messages, build_query_messages

SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD") or 0.50)  # min dense similarity of the English query
STATUSES = {"ANSWERED", "CLARIFY", "NOT_IN_KB"}

CLARIFY_TEXT = {
    "en": "Hello 😊 Could you please tell us what issue you're facing?",
    "hinglish": "Hello 😊 Aap kis issue me help chahte hain? Please thoda detail me bataiye.",
    "hi": "नमस्ते 😊 कृपया बताइए कि आपको किस समस्या में मदद चाहिए?",
}
HOLD_TEXT = {
    "en": "Let me check and get back to you, thank you for reporting.",
    "hinglish": "Hum check karke aapko update karte hain, report karne ke liye thank you.",
    "hi": "हम जाँच करके आपको अपडेट करते हैं, रिपोर्ट करने के लिए धन्यवाद।",
}

LANG_NOTE = {
    "en": "The customer wrote in English. Reply in clean English only, no Hinglish.",
    "hinglish": "The customer wrote Hinglish in Roman script. Reply in Roman-script Hinglish.",
    "hi": "The customer wrote Hindi in Devanagari. Reply in Devanagari Hindi.",
}

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from app.retriever import Retriever
        _retriever = Retriever()
    return _retriever


def get_past_replies(english_query, conv_id=None, query_vector=None):
    """Earlier agent-approved replies for similar questions in the same week (empty list if none or on any failure)."""
    from app import past_replies
    return past_replies.search(english_query, conv_id=conv_id, query_vector=query_vector)


class DraftState(TypedDict, total=False):
    conv_id: str
    thread: str
    query_info: dict
    chunks: list
    past: list
    top_dense: float
    result: dict
    tokens: dict


def _lang_key(info):
    l = (info.get("language_and_script") or "").lower()
    if l == "english":
        return "en"
    if "devanagari" in l:
        return "hi"
    return "hinglish"


_DEVA = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")


def _fix_language(thread, lang):
    """Decide the script from the actual letters, so the reply script cannot be guessed wrong."""
    deva, latin = len(_DEVA.findall(thread)), len(_LATIN.findall(thread))
    if deva == 0:
        return "english" if lang == "english" else "hinglish_roman"
    return "hindi_devanagari" if deva >= latin else "mixed"


def _add_tokens(state, usage):
    t = dict(state.get("tokens") or {"in": 0, "out": 0})
    t["in"] += usage["in"]
    t["out"] += usage["out"]
    return t


def _clean_questions(raw, intent, max_questions=4):
    """Separate customer questions [{q, feature}] in the order asked, repeats removed. [] when there is only one."""
    out, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        q = str(item.get("q") if isinstance(item, dict) else item or "").strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append({"q": q, "feature": str((item.get("feature") if isinstance(item, dict) else "") or "").strip()})
    return out[:max_questions] if intent in ("real_question", "off_topic") and len(out) > 1 else []


QUERY_MAX_TOKENS = int(os.getenv("QUERY_MAX_TOKENS") or 800)  # build_query output limit (reasoning + JSON)
SHRINK_TARGET_CHARS = 800  # total size of the customer lines after the one-time shrink

_SHRINK_PROMPT = ("Shorten these chat lines so that together they are under %d characters. Keep EVERY question, "
                  "feature name (Tug of War, PK, Live...), number and the language/script of each line. Remove only "
                  "filler and repeated words. Return JSON only: {\"lines\": [\"...\"]} with exactly %d items, same order.")


def _shrink_thread(thread, conv_id):
    """Only used when build_query was CUT by the token limit. One extra LLM call shortens the Customer lines
    (the Agent lines stay as they are). Returns (shorter_thread, usage). Raises LLMError if it cannot."""
    rows = [re.match(r"^(\d+)\.\s*([^:]+):\s*(.*)$", l) for l in thread.splitlines()]
    cust = [m for m in rows if m and m.group(2).startswith("Customer")]
    if not cust:
        raise LLMError("nothing to shrink")
    texts = [m.group(3) for m in cust]
    data, usage, _ = chat_json(
        [{"role": "system", "content": _SHRINK_PROMPT % (SHRINK_TARGET_CHARS, len(texts))},
         {"role": "user", "content": json.dumps(texts, ensure_ascii=False)}],
        QUERY_MODEL, 600, "shrink_context", conv_id)
    new = data.get("lines")
    if not isinstance(new, list) or len(new) != len(texts):
        raise LLMError("shrink_context: wrong number of lines")
    it = iter(new)
    out = []
    for l, m in zip(thread.splitlines(), rows):
        out.append(f"{m.group(1)}. {m.group(2)}: {next(it)}" if m in cust else l)
    return "\n".join(out), usage


def n_build_query(state):
    shrunk = None
    try:
        data, usage, _ = chat_json(build_query_messages(state["thread"]), QUERY_MODEL, QUERY_MAX_TOKENS,
                                   "build_query", state["conv_id"])
    except LLMOutputTooLong as first:
        # Cut by the limit. ONE extra call shortens the customer lines, then build_query runs once more.
        log_error("LLM_ERROR", "build_query", f"conv={state['conv_id']} output too long ({first.max_tokens} tokens). "
                  f"Shrinking the customer lines (one extra call) and retrying once.")
        try:
            shrunk, u1 = _shrink_thread(state["thread"], state["conv_id"])
            log_info("shrink_context", f"conv={state['conv_id']} before={len(state['thread'])} chars "
                                       f"after={len(shrunk)} chars. Shrunk thread: {shrunk!r}")
            data, usage, _ = chat_json(build_query_messages(shrunk), QUERY_MODEL, QUERY_MAX_TOKENS,
                                       "build_query", state["conv_id"])
            usage = {"in": usage["in"] + u1["in"] + first.usage["in"], "out": usage["out"] + u1["out"] + first.usage["out"]}
        except LLMOutputTooLong as second:
            second.first_raw, second.shrunk = first.raw, shrunk  # saved on the draft row for the reviewer
            raise
        except LLMError as e:
            first.shrink_error = str(e)
            first.shrunk = shrunk
            raise first
    intent = data.get("intent") if data.get("intent") in {"greeting", "vague", "real_question", "off_topic"} \
        else "real_question"
    role = data.get("role_guess") if data.get("role_guess") in {"broadcaster", "participant"} else "unknown"
    q = (data.get("english_query") or "").strip()
    if intent in ("real_question", "off_topic") and not q:
        q = state["thread"]
    info = {"intent": intent, "english_query": q, "role_guess": role,
            "language_and_script": _fix_language(state["thread"], data.get("language_and_script") or ""),
            "ambiguous": bool(data.get("ambiguous")),
            "feature": (data.get("feature") or "unknown").strip(),
            "questions": _clean_questions(data.get("questions"), intent)}
    if shrunk:
        info["shrunk_thread"] = shrunk  # saved in query_json: shows what the AI really read
    log_info("build_query", f"conv={state['conv_id']} intent={intent} role={role} "
             f"lang={info['language_and_script']} ambiguous={info['ambiguous']} feature={info['feature']} query={q[:80]!r}")
    return {"query_info": info, "tokens": _add_tokens(state, usage)}


def route_after_query(state):
    intent = state["query_info"]["intent"]
    # off_topic still goes to retrieve: the similarity gate and the draft model's NOT_IN_KB decide
    return "clarify" if intent in ("greeting", "vague") else "retrieve"


def n_clarify(state):
    """Greeting / no topic: the model writes a short, varied 'how can I help' question (no KB search)."""
    lang = _lang_key(state["query_info"])
    draft, usage = "", {"in": 0, "out": 0}
    try:
        data, usage, _ = chat_json(build_clarify_messages(state["thread"], LANG_NOTE[lang]), DRAFT_MODEL, 250,
                                   "clarify", state["conv_id"])
        draft = (data.get("draft") or "").strip().replace("\u202f", " ").replace("\u00a0", " ")
    except LLMError as e:
        log_error("LLM_ERROR", "clarify", f"conv={state['conv_id']} Clarify draft failed, using the fixed text: {e}")
    return {"result": {"status": "CLARIFY", "draft": draft or CLARIFY_TEXT[lang], "used": []},
            "tokens": _add_tokens(state, usage)}


def n_retrieve(state):
    info = state["query_info"]
    eq = info["english_query"]
    qv = embed([eq])[0]
    k = 3 if info["ambiguous"] else 2
    from app import features
    from app.retriever import keyword_features, match_features
    feats = features.load()
    qs = info.get("questions") or [{"q": eq, "feature": info.get("feature")}]
    per_q = 2 if len(qs) <= 2 else 1  # several questions: each gets its own search (fewer chunks each)
    # thread lines look like "3. Customer: text" (numbered), so match the number first
    customer_text = "\n".join(re.sub(r"^\d+\.\s*Customer:\s*", "", l) for l in state["thread"].splitlines()
                              if re.match(r"^\d+\.\s*Customer:", l))
    chunks, seen = [], set()
    for item in qs:
        q = item["q"]
        fids = match_features(item.get("feature") or q, feats)
        # Feature keywords ("tow" -> Tug of War), looked for in the customer's lines only. Soft signal.
        hits = [f for f in keyword_features(q if len(qs) > 1 else customer_text + "\n" + q, feats) if f not in fids]
        fids = fids + hits
        search_q, search_v = q, (qv if q == eq else embed([q])[0])
        names = [f["name"] for f in feats if f["id"] in hits and f["name"].lower() not in q.lower()]
        if names:  # add the real feature name so the word match and the vectors see it
            search_q = q + " " + " ".join(names)
            search_v = embed([search_q])[0]
            log_info("keywords", f"conv={state['conv_id']} matched={hits} query={search_q[:80]!r}")
        k_q = k if len(qs) == 1 else per_q
        for c in get_retriever().search(search_q, role=None, top_k=k_q if len(fids) < 2 else max(k_q, len(fids)),
                                        conv_id=state["conv_id"], query_vector=search_v, feature_ids=fids):
            if c["id"] not in seen:
                seen.add(c["id"])
                chunks.append(c)
    past = get_past_replies(eq, conv_id=state["conv_id"], query_vector=qv)
    top = max((c["dense_sim"] for c in chunks if c["dense_sim"] is not None), default=0.0)
    return {"chunks": chunks, "top_dense": top, "past": past}


def route_after_retrieve(state):
    # a matching past reply is enough to draft even when the KB match is weak
    return "low_sim" if state["top_dense"] < SIM_THRESHOLD and not state.get("past") else "draft"


def n_low_sim(state):
    log_info("similarity_check", f"conv={state['conv_id']} top dense {state['top_dense']:.3f} is below "
             f"{SIM_THRESHOLD}. Returning holding reply without calling the draft model.")
    return {"result": {"status": "NOT_IN_KB", "draft": HOLD_TEXT[_lang_key(state["query_info"])], "used": []}}


def n_draft(state):
    chunks = state["chunks"]
    past = state.get("past") or []
    msgs = build_draft_messages(state["thread"], [{"id": c["id"], "text": c["text"], "feature": c.get("feature_name")} for c in chunks],
                                past, LANG_NOTE[_lang_key(state["query_info"])],
                                [x["q"] for x in state["query_info"].get("questions") or []])
    data, usage, _ = chat_json(msgs, DRAFT_MODEL, 600, "draft", state["conv_id"])
    status = data.get("status")
    draft = (data.get("draft") or "").strip().replace("\u2011", "-").replace("\u2010", "-") \
        .replace("\u202f", " ").replace("\u00a0", " ")
    lang = _lang_key(state["query_info"])
    if status not in STATUSES or not draft:
        log_error("LLM_ERROR", "draft", f"conv={state['conv_id']} Draft answer had status={status!r} and "
                   f"{'an empty' if not draft else 'a'} draft. Using the safe holding reply.")
        status, draft = "NOT_IN_KB", HOLD_TEXT[lang]
    ids = {c["id"] for c in chunks} | {p["id"] for p in past}
    used = []
    for u in data.get("used") or []:
        u = str(u).strip()
        # the model sometimes drops the feature prefix ("B1" for "tow:B1"): map it back to the ids it was shown
        for full in ([u] if u in ids else sorted(i for i in ids if i.split(":")[-1] == u)):
            if full not in used:
                used.append(full)
    log_info("draft", f"conv={state['conv_id']} status={status} used={used}")
    return {"result": {"status": status, "draft": draft, "used": used},
            "tokens": _add_tokens(state, usage)}


_graph = None


def _build_graph():
    g = StateGraph(DraftState)
    g.add_node("build_query", n_build_query)
    g.add_node("clarify", n_clarify)
    g.add_node("retrieve", n_retrieve)
    g.add_node("low_sim", n_low_sim)
    g.add_node("draft", n_draft)
    g.set_entry_point("build_query")
    g.add_conditional_edges("build_query", route_after_query,
                            {"clarify": "clarify", "retrieve": "retrieve"})
    g.add_conditional_edges("retrieve", route_after_retrieve, {"low_sim": "low_sim", "draft": "draft"})
    for n in ("clarify", "low_sim", "draft"):
        g.add_edge(n, END)
    return g.compile()


def generate_draft(messages, conv_id="local"):
    """messages: the customer's unanswered messages, oldest first."""
    global _graph
    if _graph is None:
        _graph = _build_graph()
    t0 = time.time()
    thread = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(messages))
    try:
        out = _graph.invoke({"conv_id": conv_id, "thread": thread, "tokens": {"in": 0, "out": 0}})
    except LLMOutputTooLong as e:
        # Still too long after the one shrink: show the agent a clear message; keep the raw text for the reviewer.
        log_error("LLM_ERROR", "pipeline", f"conv={conv_id} Generated answer is too large: {e}")
        return {"status": "ERROR", "draft": "", "used": [], "error": "Generated answer is too large. "
                "Please write the reply by hand.", "ms": int((time.time() - t0) * 1000),
                "query": {"error": str(e), "raw_output": e.raw, "first_raw_output": getattr(e, "first_raw", None),
                          "shrunk_thread": getattr(e, "shrunk", None), "english_query": "(output was cut)"}}
    except LLMError as e:
        log_error("LLM_ERROR", "pipeline", f"conv={conv_id} Draft could not be generated: {e}")
        return {"status": "ERROR", "draft": "", "used": [], "error": f"LLM problem: {e}",
                "ms": int((time.time() - t0) * 1000)}
    except Exception as e:  # retrieval / unexpected: details already logged by the failing module
        log_error("RETRIEVAL_ERROR", "pipeline", f"conv={conv_id} Pipeline stopped", e)
        return {"status": "ERROR", "draft": "", "used": [], "error": f"{type(e).__name__}: {e}",
                "ms": int((time.time() - t0) * 1000)}
    res = out["result"]
    res.update({
        "sources": [{"id": c["id"], "dense_sim": c["dense_sim"], "bm25": c["bm25"]} for c in out.get("chunks", [])],
        "query": out.get("query_info"), "tokens": out.get("tokens"), "ms": int((time.time() - t0) * 1000),
    })
    return res
