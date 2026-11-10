"""Draft pipeline (LangGraph):
build_query -> [greeting/vague: clarify | anything else: retrieve]
retrieve (KB + past replies, no role filter) -> [KB similarity low and no past reply: holding reply | else: draft]
Public function: generate_draft(list_of_customer_messages) -> dict. It never sends anything."""
import os
import re
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.kb_common import log_error, log_info
from app.llm import DRAFT_MODEL, QUERY_MODEL, LLMError, chat_json
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


def get_past_replies(english_query, conv_id=None):
    """Earlier agent-approved replies for similar questions (empty list if none or on any failure)."""
    from app import past_replies
    return past_replies.search(english_query, conv_id=conv_id)


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


def n_build_query(state):
    data, usage, _ = chat_json(build_query_messages(state["thread"]), QUERY_MODEL, 900,
                               "build_query", state["conv_id"])
    intent = data.get("intent") if data.get("intent") in {"greeting", "vague", "real_question", "off_topic"} \
        else "real_question"
    role = data.get("role_guess") if data.get("role_guess") in {"broadcaster", "participant"} else "unknown"
    q = (data.get("english_query") or "").strip()
    if intent in ("real_question", "off_topic") and not q:
        q = state["thread"]
    info = {"intent": intent, "english_query": q, "role_guess": role,
            "language_and_script": _fix_language(state["thread"], data.get("language_and_script") or ""),
            "ambiguous": bool(data.get("ambiguous")),
            "feature": (data.get("feature") or "unknown").strip()}
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
        data, usage, _ = chat_json(build_clarify_messages(state["thread"], LANG_NOTE[lang]), DRAFT_MODEL, 600,
                                   "clarify", state["conv_id"])
        draft = (data.get("draft") or "").strip().replace("\u202f", " ").replace("\u00a0", " ")
    except LLMError as e:
        log_error("LLM_ERROR", "clarify", f"conv={state['conv_id']} Clarify draft failed, using the fixed text: {e}")
    return {"result": {"status": "CLARIFY", "draft": draft or CLARIFY_TEXT[lang], "used": []},
            "tokens": _add_tokens(state, usage)}


def n_retrieve(state):
    info = state["query_info"]
    # No role filter: a wrong role guess used to hide the right chunks (e.g. credit chunks are broadcaster-only)
    chunks = get_retriever().search(info["english_query"], role=None,
                                    top_k=4 if info["ambiguous"] else 3, conv_id=state["conv_id"])
    top = max((c["dense_sim"] for c in chunks if c["dense_sim"] is not None), default=0.0)
    return {"chunks": chunks, "top_dense": top,
            "past": get_past_replies(info["english_query"], state["conv_id"])}


def route_after_retrieve(state):
    # a matching past reply is enough to draft even when the KB match is weak
    return "low_sim" if state["top_dense"] < SIM_THRESHOLD and not state.get("past") else "draft"


def n_low_sim(state):
    log_info("similarity_check", f"conv={state['conv_id']} top dense {state['top_dense']:.3f} is below "
             f"{SIM_THRESHOLD}. Returning holding reply without calling the draft model.")
    return {"result": {"status": "NOT_IN_KB", "draft": HOLD_TEXT[_lang_key(state["query_info"])], "used": []}}


def n_draft(state):
    chunks = state["chunks"]
    msgs = build_draft_messages(state["thread"], [{"id": c["id"], "text": c["text"]} for c in chunks],
                                state.get("past"), LANG_NOTE[_lang_key(state["query_info"])])
    data, usage, _ = chat_json(msgs, DRAFT_MODEL, 1800, "draft", state["conv_id"])
    status = data.get("status")
    draft = (data.get("draft") or "").strip().replace("\u2011", "-").replace("\u2010", "-") \
        .replace("\u202f", " ").replace("\u00a0", " ")
    lang = _lang_key(state["query_info"])
    if status not in STATUSES or not draft:
        log_error("LLM_ERROR", "draft", f"conv={state['conv_id']} Draft answer had status={status!r} and "
                  f"{'an empty' if not draft else 'a'} draft. Using the safe holding reply.")
        status, draft = "NOT_IN_KB", HOLD_TEXT[lang]
    ids = {c["id"] for c in chunks} | {p["id"] for p in (state.get("past") or [])}
    used = [u for u in (data.get("used") or []) if u in ids]
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
