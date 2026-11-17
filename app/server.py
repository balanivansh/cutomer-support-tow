"""FastAPI server for the Support Agent Web Portal: drafts only when the support agent explicitly
requests Generate. Nothing is sent automatically.
Run from the project root:  uvicorn app.server:app --port 8000"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import features, review, store
from app.kb_common import log_error, log_info
from app.llm import DRAFT_MODEL, QUERY_MODEL, LLMError, chat_json
from app.pipeline import generate_draft, get_retriever

MAX_TEXT = 2000
REVIEW_HTML = Path(__file__).parent / "static" / "review.html"

_generating = set()
_gate = asyncio.Semaphore(2)  # local protection; replace with a shared queue before multi-instance deployment


def _warm_up():
    try:
        get_retriever().search("warm up", role=None, top_k=1, conv_id="warmup")
        log_info("startup", "Retriever and embedding model loaded")
    except Exception as e:
        log_error("RETRIEVAL_ERROR", "startup", "Warm-up search failed; the first draft will load the model", e)


@asynccontextmanager
async def lifespan(app):
    store.init_db()
    await asyncio.to_thread(_warm_up)
    yield


app = FastAPI(lifespan=lifespan)

# Dev: allow the Lovable agent portal (any origin). Tighten allow_origins to the real portal URLs before production.
# We send no cookies (auth is the company token), so allow_credentials stays False, which is what makes "*" valid.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   allow_credentials=False)


class Text(BaseModel):
    text: str


class Msg(BaseModel):
    sender: Literal["customer", "agent"]
    text: str


class DraftRequest(BaseModel):
    chat_id: int  # rox-chat chat.id, used only as a label and a per-chat lock
    latest_message_id: int  # rox-chat chat_message.id of the newest message sent in `messages`
    messages: list[Msg] = Field(min_length=1, max_length=50)  # oldest first
    agent_user_id: int  # REQUIRED: who clicked Generate (the logged-in userId). No silent default.
    customer_user_id: int | None = None  # the customer who sent the message (profile id), for the audit row


class FeedbackRequest(BaseModel):
    draft_id: int  # the id returned by /api/draft
    final_text: str  # what the agent really sent
    customer_text: str = ""
    sent_message_id: int | None = None  # chat_message.id of the sent reply, if the portal knows it


@app.post("/api/draft")
async def api_draft(body: DraftRequest):
    """Stateless draft: rox-chat sends the recent messages, gets the draft text back in one call."""
    cid = f"chat-{body.chat_id}"
    items = [{"sender": m.sender, "text": m.text.strip()[:MAX_TEXT]} for m in body.messages if m.text.strip()]
    context = store.bound_context(items)
    last_agent = max((n for n, m in enumerate(context) if m["sender"] == "agent"), default=-1)
    if not any(m["sender"] == "customer" for m in context[last_agent + 1:]):
        raise HTTPException(409, "There are no recent customer messages to answer")
    if store.message_has_draft(body.chat_id, body.latest_message_id):
        raise HTTPException(409, "A draft already exists for this customer message. Edit it instead")
    if cid in _generating:
        raise HTTPException(409, "A draft is already being generated for this chat")
    _generating.add(cid)
    try:
        lines = [f"{m['sender'].title()}: {m['text']}" for m in context]
        async with _gate:
            result = await asyncio.to_thread(generate_draft, lines, cid)
        draft_id = store.save_draft(body.chat_id, result, body.latest_message_id, body.agent_user_id,
                                    body.customer_user_id, DRAFT_MODEL)
        log_info("api_draft", f"chat={body.chat_id} draft={draft_id} status={result.get('status')} "
                              f"agent={body.agent_user_id} customer={body.customer_user_id}")
        return {"draft_id": draft_id, "status": result.get("status"), "draft": result.get("draft", ""),
                "used": result.get("used") or [], "ms": result.get("ms")}
    except store.DraftExists:
        raise HTTPException(409, "A draft already exists for this customer message. Edit it instead")
    except Exception as e:
        log_error("API_ERROR", "api_draft", f"chat={body.chat_id} Draft generation failed", e)
        raise HTTPException(500, "Draft generation failed")
    finally:
        _generating.discard(cid)


@app.post("/api/feedback")
async def api_feedback(body: FeedbackRequest):
    """Stateless outcome: compares the sent reply with the saved draft and feeds the weekly review."""
    final = body.final_text.strip()
    if not final:
        raise HTTPException(400, "Empty reply")
    if not store.draft_exists(body.draft_id):
        raise HTTPException(404, "Draft not found")
    outcome = store.save_feedback(body.draft_id, body.customer_text[:MAX_TEXT], final[:MAX_TEXT],
                                  body.sent_message_id)
    return {"outcome": outcome}


class Apply(BaseModel):
    mode: str = "replace"  # replace | add
    feature: str = ""
    part: str = ""  # topic letter, used by add only
    title: str = ""  # the question
    audience: str = ""  # "Broadcaster, Participant" or "Everyone"
    body: str = ""
    keywords: str = ""


class NewAudience(BaseModel):
    name: str


@app.get("/review")
async def review_page():
    return FileResponse(REVIEW_HTML)


def _feature_json(f):
    """One shape for GET and POST, so the page can use the POST result directly."""
    return {"id": f["id"], "name": f["name"], "audiences": f.get("audiences", []), "topics": features.topics(f)}


@app.get("/api/features")
async def features_list():
    return [_feature_json(f) for f in features.load()]


class NewFeature(BaseModel):
    name: str


@app.post("/api/features")
async def features_add(b: NewFeature):
    try:
        return _feature_json(features.add_feature(b.name))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/features/{fid}/audience")
async def features_add_audience(fid: str, b: NewAudience):
    try:
        return {"audiences": features.add_audience(fid, b.name)}
    except ValueError as e:
        raise HTTPException(400, str(e))


class NewTopic(BaseModel):
    name: str


@app.post("/api/features/{fid}/topics")
async def features_add_topic(fid: str, b: NewTopic):
    """Human action: adds a '## PART X: NAME' heading to this feature's own KB file only."""
    try:
        features.add_topic(fid, b.name)
        return _feature_json(features.get(fid))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/review/proposals")
async def review_list():
    return [p for p in store.list_proposals("pending") if p.get("extra")]  # cards from the old page have no extra


@app.post("/api/review/generate")
async def review_generate():
    return {"count": await asyncio.to_thread(review.build_report)}


@app.post("/api/review/{pid}/apply")
async def review_apply(pid: int, b: Apply):
    try:
        await asyncio.to_thread(review.apply_proposal, pid, b.mode, b.feature, b.part, b.title, b.audience,
                                b.body, b.keywords)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log_error("API_ERROR", "review", f"Applying proposal {pid} failed", e)
        raise HTTPException(500, f"Could not apply the change: {e}")
    return {"ok": True}


@app.post("/api/review/{pid}/reject")
async def review_reject(pid: int):
    try:
        await asyncio.to_thread(review.reject_proposal, pid)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/translate")
async def translate(body: Text):
    text = body.text.strip()
    if not text:
        return {"english": ""}
    msgs = [{"role": "system", "content": "Translate the customer-support reply into clear, natural English. "
             "Keep emojis. Answer ONLY with JSON: {\"english\": \"<translation>\"}"},
            {"role": "user", "content": text}]
    try:
        data, _, _ = await asyncio.to_thread(chat_json, msgs, QUERY_MODEL, 900, "translate")
    except LLMError as e:
        raise HTTPException(502, f"Translation failed: {e}")
    return {"english": str(data.get("english", ""))}
