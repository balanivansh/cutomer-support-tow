"""FastAPI server: serves the two-pane page, stores customer messages, and drafts only when
the support agent explicitly requests Generate. Nothing is sent automatically.
Run from the project root:  uvicorn app.server:app --port 8000"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import features, review, store
from app.kb_common import log_error, log_info
from app.llm import QUERY_MODEL, LLMError, chat_json
from app.pipeline import generate_draft, get_retriever

MAX_TEXT = 2000
INDEX_HTML = Path(__file__).parent / "static" / "index.html"
REVIEW_HTML = Path(__file__).parent / "static" / "review.html"

_sockets = {}  # conv_id -> set of open WebSockets
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


class Text(BaseModel):
    text: str


class Send(BaseModel):
    text: str
    draft_id: int | None = None


def _snapshot(cid):
    return {"messages": store.list_messages(cid), "pending": cid in _generating,
            "draft": store.latest_draft(cid)}


async def _push(cid):
    """Send the current state to every browser watching this conversation."""
    snap = _snapshot(cid)
    for ws in list(_sockets.get(cid, ())):
        try:
            await ws.send_json(snap)
        except Exception:
            _sockets.get(cid, set()).discard(ws)


def _require(cid):
    if not store.conversation_exists(cid):
        raise HTTPException(404, "Conversation not found")


@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


@app.post("/api/conversations")
async def new_conversation():
    return {"id": store.create_conversation()}


@app.post("/api/conversations/{cid}/customer")
async def customer_message(cid: str, body: Text):
    _require(cid)
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Empty message")
    store.add_message(cid, "customer", text[:MAX_TEXT])
    await _push(cid)
    return {"ok": True}


@app.post("/api/conversations/{cid}/drafts")
async def generate(cid: str):
    """Generate one agent-requested draft from bounded context for this conversation."""
    _require(cid)
    if cid in _generating:
        raise HTTPException(409, "A draft is already being generated")
    context = store.generation_context(cid)
    last_agent = max((n for n, m in enumerate(context) if m["sender"] == "agent"), default=-1)
    if not context or not any(m["sender"] == "customer" for m in context[last_agent + 1:]):
        raise HTTPException(409, "There are no recent customer messages to answer")
    _generating.add(cid)
    await _push(cid)
    try:
        # Sender labels let the model resolve short customer follow-ups against recent agent replies.
        messages = [f"{m['sender'].title()}: {m['text']}" for m in context]
        async with _gate:
            result = await asyncio.to_thread(generate_draft, messages, cid)
        draft_id = store.save_draft(cid, result, context)
        log_info("generate", f"conv={cid} draft={draft_id} status={result.get('status')} context={[m['id'] for m in context]}")
        return {"draft_id": draft_id, "status": result.get("status"), "latest_message_id": context[-1]["id"]}
    except Exception as e:
        log_error("API_ERROR", "generate", f"conv={cid} Draft generation failed", e)
        raise HTTPException(500, "Draft generation failed")
    finally:
        _generating.discard(cid)
        await _push(cid)


@app.get("/api/conversations/{cid}/state")
async def state(cid: str):
    _require(cid)
    return _snapshot(cid)


@app.websocket("/ws/{cid}")
async def ws_endpoint(ws: WebSocket, cid: str):
    if not store.conversation_exists(cid):
        await ws.close(code=4404)
        return
    await ws.accept()
    _sockets.setdefault(cid, set()).add(ws)
    log_info("websocket", f"conv={cid} browser connected")
    try:
        await ws.send_json(_snapshot(cid))
        while True:
            await ws.receive_text()  # we only push; this just detects disconnects
    except WebSocketDisconnect:
        pass
    finally:
        _sockets.get(cid, set()).discard(ws)
        log_info("websocket", f"conv={cid} browser disconnected")


@app.post("/api/conversations/{cid}/send")
async def send(cid: str, body: Send):
    _require(cid)
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Empty reply")
    if body.draft_id and store.draft_has_new_customer_message(cid, body.draft_id):
        raise HTTPException(409, "A newer customer message arrived. Generate a fresh draft before sending.")
    customer_text = "\n".join(m["text"] for m in store.generation_context(cid) if m["sender"] == "customer")
    store.add_message(cid, "agent", text[:MAX_TEXT])
    store.mark_answered(cid, body.draft_id)
    try:
        from app import past_replies
        await asyncio.to_thread(past_replies.index_pending)
    except Exception as e:
        log_error("RETRIEVAL_ERROR", "past_index", f"conv={cid} Indexing the sent reply failed", e)
    await _push(cid)
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


@app.get("/api/features")
async def features_list():
    return [{"id": f["id"], "name": f["name"], "audiences": f.get("audiences", []), "topics": features.topics(f)}
            for f in features.load()]


class NewFeature(BaseModel):
    name: str


@app.post("/api/features")
async def features_add(b: NewFeature):
    try:
        return features.add_feature(b.name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/features/{fid}/audience")
async def features_add_audience(fid: str, b: NewAudience):
    try:
        return {"audiences": features.add_audience(fid, b.name)}
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
