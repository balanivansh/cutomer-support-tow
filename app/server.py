"""FastAPI server: serves the two-pane page, collects customer messages, waits for a quiet period,
then drafts a reply. Nothing is ever sent to a customer automatically; the agent presses Send.
Run from the project root:  uvicorn app.server:app --port 8000"""
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import features, past_replies, review, store
from app.kb_common import log_error, log_info
from app.llm import QUERY_MODEL, LLMError, chat_json
from app.pipeline import generate_draft, get_retriever

DEBOUNCE = float(os.getenv("DEBOUNCE_SECONDS") or 4)
MAX_TEXT = 2000
INDEX_HTML = Path(__file__).parent / "static" / "index.html"
REVIEW_HTML = Path(__file__).parent / "static" / "review.html"

_state = {}  # conv_id -> {"version": int, "task": asyncio.Task | None, "pending": bool}
_sockets = {}  # conv_id -> set of open WebSockets
_gate = asyncio.Semaphore(2)  # at most 2 drafts at once (Groq free tier is token-limited)


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


def _st(cid):
    return _state.setdefault(cid, {"version": 0, "task": None, "pending": False})


def _snapshot(cid):
    return {"messages": store.list_messages(cid), "pending": _st(cid)["pending"],
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


async def _draft_after_quiet(cid, version):
    s = _st(cid)
    try:
        await asyncio.sleep(DEBOUNCE)
        texts = store.unanswered_customer_texts(cid)
        if not texts:
            s["pending"] = False
            await _push(cid)
            return
        async with _gate:
            result = await asyncio.to_thread(generate_draft, texts, cid)
        if s["version"] != version:
            log_info("aggregator", f"conv={cid} Discarding late draft (a newer message or a reply arrived)")
            return
        store.save_draft(cid, version, result)
        s["pending"] = False
        log_info("aggregator", f"conv={cid} Draft saved: status={result.get('status')} for {len(texts)} message(s)")
        await _push(cid)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        s["pending"] = False
        log_error("API_ERROR", "aggregator", f"conv={cid} Drafting task failed", e)
        await _push(cid)


def _schedule(cid):
    s = _st(cid)
    s["version"] += 1
    s["pending"] = True
    if s["task"] and not s["task"].done():
        s["task"].cancel()
    s["task"] = asyncio.create_task(_draft_after_quiet(cid, s["version"]))


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
    _schedule(cid)
    await _push(cid)
    return {"ok": True}


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
    customer_text = "\n".join(store.unanswered_customer_texts(cid))
    store.add_message(cid, "agent", text[:MAX_TEXT])
    outcome = store.save_feedback(cid, body.draft_id, customer_text, text[:MAX_TEXT])
    store.mark_answered(cid)
    try:
        await asyncio.to_thread(past_replies.index_pending)
    except Exception as e:
        log_error("RETRIEVAL_ERROR", "past_index", f"conv={cid} Indexing the sent reply failed", e)
    s = _st(cid)
    s["version"] += 1  # any draft still being written is now late
    s["pending"] = False
    if s["task"] and not s["task"].done():
        s["task"].cancel()
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
