"""SQLite store: conversations, messages, drafts and agent feedback.
Feedback rows start with indexed=0; stage 4b adds the sent replies to the past_replies index."""
import difflib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from app.kb_common import log_error, log_info

DB_PATH = os.getenv("SQLITE_PATH") or "./data/app.db"
EDITED_MIN_SIMILARITY = 0.5  # final reply this similar to the draft = "edited", below = "rewritten"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conv_id TEXT NOT NULL, sender TEXT NOT NULL,
  text TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS drafts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conv_id TEXT NOT NULL, version INTEGER NOT NULL,
  status TEXT NOT NULL, draft TEXT NOT NULL, used TEXT, sources TEXT, query TEXT, tokens TEXT,
  ms INTEGER, error TEXT, answered INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conv_id TEXT NOT NULL, draft_id INTEGER,
  customer_text TEXT, draft_text TEXT, final_text TEXT NOT NULL, outcome TEXT NOT NULL,
  indexed INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS proposals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL DEFAULT 'pending', action TEXT NOT NULL,
  chunk_id TEXT, part TEXT, title TEXT, audience TEXT, body TEXT, old_text TEXT, reason TEXT,
  keywords TEXT, feedback_ids TEXT, examples TEXT, candidates TEXT, created_at TEXT NOT NULL);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        log_error("DB_ERROR", "sqlite", f"Database operation failed ({DB_PATH})", e)
        raise
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    with _db() as c:
        c.executescript(SCHEMA)
        try:  # older databases: add the status column (active / merged / rejected)
            c.execute("ALTER TABLE feedback ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        except sqlite3.OperationalError:
            pass
        try:  # older databases: proposals get a candidates column
            c.execute("ALTER TABLE proposals ADD COLUMN candidates TEXT")
        except sqlite3.OperationalError:
            pass
    log_info("init_db", f"SQLite ready at {DB_PATH}")


def create_conversation():
    cid = uuid.uuid4().hex[:8]
    with _db() as c:
        c.execute("INSERT INTO conversations (id, created_at) VALUES (?, ?)", (cid, _now()))
    return cid


def conversation_exists(cid):
    with _db() as c:
        return c.execute("SELECT 1 FROM conversations WHERE id=?", (cid,)).fetchone() is not None


def add_message(cid, sender, text):
    with _db() as c:
        cur = c.execute("INSERT INTO messages (conv_id, sender, text, created_at) VALUES (?,?,?,?)",
                        (cid, sender, text, _now()))
        return cur.lastrowid


def list_messages(cid):
    with _db() as c:
        rows = c.execute("SELECT id, sender, text, created_at FROM messages WHERE conv_id=? ORDER BY id",
                         (cid,)).fetchall()
    return [dict(r) for r in rows]


def unanswered_customer_texts(cid):
    """Customer messages sent after the agent's last reply, oldest first."""
    with _db() as c:
        last_agent = c.execute("SELECT COALESCE(MAX(id), 0) FROM messages WHERE conv_id=? AND sender='agent'",
                               (cid,)).fetchone()[0]
        rows = c.execute("SELECT text FROM messages WHERE conv_id=? AND sender='customer' AND id>? ORDER BY id",
                         (cid, last_agent)).fetchall()
    return [r["text"] for r in rows]


def save_draft(cid, version, result):
    with _db() as c:
        cur = c.execute(
            "INSERT INTO drafts (conv_id, version, status, draft, used, sources, query, tokens, ms, error, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cid, version, result.get("status", "ERROR"), result.get("draft", ""),
             json.dumps(result.get("used") or []), json.dumps(result.get("sources") or []),
             json.dumps(result.get("query") or {}), json.dumps(result.get("tokens") or {}),
             result.get("ms"), result.get("error"), _now()))
        return cur.lastrowid


def latest_draft(cid):
    """Newest draft that has not been answered yet, or None."""
    with _db() as c:
        r = c.execute("SELECT * FROM drafts WHERE conv_id=? AND answered=0 ORDER BY id DESC LIMIT 1",
                      (cid,)).fetchone()
    if not r:
        return None
    return {"id": r["id"], "status": r["status"], "draft": r["draft"], "ms": r["ms"], "error": r["error"],
            "used": json.loads(r["used"] or "[]"), "sources": json.loads(r["sources"] or "[]"),
            "query": json.loads(r["query"] or "{}"), "tokens": json.loads(r["tokens"] or "{}")}


def mark_answered(cid):
    with _db() as c:
        c.execute("UPDATE drafts SET answered=1 WHERE conv_id=?", (cid,))


def save_feedback(cid, draft_id, customer_text, final_text):
    """Compares the sent reply with the draft the agent saw. Returns the outcome label."""
    draft_text = ""
    if draft_id:
        with _db() as c:
            r = c.execute("SELECT draft, status FROM drafts WHERE id=? AND conv_id=?", (draft_id, cid)).fetchone()
        if r and r["status"] != "ERROR":
            draft_text = r["draft"] or ""
    if not draft_text.strip():
        outcome = "no_draft"
    elif final_text.strip() == draft_text.strip():
        outcome = "accepted"
    elif difflib.SequenceMatcher(None, draft_text.strip(), final_text.strip()).ratio() >= EDITED_MIN_SIMILARITY:
        outcome = "edited"
    else:
        outcome = "rewritten"
    with _db() as c:
        c.execute("INSERT INTO feedback (conv_id, draft_id, customer_text, draft_text, final_text, outcome, created_at)"
                  " VALUES (?,?,?,?,?,?,?)",
                  (cid, draft_id, customer_text, draft_text, final_text, outcome, _now()))
    log_info("feedback", f"conv={cid} outcome={outcome}")
    return outcome


def pending_feedback(limit=200):
    """Feedback rows not yet indexed (indexed=0), with the draft's status and query."""
    with _db() as c:
        rows = c.execute(
            "SELECT f.id, f.conv_id, f.customer_text, f.draft_text, f.final_text, f.outcome, f.created_at, "
            "d.status AS draft_status, d.query AS draft_query FROM feedback f "
            "LEFT JOIN drafts d ON d.id = f.draft_id WHERE f.indexed=0 ORDER BY f.id LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        x = dict(r)
        x["draft_query"] = json.loads(x["draft_query"] or "{}")
        out.append(x)
    return out


def mark_feedback_indexed(ids, value):
    """value: 1 = indexed into past_replies, 2 = skipped (not useful as a past reply)."""
    if not ids:
        return
    with _db() as c:
        c.executemany("UPDATE feedback SET indexed=? WHERE id=?", [(value, i) for i in ids])


def reset_feedback_index():
    with _db() as c:
        c.execute("UPDATE feedback SET indexed=0")


def _nums(ids):
    return [int(i[1:]) for i in ids if i[1:].isdigit()]


def active_feedback_ids(ids):
    """ids look like 'f12'. Returns the subset whose feedback row is still status 'active'."""
    nums = _nums(ids)
    if not nums:
        return set()
    with _db() as c:
        rows = c.execute(f"SELECT id FROM feedback WHERE status='active' AND id IN ({','.join('?' * len(nums))})",
                         nums).fetchall()
    return {f"f{r['id']}" for r in rows}


def review_candidate_ids():
    """Past replies in the index that are still active and waiting for the weekly review."""
    with _db() as c:
        rows = c.execute("SELECT id FROM feedback WHERE indexed=1 AND status='active' ORDER BY id").fetchall()
    return [f"f{r['id']}" for r in rows]


def set_feedback_status(ids, status):
    """status: active | merged (now in the KB) | rejected (reviewer said no)."""
    with _db() as c:
        c.executemany("UPDATE feedback SET status=? WHERE id=?", [(status, n) for n in _nums(ids)])


def clear_pending_proposals():
    with _db() as c:
        c.execute("DELETE FROM proposals WHERE status='pending'")


def add_proposal(p):
    with _db() as c:
        cur = c.execute(
            "INSERT INTO proposals (action, chunk_id, part, title, audience, body, old_text, reason, keywords,"
            " feedback_ids, examples, candidates, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p["action"], p["chunk_id"], p["part"], p["title"], p["audience"], p["body"], p["old_text"],
             p["reason"], json.dumps(p["keywords"]), json.dumps(p["feedback_ids"]),
             json.dumps(p["examples"], ensure_ascii=False), json.dumps(p["candidates"], ensure_ascii=False), _now()))
        return cur.lastrowid


def _proposal(r):
    d = dict(r)
    for k in ("keywords", "feedback_ids", "examples", "candidates"):
        d[k] = json.loads(d[k] or "[]")
    return d


def list_proposals(status="pending"):
    with _db() as c:
        rows = c.execute("SELECT * FROM proposals WHERE status=? ORDER BY id", (status,)).fetchall()
    return [_proposal(r) for r in rows]


def get_proposal(pid):
    with _db() as c:
        r = c.execute("SELECT * FROM proposals WHERE id=?", (pid,)).fetchone()
    return _proposal(r) if r else None


def set_proposal_status(pid, status, title=None, audience=None, body=None, action=None, chunk_id=None):
    """Keeps the final approved text and what was done (replace/add and which chunk): a history of KB changes."""
    with _db() as c:
        if body is None:
            c.execute("UPDATE proposals SET status=? WHERE id=?", (status, pid))
        else:
            c.execute("UPDATE proposals SET status=?, title=?, audience=?, body=?, action=?, chunk_id=? WHERE id=?",
                      (status, title, audience, body, action, chunk_id, pid))
