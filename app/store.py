"""SQLite store (temporary home for engine state until it moves to Aurora MySQL).

Two tables only (see rox-chat/docs/architecture.md for why each column exists):
  chat_support_drafts       one row = one Generate + its Send
  support_review_proposals  Weekly Review cards and history

Sent replies are recorded for the weekly review, but are never automatically
used as knowledge for a later customer draft.
"""
import difflib
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app.kb_common import KB_DIR, log_error, log_info

DB_PATH = os.getenv("SQLITE_PATH") or "./data/app.db"
EDITED_MIN_SIMILARITY = 0.5  # final reply this similar to the draft = "edited", below = "rewritten"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_support_drafts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  agent_user_id INTEGER NOT NULL,
  customer_user_id INTEGER,
  status TEXT NOT NULL,
  draft_text TEXT NOT NULL DEFAULT '',
  final_text TEXT,
  customer_text TEXT,
  outcome TEXT NOT NULL DEFAULT 'pending',
  latest_message_id INTEGER,
  sent_message_id INTEGER,
  used_chunks TEXT,
  sources TEXT,
  query_json TEXT,
  error TEXT,
  tokens_in INTEGER,
  tokens_out INTEGER,
  ms INTEGER,
  llm_model TEXT,
  kb_version TEXT,
  review_state INTEGER NOT NULL DEFAULT 0,
  review_status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL,
  sent_at TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS ux_drafts_chat_msg ON chat_support_drafts (chat_id, latest_message_id);
CREATE INDEX IF NOT EXISTS ix_drafts_outcome ON chat_support_drafts (outcome);
CREATE INDEX IF NOT EXISTS ix_drafts_review ON chat_support_drafts (review_state, review_status);
CREATE TABLE IF NOT EXISTS support_review_proposals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  status TEXT NOT NULL DEFAULT 'pending',
  action TEXT NOT NULL,
  chunk_id TEXT, part TEXT, title TEXT, audience TEXT, body TEXT, old_text TEXT,
  keywords TEXT, draft_ids TEXT, examples TEXT, extra TEXT,
  created_at TEXT NOT NULL,
  decided_by INTEGER,
  decided_at TEXT);
"""


class DraftExists(Exception):
    """This customer message already has a draft. There is no regenerate: the agent edits the draft."""


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
    log_info("init_db", f"SQLite ready at {DB_PATH}")


def kb_version():
    """Short fingerprint of all KB files, so a draft row says which KB produced it."""
    try:
        h = hashlib.sha1()
        for f in sorted(KB_DIR.glob("*.md")):
            h.update(f.name.encode() + f.read_bytes())
        return h.hexdigest()[:10]
    except OSError:
        return None


def bound_context(items, max_messages=4, max_chars=1000):
    """Pure rule (no DB): items = [{sender, text, ...}] oldest first. Keep the last agent exchange plus
    all newer customer messages, trimmed to max_messages / max_chars."""
    items = list(items)
    if not items:
        return []
    last_agent = max((n for n, m in enumerate(items) if m["sender"] == "agent"), default=-1)
    # Include the last two exchanges before the answer, then all newer customer messages.
    window = items[max(0, last_agent - 3):] if last_agent >= 0 else items
    while window and (len(window) > max_messages or sum(len(m["text"]) for m in window) > max_chars):
        window.pop(0)
    return window


def save_draft(chat_id, result, latest_message_id, agent_user_id, customer_user_id=None, llm_model=None):
    """Insert the draft row (outcome 'pending'). Raises DraftExists if this chat message already has a draft."""
    tokens = result.get("tokens") or {}
    vals = (result.get("status", "ERROR"), result.get("draft", ""),
            json.dumps(result.get("used") or []), json.dumps(result.get("sources") or []),
            json.dumps(result.get("query") or {}, ensure_ascii=False), result.get("error"),
            tokens.get("in"), tokens.get("out"), result.get("ms"), llm_model, kb_version())
    with _db() as c:
        try:
            cur = c.execute(
                "INSERT INTO chat_support_drafts (chat_id, agent_user_id, customer_user_id, latest_message_id, "
                "status, draft_text, used_chunks, sources, query_json, error, tokens_in, tokens_out, ms, "
                "llm_model, kb_version, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (chat_id, agent_user_id, customer_user_id, latest_message_id) + vals + (_now(),))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            raise DraftExists()  # one draft per customer message; the agent edits it instead of regenerating


def message_has_draft(chat_id, latest_message_id):
    with _db() as c:
        return c.execute("SELECT 1 FROM chat_support_drafts WHERE chat_id=? AND latest_message_id=?",
                         (chat_id, latest_message_id)).fetchone() is not None


def draft_exists(draft_id):
    with _db() as c:
        return c.execute("SELECT 1 FROM chat_support_drafts WHERE id=?", (draft_id,)).fetchone() is not None


def save_feedback(draft_id, customer_text, final_text, sent_message_id=None):
    """Compares the sent reply with the draft the agent saw and stores it on the same row. Returns the outcome."""
    with _db() as c:
        r = c.execute("SELECT status, draft_text, outcome, chat_id FROM chat_support_drafts WHERE id=?",
                      (draft_id,)).fetchone()
        if not r:
            raise KeyError(draft_id)
        if r["outcome"] != "pending":  # idempotent: a retry returns the first result and writes nothing
            log_info("feedback", f"draft={draft_id} repeat ignored")
            return r["outcome"]
        draft_text = (r["draft_text"] or "") if r["status"] != "ERROR" else ""
        if not draft_text.strip():
            outcome = "no_draft"
        elif final_text.strip() == draft_text.strip():
            outcome = "accepted"
        elif difflib.SequenceMatcher(None, draft_text.strip(), final_text.strip()).ratio() >= EDITED_MIN_SIMILARITY:
            outcome = "edited"
        else:
            outcome = "rewritten"
        c.execute("UPDATE chat_support_drafts SET final_text=?, customer_text=?, outcome=?, sent_message_id=?, "
                  "sent_at=? WHERE id=?", (final_text, customer_text, outcome, sent_message_id, _now(), draft_id))
    log_info("feedback", f"draft={draft_id} outcome={outcome}")
    return outcome


def pending_feedback(limit=200):
    """Sent drafts not yet prepared as private weekly-review candidates."""
    with _db() as c:
        rows = c.execute(
            "SELECT id, chat_id, customer_text, draft_text, final_text, outcome, sent_at, status, query_json "
            "FROM chat_support_drafts WHERE sent_at IS NOT NULL AND review_state=0 ORDER BY id LIMIT ?",
            (limit,)).fetchall()
    out = []
    for r in rows:
        out.append({"id": r["id"], "conv_id": f"chat-{r['chat_id']}", "customer_text": r["customer_text"],
                    "draft_text": r["draft_text"], "final_text": r["final_text"], "outcome": r["outcome"],
                    "created_at": r["sent_at"], "draft_status": r["status"],
                    "draft_query": json.loads(r["query_json"] or "{}")})
    return out


def mark_feedback_indexed(ids, value):
    """value: 1 = prepared for weekly review, 2 = skipped as non-actionable."""
    if not ids:
        return
    with _db() as c:
        c.executemany("UPDATE chat_support_drafts SET review_state=? WHERE id=?", [(value, i) for i in ids])


def reset_feedback_index():
    with _db() as c:
        c.execute("UPDATE chat_support_drafts SET review_state=0")


def _nums(ids):
    return [int(i[1:]) for i in ids if i[1:].isdigit()]


def active_feedback_ids(ids):
    """ids look like 'f12' (f + draft id). Returns the subset whose review_status is still 'active'."""
    nums = _nums(ids)
    if not nums:
        return set()
    with _db() as c:
        rows = c.execute("SELECT id FROM chat_support_drafts WHERE review_status='active' AND id IN "
                         f"({','.join('?' * len(nums))})", nums).fetchall()
    return {f"f{r['id']}" for r in rows}


def review_candidate_ids():
    """Private review candidates that are still waiting for a reviewer decision."""
    with _db() as c:
        rows = c.execute("SELECT id FROM chat_support_drafts WHERE review_state=1 AND review_status='active' "
                         "ORDER BY id").fetchall()
    return [f"f{r['id']}" for r in rows]


def feedback_details(ids):
    """ids like 'f12'. Returns {'f12': {id, customer_text, final_text, outcome, draft_status, draft_used,
    draft_feature}}. draft_used = ids the draft relied on (KB points like 'B5' or past replies like 'f3')."""
    nums = _nums(ids)
    if not nums:
        return {}
    with _db() as c:
        rows = c.execute(
            "SELECT id, customer_text, final_text, outcome, status AS draft_status, used_chunks, query_json "
            f"FROM chat_support_drafts WHERE id IN ({','.join('?' * len(nums))})", nums).fetchall()
    out = {}
    for r in rows:
        x = dict(r)
        x["draft_used"] = json.loads(x.pop("used_chunks") or "[]")
        x["draft_feature"] = json.loads(x.pop("query_json") or "{}").get("feature", "")
        out[f"f{x['id']}"] = x
    return out


def set_feedback_status(ids, status):
    """status: active | merged (now in the KB) | rejected (reviewer said no)."""
    with _db() as c:
        c.executemany("UPDATE chat_support_drafts SET review_status=? WHERE id=?", [(status, n) for n in _nums(ids)])


def clear_pending_proposals():
    with _db() as c:
        c.execute("DELETE FROM support_review_proposals WHERE status='pending'")


def add_proposal(p):
    with _db() as c:
        cur = c.execute(
            "INSERT INTO support_review_proposals (action, chunk_id, part, title, audience, body, old_text, "
            "keywords, draft_ids, examples, extra, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (p["action"], p["chunk_id"], p["part"], p["title"], p["audience"], p["body"], p["old_text"],
             json.dumps(p["keywords"]), json.dumps(p["feedback_ids"]),
             json.dumps(p["examples"], ensure_ascii=False), json.dumps(p.get("extra") or {}, ensure_ascii=False),
             _now()))
        return cur.lastrowid


def _proposal(r):
    d = dict(r)
    for k in ("keywords", "draft_ids", "examples"):
        d[k] = json.loads(d[k] or "[]")
    d["feedback_ids"] = d["draft_ids"]  # old name, kept so the review page keeps working
    d["extra"] = json.loads(d.get("extra") or "{}")
    return d


def list_proposals(status="pending"):
    with _db() as c:
        rows = c.execute("SELECT * FROM support_review_proposals WHERE status=? ORDER BY id", (status,)).fetchall()
    return [_proposal(r) for r in rows]


def get_proposal(pid):
    with _db() as c:
        r = c.execute("SELECT * FROM support_review_proposals WHERE id=?", (pid,)).fetchone()
    return _proposal(r) if r else None


def set_proposal_status(pid, status, title=None, audience=None, body=None, action=None, chunk_id=None,
                        decided_by=None):
    """Keeps the final approved text, what was done (replace/add and which chunk), and who decided and when."""
    with _db() as c:
        if body is None:
            c.execute("UPDATE support_review_proposals SET status=?, decided_by=?, decided_at=? WHERE id=?",
                      (status, decided_by, _now(), pid))
        else:
            c.execute("UPDATE support_review_proposals SET status=?, title=?, audience=?, body=?, action=?, "
                      "chunk_id=?, decided_by=?, decided_at=? WHERE id=?",
                      (status, title, audience, body, action, chunk_id, decided_by, _now(), pid))
