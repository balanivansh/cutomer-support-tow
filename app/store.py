"""SQLite store (temporary home for engine state until it moves to Aurora MySQL).

Two tables only (see rox-chat/docs/architecture.md for why each column exists):
  chat_support_drafts       one row = one Generate + its Send
  support_review_proposals  Weekly Review cards and history

Sent replies are recorded for the weekly review, but are never automatically
used as knowledge for a later customer draft.
"""
import difflib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app.kb_common import log_error, log_info

DB_PATH = os.getenv("SQLITE_PATH") or "./data/app.db"
EDITED_MIN_SIMILARITY = 0.5  # final reply this similar to the draft = "edited", below = "rewritten"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_support_drafts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  agent_user_id INTEGER NOT NULL,
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
  tokens_in INTEGER,
  tokens_out INTEGER,
  ms INTEGER,
  review_state INTEGER NOT NULL DEFAULT 0,
  review_status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS ux_drafts_chat_msg ON chat_support_drafts (chat_id, latest_message_id);
CREATE INDEX IF NOT EXISTS ix_drafts_outcome ON chat_support_drafts (outcome);
CREATE INDEX IF NOT EXISTS ix_drafts_review ON chat_support_drafts (review_state, review_status);
CREATE TABLE IF NOT EXISTS support_review_proposals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  status TEXT NOT NULL DEFAULT 'pending',
  action TEXT NOT NULL,
  chunk_id TEXT, part TEXT, title TEXT, audience TEXT, body TEXT,
  keywords TEXT, draft_ids TEXT, extra TEXT,
  created_at TEXT NOT NULL);
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


def _shorten(text, limit=1000, head=600, tail=400):
    """Long message: keep the start (context) and the end (usually the ask). Returns (text, was_cut)."""
    if len(text) <= limit:
        return text, False
    return text[:head] + " [...] " + text[-tail:], True


def bound_context(items, max_unanswered=6, max_chars=3000, max_msg_chars=1000):
    """Pure rule (no DB): items = [{sender, text, ...}] oldest first.
    Keep (1) all customer messages after the last agent reply, newest 6 at most, then
    (2) the last agent reply and the customer message it answered. A message over max_msg_chars keeps its
    first 600 and last 400 characters (marked cut=True), never dropped. Over max_chars: drop the OLDEST whole
    messages, but always keep the newest customer message."""
    out = []
    for m in items:
        text, cut = _shorten(m["text"], max_msg_chars)
        out.append({**m, "text": text, "cut": cut or m.get("cut", False)})
    items = out
    if not items:
        return []
    last_agent = max((n for n, m in enumerate(items) if m["sender"] == "agent"), default=-1)
    unanswered = [m for m in items[last_agent + 1:] if m["sender"] == "customer"][-max_unanswered:]
    pair = []
    if last_agent >= 0:
        answered = max((n for n in range(last_agent) if items[n]["sender"] == "customer"), default=-1)
        pair = ([items[answered]] if answered >= 0 else []) + [items[last_agent]]
    window = pair + unanswered
    while len(window) > 1 and sum(len(m["text"]) for m in window) > max_chars:
        window.pop(0)
    return window


def save_draft(chat_id, result, latest_message_id, agent_user_id):
    """Insert the draft row (outcome 'pending'). Raises DraftExists if this chat message already has a draft."""
    tokens = result.get("tokens") or {}
    vals = (result.get("status", "ERROR"), result.get("draft", ""),
            json.dumps(result.get("used") or []), json.dumps(result.get("sources") or []),
            json.dumps(result.get("query") or {}, ensure_ascii=False),
            tokens.get("in"), tokens.get("out"), result.get("ms"))
    with _db() as c:
        try:
            cur = c.execute(
                "INSERT INTO chat_support_drafts (chat_id, agent_user_id, latest_message_id, "
                "status, draft_text, used_chunks, sources, query_json, tokens_in, tokens_out, ms, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (chat_id, agent_user_id, latest_message_id) + vals + (_now(),))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            raise DraftExists()  # one draft per customer message; the agent edits it instead of regenerating


def message_has_draft(chat_id, latest_message_id):
    with _db() as c:
        return c.execute("SELECT 1 FROM chat_support_drafts WHERE chat_id=? AND latest_message_id=?",
                         (chat_id, latest_message_id)).fetchone() is not None


def get_message_draft(chat_id, latest_message_id):
    """The saved draft row for this customer message, or None."""
    with _db() as c:
        r = c.execute("SELECT id, status, draft_text, used_chunks, ms, outcome FROM chat_support_drafts "
                      "WHERE chat_id=? AND latest_message_id=?", (chat_id, latest_message_id)).fetchone()
        return dict(r) if r else None


def delete_failed_draft(draft_id):
    """A draft that failed (status ERROR, nothing sent) is useless: remove it so the agent can generate again."""
    with _db() as c:
        c.execute("DELETE FROM chat_support_drafts WHERE id=? AND status='ERROR' AND outcome='pending'", (draft_id,))


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
        c.execute("UPDATE chat_support_drafts SET final_text=?, customer_text=?, outcome=?, sent_message_id=? "
                  "WHERE id=?", (final_text, customer_text, outcome, sent_message_id, draft_id))
    log_info("feedback", f"draft={draft_id} outcome={outcome}")
    return outcome


def pending_feedback(limit=200):
    """Sent drafts not yet prepared as private weekly-review candidates."""
    with _db() as c:
        rows = c.execute(
            "SELECT id, chat_id, customer_text, draft_text, final_text, outcome, created_at, status, query_json "
            "FROM chat_support_drafts WHERE outcome != 'pending' AND review_state=0 ORDER BY id LIMIT ?",
            (limit,)).fetchall()
    out = []
    for r in rows:
        out.append({"id": r["id"], "conv_id": f"chat-{r['chat_id']}", "customer_text": r["customer_text"],
                    "draft_text": r["draft_text"], "final_text": r["final_text"], "outcome": r["outcome"],
                    "created_at": r["created_at"], "draft_status": r["status"],
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
            "SELECT id, customer_text, draft_text, final_text, outcome, status AS draft_status, used_chunks, query_json "
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
    """Generate report replaces the auto cards that are still pending. Manual cards and staged cards stay."""
    with _db() as c:
        c.execute("DELETE FROM support_review_proposals WHERE status='pending' "
                  "AND COALESCE(extra, '') NOT LIKE '%\"manual\": true%'")


def add_manual_proposal():
    """A blank card the reviewer fills in by hand (+ Add card). No agent replies behind it."""
    return add_proposal({"action": "add", "chunk_id": "", "part": "", "title": "", "audience": "Everyone",
                         "body": "", "keywords": [], "feedback_ids": [],
                         "extra": {"manual": True, "feature_id": "", "stored": None}})


def stage_proposal(pid, stage, title, audience, body, keywords):
    """Reviewer pressed Save card: keep the final fields and what to do. The KB is NOT touched until Update KB."""
    with _db() as c:
        r = c.execute("SELECT extra FROM support_review_proposals WHERE id=? AND status IN ('pending','staged')",
                      (pid,)).fetchone()
        if not r:
            raise KeyError(pid)
        extra = json.loads(r["extra"] or "{}")
        extra["stage"] = stage
        c.execute("UPDATE support_review_proposals SET status='staged', title=?, audience=?, body=?, keywords=?, "
                  "extra=? WHERE id=?", (title, audience, body, json.dumps(keywords), json.dumps(extra, ensure_ascii=False), pid))


def unstage_proposal(pid):
    with _db() as c:
        r = c.execute("SELECT extra FROM support_review_proposals WHERE id=? AND status='staged'", (pid,)).fetchone()
        if not r:
            raise KeyError(pid)
        extra = json.loads(r["extra"] or "{}")
        extra.pop("stage", None)
        c.execute("UPDATE support_review_proposals SET status='pending', extra=? WHERE id=?",
                  (json.dumps(extra, ensure_ascii=False), pid))


def add_proposal(p):
    with _db() as c:
        cur = c.execute(
            "INSERT INTO support_review_proposals (action, chunk_id, part, title, audience, body, "
            "keywords, draft_ids, extra, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (p["action"], p["chunk_id"], p["part"], p["title"], p["audience"], p["body"],
             json.dumps(p["keywords"]), json.dumps(p["feedback_ids"]),
             json.dumps(p.get("extra") or {}, ensure_ascii=False), _now()))
        return cur.lastrowid


def _proposal(r):
    d = dict(r)
    for k in ("keywords", "draft_ids"):
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


def set_proposal_status(pid, status, title=None, audience=None, body=None, action=None, chunk_id=None):
    """Keeps the final approved text and what was done (replace/add and which chunk)."""
    with _db() as c:
        if body is None:
            c.execute("UPDATE support_review_proposals SET status=? WHERE id=?", (status, pid))
        else:
            c.execute("UPDATE support_review_proposals SET status=?, title=?, audience=?, body=?, action=?, "
                      "chunk_id=? WHERE id=?", (status, title, audience, body, action, chunk_id, pid))
