"""app/mysql.py
Tiny MySQL helper. The engine writes ONLY to `chat_support_drafts` (audit trail for the Weekly Review).
It never touches `chat`, `chat_message` or `chat_unread_counts`.

Credentials: DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME from the engine environment or .env.
If they are not set, we read them (read-only) from ../rox-chat/.env so secrets are not copied around.

Every function is best-effort: a MySQL problem is logged and swallowed, so it can never block drafting.
"""
import json
import os
from pathlib import Path

import pymysql
from dotenv import dotenv_values

from app.kb_common import log_error, log_info

ROX_ENV = Path(__file__).resolve().parents[2] / "rox-chat" / ".env"
DEFAULT_AGENT_ID = 22222  # support account (same as the column default)
OUTCOMES = {"accepted", "edited", "rewritten", "ignored", "pending"}


def _cfg():
    vals = {}
    if ROX_ENV.exists():
        vals.update(dotenv_values(ROX_ENV))
    vals.update({k: v for k, v in dotenv_values(".env").items() if k.startswith("DB_")})
    vals.update({k: v for k, v in os.environ.items() if k.startswith("DB_")})
    return vals


def _connect():
    c = _cfg()
    host = c.get("DB_HOST", "")
    if "dev" not in host.lower():  # safety: this feature is Dev-only until promoted on purpose
        raise RuntimeError("Refusing to connect: DB_HOST does not look like the dev database")
    return pymysql.connect(
        host=host, port=int(c.get("DB_PORT") or 3306), user=c.get("DB_USER"), password=c.get("DB_PASSWORD"),
        database=c.get("DB_NAME"), charset="utf8mb4", connect_timeout=5, read_timeout=10, write_timeout=10,
        autocommit=True,
    )


def _run(sql, params=()):
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.lastrowid, cur.rowcount
    finally:
        conn.close()


def save_draft_row(chat_id, engine_draft_id, status, draft_text, latest_message_id, used, agent_user_id,
                   customer_user_id=None):
    """Insert one audit row when a draft is generated. Returns the MySQL id, or None on failure.
    agent_user_id is required (no silent default): it is the id the portal says clicked Generate."""
    try:
        new_id, _ = _run(
            "INSERT INTO chat_support_drafts "
            "(chat_id, engine_draft_id, agent_user_id, customer_user_id, status, draft_text, latest_message_id, "
            "used_chunks, outcome) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending')",
            (chat_id, engine_draft_id, agent_user_id, customer_user_id, status, draft_text or "",
             latest_message_id, json.dumps(used or [])),
        )
        log_info("mysql", f"draft row {new_id} saved for engine_draft={engine_draft_id}")
        return new_id
    except Exception as e:
        log_error("MYSQL_ERROR", "save_draft_row", f"engine_draft={engine_draft_id} could not be saved", e)
        return None


def save_outcome(engine_draft_id, outcome, sent_message_id=None):
    """Set the outcome once the agent sends. Matches on engine_draft_id. Returns rows changed (0 if none)."""
    if outcome not in OUTCOMES:
        return 0  # e.g. "no_draft": nothing to record
    try:
        _, n = _run(
            "UPDATE chat_support_drafts SET outcome=%s, sent_message_id=COALESCE(%s, sent_message_id) "
            "WHERE engine_draft_id=%s",
            (outcome, sent_message_id, engine_draft_id),
        )
        return n
    except Exception as e:
        log_error("MYSQL_ERROR", "save_outcome", f"engine_draft={engine_draft_id} outcome not saved", e)
        return 0
