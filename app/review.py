"""Weekly review. Active past replies are grouped by topic (local embeddings, no LLM). For each topic the
LLM suggests one KB change. The reviewer sees the matching KB chunks and chooses:
  Replace this Q&A (chunk rewritten in place), Add as new point, or Reject.
Replace/Add: KB file is backed up and changed, KB is re-indexed, the replies stop being used (status 'merged').
Reject: the replies stop being used (status 'rejected')."""
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime

import numpy as np

from app import keywords, past_replies, store
from app.kb_common import KB_PATH, ROOT, log_error, log_info
from app.llm import DRAFT_MODEL, LLMError, chat_json
from app.pipeline import get_retriever

CLUSTER_SIM = 0.80  # English queries at least this similar = same topic
MAX_TOPICS = 15
MAX_EXAMPLES = 6
HEAD = re.compile(r"^### ([A-Z]\d+)\. (.*)$")
_lock = threading.Lock()

REVIEW_PROMPT = """You maintain a RoxStar support KB of Q&A chunks. You get KB chunks and past replies that human support agents approved for ONE topic. The agents are right: where they differ from the KB, the KB is outdated.
Decide:
- "replace": one KB chunk covers this topic. Return its chunk_id, its title (the question; keep or improve it) and the full new body. Rewrite the whole body so it stays consistent: fix or remove outdated lines, do not just append a line. If the KB already says the same, return the chunk unchanged.
- "add": no KB chunk covers this topic. Return a new title (a question), audience, body and the part letter (B broadcaster topics, C participant topics, A overview, D leaderboards).
Rules: plain English facts, short, same style as the KB, no promises, never invent facts beyond the replies. audience is one of: Everyone, Broadcaster, Participant, "Broadcaster, Participant". body = the text after the Audience line, without heading.
Also list keywords: short RoxStar words customers use for this topic that are not in the keyword list (max 5, may be empty).
JSON only: {"action": "replace|add", "chunk_id": "", "part": "", "title": "", "audience": "", "body": "", "reason": "one short sentence", "keywords": []}"""


# ---------- KB markdown helpers ----------
def _boundary(line):
    return line.startswith("### ") or line.startswith("## ") or line.strip() == "---"


def _find_chunk(lines, cid):
    for n, line in enumerate(lines):
        m = HEAD.match(line)
        if m and m.group(1) == cid:
            e = n + 1
            while e < len(lines) and not _boundary(lines[e]):
                e += 1
            return n, e
    raise ValueError(f"Chunk {cid} not found in the KB file")


def chunk_info(lines, cid):
    s, e = _find_chunk(lines, cid)
    aud = lines[s + 1].replace("**Audience:**", "").strip() if s + 1 < e else ""
    return {"id": cid, "title": HEAD.match(lines[s]).group(2), "audience": aud,
            "body": "\n".join(lines[s + 2:e]).strip(), "text": "\n".join(lines[s:e]).strip()}


def edit_chunk(lines, cid, title, audience, body):
    """Replaces one chunk in place: same ID, new title (question), audience and body."""
    s, e = _find_chunk(lines, cid)
    head = f"### {cid}. {title.strip()}" if title.strip() else lines[s]
    aud = f"**Audience:** {audience.strip()}" if audience.strip() else lines[s + 1]
    return lines[:s] + [head, aud] + body.strip().splitlines() + [""] + lines[e:]


def add_chunk(lines, part, title, audience, body):
    part = part.strip().upper()[:1]
    nums = [int(m.group(1)[1:]) for m in (HEAD.match(l) for l in lines) if m and m.group(1)[0] == part]
    new_id = f"{part}{max(nums, default=0) + 1}"
    start = next((n for n, l in enumerate(lines) if l.startswith(f"## PART {part}")), None)
    if start is None:
        raise ValueError(f"Part {part} not found in the KB file")
    end = next((n for n in range(start + 1, len(lines)) if lines[n].strip() == "---"), len(lines))
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    block = ["", f"### {new_id}. {title.strip()}", f"**Audience:** {audience.strip() or 'Everyone'}"] \
        + body.strip().splitlines()
    return lines[:end] + block + lines[end:], new_id


def _kb_lines():
    return KB_PATH.read_text(encoding="utf-8").splitlines()


# ---------- build the weekly report ----------
def _clusters():
    ids = store.review_candidate_ids()
    if not ids:
        return []
    data = past_replies._collection().get(ids=ids, include=["embeddings", "documents", "metadatas"])
    items = [{"id": i, "vec": np.array(v), "reply": d, "q": m.get("question", ""), "eq": m.get("english_query", "")}
             for i, v, d, m in zip(data["ids"], data["embeddings"], data["documents"], data["metadatas"])]
    clusters = []
    for it in items:
        for c in clusters:
            if float(np.dot(c[0]["vec"], it["vec"])) >= CLUSTER_SIM:
                c.append(it)
                break
        else:
            clusters.append([it])
    clusters.sort(key=len, reverse=True)
    return clusters[:MAX_TOPICS]


def _propose(cluster):
    lines = _kb_lines()
    cands = []
    for c in get_retriever().search(cluster[0]["eq"] or cluster[0]["q"], role=None, top_k=3):
        try:
            cands.append(chunk_info(lines, c["id"]))
        except ValueError:
            pass
    examples = cluster[:MAX_EXAMPLES]
    user = ("KEYWORD LIST: " + ", ".join(keywords.load()) + "\n\nKB CHUNKS:\n" + "\n\n".join(c["text"] for c in cands)
            + "\n\nPAST REPLIES:\n" + "\n".join(f"Q: {e['eq'] or e['q']}\nA: {e['reply']}" for e in examples))
    data, _, _ = chat_json([{"role": "system", "content": REVIEW_PROMPT}, {"role": "user", "content": user}],
                           DRAFT_MODEL, 1800, "review")
    chosen = next((c for c in cands if c["id"] == (data.get("chunk_id") or "").strip()), None)
    if data.get("action") == "replace" and chosen:
        action, cid = "replace", chosen["id"]
        title = data.get("title") or chosen["title"]
        audience = data.get("audience") or chosen["audience"]
        body = data.get("body") or chosen["body"]
        part = cid[0]
    else:  # no close chunk: suggest a new point
        action, cid = "add", ""
        title, audience, body = data.get("title") or "", data.get("audience") or "", data.get("body") or ""
        part = (data.get("part") or (cands[0]["id"][0] if cands else "B")).strip().upper()[:1]
    return {"action": action, "chunk_id": cid, "part": part, "title": title, "audience": audience, "body": body,
            "old_text": "", "reason": data.get("reason") or "",
            "keywords": [str(k) for k in (data.get("keywords") or [])][:5],
            "feedback_ids": [e["id"] for e in cluster],
            "examples": [{"q": e["eq"] or e["q"], "reply": e["reply"]} for e in cluster[:3]],
            "candidates": [{k: c[k] for k in ("id", "title", "audience", "body")} for c in cands]}


def build_report():
    """Replaces pending proposals with fresh ones. Returns how many topics were proposed."""
    with _lock:
        store.clear_pending_proposals()
        count = 0
        for cluster in _clusters():
            try:
                store.add_proposal(_propose(cluster))
                count += 1
            except LLMError as e:
                log_error("LLM_ERROR", "review", f"Could not make a proposal for topic {cluster[0]['id']}: {e}")
        log_info("review", f"Weekly report ready: {count} topic(s)")
        return count


# ---------- apply / reject ----------
def _reingest():
    r = subprocess.run([sys.executable, "-m", "app.kb_ingest"], cwd=str(ROOT), capture_output=True,
                       text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError("KB re-index failed: " + (r.stderr or r.stdout)[-300:])


def apply_proposal(pid, mode, chunk_id, part, title, audience, body, keywords_text):
    """mode 'replace' rewrites chunk_id in place; mode 'add' adds a new chunk at the end of its part."""
    p = store.get_proposal(pid)
    if not p or p["status"] != "pending":
        raise ValueError("Proposal not found or already handled")
    body, chunk_id = body.strip(), chunk_id.strip()
    if not body:
        raise ValueError("The text is empty")
    if mode == "replace" and not chunk_id:
        raise ValueError("Pick the KB chunk to replace")
    if mode == "add" and not title.strip():
        raise ValueError("A new point needs a title")
    if mode not in ("replace", "add"):
        raise ValueError("Unknown mode")
    with _lock:
        lines = _kb_lines()
        if mode == "replace":
            new_lines, target = edit_chunk(lines, chunk_id, title, audience, body), chunk_id
        else:
            new_lines, target = add_chunk(lines, part or "B", title, audience, body)
        backup_dir = KB_PATH.parent / "kb_backups"
        backup_dir.mkdir(exist_ok=True)
        backup = backup_dir / f"kb_{datetime.now():%Y%m%d_%H%M%S}.md"
        shutil.copy2(KB_PATH, backup)
        KB_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        try:
            _reingest()
        except Exception:
            shutil.copy2(backup, KB_PATH)  # put the old KB back
            raise
        import app.pipeline as pipeline_mod
        pipeline_mod._retriever = None  # next search opens the re-indexed KB
        get_retriever()
        log_info("review", f"Proposal {pid}: {mode} {target} applied, backup {backup.name}")
        store.set_feedback_status(p["feedback_ids"], "merged")
        keywords.add(keywords_text.split(","))
        store.set_proposal_status(pid, "approved", title, audience, body, mode, target)


def reject_proposal(pid):
    p = store.get_proposal(pid)
    if not p or p["status"] != "pending":
        raise ValueError("Proposal not found or already handled")
    store.set_feedback_status(p["feedback_ids"], "rejected")
    store.set_proposal_status(pid, "rejected")
    log_info("review", f"Proposal {pid} rejected, {len(p['feedback_ids'])} repl(ies) marked rejected")
