"""Weekly review. Private reply candidates are grouped by topic (local embeddings). For each topic ONE small LLM call
writes: the English question, the main solution (short summary if long), the topic and search words.
The reviewer picks the feature, topic and audience, edits the text and presses:
  Replace this Q&A (the stored point is overwritten), Add new (new point at the end of the topic) or Reject.
Replace/Add: the answer is translated to English, the KB file is backed up and changed, the KB is re-indexed,
the replies stop being used (status 'merged'). Reject: the replies stop being used (status 'rejected')."""
import re
import shutil
import subprocess
import sys
import threading
from collections import Counter
from datetime import datetime

import numpy as np

from app import features, keywords, past_replies, store
from app.kb_common import KB_PATH, ROOT, log_error, log_info
from app.llm import DRAFT_MODEL, QUERY_MODEL, LLMError, chat_json
from app.pipeline import get_retriever

CLUSTER_SIM = 0.80  # English queries at least this similar = same topic
MAX_TOPICS = 15
MAX_EXAMPLES = 3
HEAD = re.compile(r"^### ([A-Z]\d+)\. (.*)$")
KB_ID = re.compile(r"^[A-Z]\d+$")
_lock = threading.Lock()

# Edit these two prompts freely. They are generic on purpose.
REVIEW_PROMPT = """You help a support team add knowledge from the replies of human support agents. You get what customers asked and what the agents replied on ONE topic (oldest first).
Return:
- question: one clear English question a customer would ask on this topic, in plain words.
- solution: the main answer the agents gave, for the reviewer to check. Skip replies that are only thanks, greetings or "we will check" replies. If the replies differ, follow the latest one. If the main answer is long (more than about 60 words), write a short summary of it instead and keep every fact. Keep the same language and script the agents used. Never add a fact that is not in the replies.
- topic: the best matching name from the TOPIC LIST, copied exactly.
- keywords: up to 5 short words customers use for this topic that are not in the KEYWORD LIST (may be empty).
JSON only: {"question": "", "solution": "", "topic": "", "keywords": []}"""

TRANSLATE_PROMPT = """Translate this support answer into clear, plain English for a knowledge base. Write it as facts: no greeting, no "Sir" or "Ma'am", no sentence that talks to one customer. Keep every fact, number, emoji and list. If it is already English, only tidy it. JSON only: {"english": ""}"""

TRANSLATE_QUESTION_PROMPT = """Translate or reframe this question into a clear, concise English heading/question for a knowledge base (for example: "Why is Tug and PK not showing to me?"). Keep it short, direct, and in plain English. If it is already in clear English, keep it as is. JSON only: {"english": ""}"""


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
    part_str = (part or "").strip()
    if len(part_str) == 1 and part_str.isalpha():
        part_letter = part_str.upper()
        topic_title = f"TOPIC {part_letter}"
    else:
        existing_parts = [m.group(1).upper() for m in (re.match(r"^##\s+PART\s+([A-Z]):", l) for l in lines) if m]
        next_letters = [chr(c) for c in range(ord('A'), ord('Z') + 1) if chr(c) not in existing_parts]
        part_letter = next_letters[0] if next_letters else "Z"
        topic_title = part_str.upper() if part_str else f"TOPIC {part_letter}"

    nums = [int(m.group(1)[1:]) for m in (HEAD.match(l) for l in lines) if m and m.group(1)[0] == part_letter]
    new_id = f"{part_letter}{max(nums, default=0) + 1}"
    start = next((n for n, l in enumerate(lines) if l.startswith(f"## PART {part_letter}")), None)
    if start is None:
        new_block = [
            "",
            "---",
            "",
            f"## PART {part_letter}: {topic_title}",
            "",
            f"### {new_id}. {title.strip()}",
            f"**Audience:** {audience.strip() or 'Everyone'}",
        ] + body.strip().splitlines()
        return lines + new_block, new_id

    end = next((n for n in range(start + 1, len(lines)) if lines[n].strip() == "---"), len(lines))
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    block = ["", f"### {new_id}. {title.strip()}", f"**Audience:** {audience.strip() or 'Everyone'}"] \
        + body.strip().splitlines()
    return lines[:end] + block + lines[end:], new_id


def _clean_body(text):
    """Keeps reviewer text from breaking the KB structure (no heading or divider lines)."""
    out = []
    for line in text.strip().splitlines():
        if line.strip() == "---":
            continue
        out.append(re.sub(r"^#+\s*", "", line.rstrip()))
    return "\n".join(out).strip()


def _audience_text(audience):
    """'Broadcaster, Participant' stays; nothing ticked or only Everyone -> 'Everyone'."""
    names = [a.strip() for a in (audience or "").split(",") if a.strip() and a.strip().lower() != "everyone"]
    return ", ".join(names) or "Everyone"


# ---------- build the weekly report ----------
def _clusters():
    ids = store.review_candidate_ids()
    if not ids:
        return []
    data = past_replies._collection().get(ids=ids, include=["embeddings", "documents", "metadatas"])
    items = [{"id": i, "vec": np.array(v), "reply": d, "q": m.get("question", ""), "eq": m.get("english_query", "")}
             for i, v, d, m in zip(data["ids"], data["embeddings"], data["documents"], data["metadatas"])]
    items.sort(key=lambda x: int(x["id"][1:]))  # oldest first
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


def _pick_feature(details):
    """The feature the drafts named most often; else the first feature in features.json."""
    names = Counter(d.get("draft_feature") for d in details if d.get("draft_feature"))
    for name, _ in names.most_common():
        f = features.find_by_name(name)
        if f:
            return f
    return features.load()[0]


def _stored_point(lines, details):
    """The KB point the drafts answered from (most used). None if the drafts were holding replies
    (NOT_IN_KB) or only used past replies."""
    count = Counter()
    for d in details:
        if d.get("draft_status") != "ANSWERED":
            continue
        for u in d.get("draft_used") or []:
            if KB_ID.match(str(u)):
                count[str(u)] += 1
                break
    for cid, _ in count.most_common():
        try:
            return chunk_info(lines, cid)
        except ValueError:
            continue
    return None


def _propose(cluster):
    details = store.feedback_details([e["id"] for e in cluster]).values()
    feature = _pick_feature(details)
    lines = features.kb_path(feature).read_text(encoding="utf-8").splitlines()
    stored = _stored_point(lines, details)
    topics = features.topics(feature)
    pairs = "\n".join(f"Customer: {e['eq'] or e['q']}\nAgent reply: {e['reply']}" for e in cluster[-MAX_EXAMPLES:])
    user = ("TOPIC LIST: " + ", ".join(t["name"] for t in topics) + "\nKEYWORD LIST: " + ", ".join(keywords.load())
            + "\n\nREPLIES (oldest first):\n" + pairs)
    data, _, _ = chat_json([{"role": "system", "content": REVIEW_PROMPT}, {"role": "user", "content": user}],
                           DRAFT_MODEL, 1500, "review")
    question = " ".join(str(data.get("question") or cluster[0]["eq"] or cluster[0]["q"]).split())
    solution = str(data.get("solution") or cluster[-1]["reply"]).strip()
    topic = str(data.get("topic") or "").strip().lower()
    part = stored["id"][0] if stored else next((t["part"] for t in topics if t["name"].lower() == topic), "")
    return {"action": "replace" if stored else "add", "chunk_id": stored["id"] if stored else "", "part": part,
            "title": stored["title"] if stored else question,
            "audience": stored["audience"] if stored else "Everyone",
            "body": solution, "old_text": stored["body"] if stored else "", "reason": "",
            "keywords": [str(k).strip() for k in (data.get("keywords") or []) if str(k).strip()][:5],
            "feedback_ids": [e["id"] for e in cluster],
            "examples": [{"q": e["eq"] or e["q"], "reply": e["reply"]} for e in cluster[:3]],
            "extra": {"feature_id": feature["id"], "llm_question": question,
                      "stored": {k: stored[k] for k in ("id", "title", "audience", "body")} if stored else None}}


def build_report():
    """Replaces pending proposals with fresh ones. Returns how many topics were proposed."""
    with _lock:
        # This collection is only a private review workspace. Draft generation
        # never queries it, so an unreviewed human reply cannot affect customers.
        past_replies.index_pending()
        store.clear_pending_proposals()
        count = 0
        for cluster in _clusters():
            try:
                store.add_proposal(_propose(cluster))
                count += 1
            except (LLMError, OSError) as e:
                log_error("LLM_ERROR", "review", f"Could not make a card for topic {cluster[0]['id']}: {e}")
        log_info("review", f"Weekly report ready: {count} topic(s)")
        return count


# ---------- apply / reject ----------
def _reingest():
    r = subprocess.run([sys.executable, "-m", "app.kb_ingest"], cwd=str(ROOT), capture_output=True,
                       text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError("KB re-index failed: " + (r.stderr or r.stdout)[-300:])


def _to_english(text):
    try:
        data, _, _ = chat_json([{"role": "system", "content": TRANSLATE_PROMPT}, {"role": "user", "content": text}],
                               QUERY_MODEL, 300, "review_translate")
    except LLMError as e:
        raise RuntimeError(f"Translation failed, please try again ({e})")
    out = str(data.get("english") or "").strip()
    if not out:
        raise RuntimeError("Translation came back empty, please try again")
    return out


def _question_to_english(text):
    text = (text or "").strip()
    if not text:
        return ""
    try:
        data, _, _ = chat_json([{"role": "system", "content": TRANSLATE_QUESTION_PROMPT}, {"role": "user", "content": text}],
                               QUERY_MODEL, 200, "review_translate_question")
        out = str(data.get("english") or "").strip()
        return out if out else text
    except Exception:
        return text


def apply_proposal(pid, mode, feature_id, part, title, audience, body, keywords_text):
    """mode 'replace' overwrites the stored point of this card; mode 'add' adds a new point at the end of the
    chosen topic. Both question and answer are translated to English before saving."""
    p = store.get_proposal(pid)
    if not p or p["status"] != "pending":
        raise ValueError("This card was not found or is already handled")
    feature = features.get(feature_id)
    if not feature:
        raise ValueError("Pick a feature")
    title = " ".join(title.split())
    body = body.strip()
    if mode not in ("replace", "add"):
        raise ValueError("Unknown action")
    if not title:
        raise ValueError("The question is empty")
    if not body:
        raise ValueError("The answer is empty")
    if mode == "replace":
        chunk_id = (p.get("chunk_id") or "").strip()
        if not chunk_id or feature_id != (p.get("extra") or {}).get("feature_id"):
            raise ValueError("Nothing is stored for this one. Use Add new")
    else:
        chunk_id = ""
        if not part.strip():
            raise ValueError("Pick a topic")
    kb = features.kb_path(feature)
    if kb.resolve() != KB_PATH.resolve():
        raise ValueError("This feature is not set up for review yet")
    audience_text = _audience_text(audience)
    english_title = _clean_body(_question_to_english(title))
    english = _clean_body(_to_english(body))  # before the KB is touched: a failed translation changes nothing
    with _lock:
        lines = kb.read_text(encoding="utf-8").splitlines()
        if mode == "replace":
            new_lines, target = edit_chunk(lines, chunk_id, english_title, audience_text, english), chunk_id
        else:
            new_lines, target = add_chunk(lines, part, english_title, audience_text, english)
        backup_dir = kb.parent / "kb_backups"
        backup_dir.mkdir(exist_ok=True)
        backup = backup_dir / f"kb_{datetime.now():%Y%m%d_%H%M%S}.md"
        shutil.copy2(kb, backup)
        kb.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        try:
            _reingest()
        except Exception:
            shutil.copy2(backup, kb)  # put the old KB back
            raise
        import app.pipeline as pipeline_mod
        pipeline_mod._retriever = None  # next search opens the re-indexed KB
        get_retriever()
        log_info("review", f"Proposal {pid}: {mode} {target} applied, backup {backup.name}")
        store.set_feedback_status(p["feedback_ids"], "merged")
        keywords.add((keywords_text or "").split(","))
        store.set_proposal_status(pid, "approved", english_title, audience_text, english, mode, target)


def reject_proposal(pid):
    p = store.get_proposal(pid)
    if not p or p["status"] != "pending":
        raise ValueError("This card was not found or is already handled")
    store.set_feedback_status(p["feedback_ids"], "rejected")
    store.set_proposal_status(pid, "rejected")
    log_info("review", f"Proposal {pid} rejected, {len(p['feedback_ids'])} repl(ies) marked rejected")
