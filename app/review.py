"""Weekly review. Private reply candidates are grouped by topic (local embeddings). For each topic ONE small LLM call
writes: the English question, the main solution (short summary if long), the topic and search words.
The reviewer picks the feature, topic and audience, edits the text and presses:
  Save card (staged: nothing is written yet) or Reject. "+ Add card" makes a blank card by hand.
Update KB (one button for all staged cards): answers are translated to English first, then each touched KB file is
backed up and changed, the KB is re-indexed ONCE, and the replies stop being used (status 'merged'). All or nothing:
if anything fails the files are put back. Reject: the replies stop being used (status 'rejected').
Search words are NOT suggested by the AI. They are blank unless the reviewer types them; they are saved in the
feature's KB file (see features.py)."""
import re
import subprocess
import sys
import threading
from collections import Counter

import numpy as np

from app import features, past_replies, store
from app.kb_common import ROOT, log_error, log_info
from app.llm import DRAFT_MODEL, QUERY_MODEL, LLMError, chat_json
from app.pipeline import get_retriever

CLUSTER_SIM = 0.80  # English queries at least this similar = same topic
MAX_TOPICS = 15
MAX_EXAMPLES = 3
HEAD = re.compile(r"^### ([A-Z]\d+)\. (.*)$")
KB_ID = re.compile(r"^(?:([a-z0-9_]+):)?([A-Z]\d+)$")  # "tow:B5" (old rows: "B5")
_lock = threading.Lock()

# Edit these two prompts freely. They are generic on purpose.
REVIEW_PROMPT = """You help a support team add knowledge from the replies of human support agents. You get what customers asked and what the agents replied on ONE topic (oldest first).
Return:
- question: one clear English question a customer would ask on this topic, in plain words.
- solution: the main answer the agents gave, for the reviewer to check. Skip replies that are only thanks, greetings or "we will check" replies. If the replies differ, follow the latest one. If the main answer is long (more than about 60 words), write a short summary of it instead and keep every fact. Keep the same language and script the agents used. Never add a fact that is not in the replies.
- topic: the best matching name from the TOPIC LIST, copied exactly.
- If a FEATURE line is given and the replies talk about more than one feature, keep ONLY the facts about that FEATURE in solution and question. The KB keeps one feature per entry.
JSON only: {"question": "", "solution": "", "topic": ""}"""

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
    # The Audience line is optional: pk.md, popular-singer.md and friendship.md entries have none.
    has_aud = s + 1 < e and lines[s + 1].lstrip().startswith("**Audience:**")
    aud = lines[s + 1].replace("**Audience:**", "").strip() if has_aud else "Everyone"
    return {"id": cid, "title": HEAD.match(lines[s]).group(2), "audience": aud,
            "body": "\n".join(lines[s + (2 if has_aud else 1):e]).strip(), "text": "\n".join(lines[s:e]).strip()}


def edit_chunk(lines, cid, title, audience, body):
    """Replaces one chunk in place: same ID, new title (question), audience and body."""
    s, e = _find_chunk(lines, cid)
    head = f"### {cid}. {title.strip()}" if title.strip() else lines[s]
    aud = f"**Audience:** {audience.strip() or 'Everyone'}"
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
def is_multi_feature(detail):
    """True if the draft touched more than one feature: its answer used chunks of 2+ features, or the query
    builder named 2+ features (e.g. "PK Battle, Tug of War"). Such drafts never make a card: one card = one feature."""
    used = {m.group(1) for u in detail.get("draft_used") or []
            if (m := KB_ID.match(str(u))) and m.group(1) and features.get(m.group(1))}
    named = {f["id"] for n in (detail.get("draft_feature") or "").split(",") if (f := features.find_by_name(n))}
    return len(used) > 1 or len(named) > 1


# Words an agent adds out of politeness. An edit that only adds/removes these (or emoji, punctuation) teaches nothing.
_FILLER = {"thank", "thanks", "thankyou", "thanku", "thnx", "hi", "hello", "hii", "hey", "dear", "sir", "madam", "maam",
           "ok", "okay", "regards", "dhanyawad", "dhanyavaad", "shukriya", "shukriyaa", "dhanyavad", "you", "so", "much"}


def _norm(text):
    """Lower case, no emoji or punctuation, no polite filler words: only the real content is left."""
    t = re.sub(r"[^\w\s]|_", " ", (text or "").lower())
    return " ".join(w for w in t.split() if w not in _FILLER)


def skip_reason(detail):
    """Why a draft must NOT make a review card (None = fine). Cards come only from clean, single-feature drafts
    where the agent really changed the ANSWER."""
    if is_multi_feature(detail):
        return "multi-feature"
    final, draft = (detail.get("final_text") or "").strip(), (detail.get("draft_text") or "").strip()
    if draft and _norm(final) == _norm(draft):
        return "trivial edit (thanks, greeting, emoji or punctuation only)"
    # The agent typed a question or clarification instead of an answer: it ends with "?" and does not contain the
    # draft (an answer plus a polite closing question is still an answer).
    if re.sub(r"[^\w?？]+$", "", final).endswith(("?", "？")) and (not draft or _norm(draft) not in _norm(final)):
        return "agent question / clarification"
    return None


def _clusters():
    ids = store.review_candidate_ids()
    if not ids:
        return []
    details = store.feedback_details(ids)
    skipped = {}
    for i in ids:
        why = skip_reason(details[i]) if i in details else None
        if why:
            skipped[i] = why
    if skipped:
        log_info("review", f"Skipped {len(skipped)} draft(s), no card made: {skipped}")
        ids = [i for i in ids if i not in skipped]
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
    """Returns (feature or None, [other feature names the answers also used]).
    1) the feature of the KB chunks the ANSWERED drafts used most; 2) else the query builder's guess;
    3) else None: the reviewer must choose (no silent default to the first feature)."""
    used = Counter()
    for d in details:
        if d.get("draft_status") != "ANSWERED":
            continue
        for u in d.get("draft_used") or []:
            m = KB_ID.match(str(u))
            if m and m.group(1) and features.get(m.group(1)):
                used[m.group(1)] += 1
    if used:
        ranked = [features.get(fid) for fid, _ in used.most_common()]
        return ranked[0], [f["name"] for f in ranked[1:]]
    names = Counter(d.get("draft_feature") for d in details if d.get("draft_feature"))
    for name, _ in names.most_common():
        f = features.find_by_name(name)
        if f:
            return f, []
    return None, []


def _stored_point(lines, details, feature_id):
    """The KB point the drafts answered from (most used). None if the drafts were holding replies
    (NOT_IN_KB) or only used past replies. Only points of this feature count."""
    count = Counter()
    for d in details:
        if d.get("draft_status") != "ANSWERED":
            continue
        for u in d.get("draft_used") or []:
            m = KB_ID.match(str(u))
            if m and m.group(1) in (None, feature_id):
                count[m.group(2)] += 1
                break
    for cid, _ in count.most_common():
        try:
            return chunk_info(lines, cid)
        except ValueError:
            continue
    return None


def _propose(cluster):
    details = store.feedback_details([e["id"] for e in cluster]).values()
    feature, also = _pick_feature(details)
    lines = features.kb_path(feature).read_text(encoding="utf-8").splitlines() if feature else []
    stored = _stored_point(lines, details, feature["id"]) if feature else None
    topics = features.topics(feature) if feature else []
    pairs = "\n".join(f"Customer: {e['eq'] or e['q']}\nAgent reply: {e['reply']}" for e in cluster[-MAX_EXAMPLES:])
    user = ((f"FEATURE: {feature['name']}\n" if feature else "") + "TOPIC LIST: " + ", ".join(t["name"] for t in topics)
            + "\n\nREPLIES (oldest first):\n" + pairs)
    data, _, _ = chat_json([{"role": "system", "content": REVIEW_PROMPT}, {"role": "user", "content": user}],
                           DRAFT_MODEL, 1500, "review")
    question = " ".join(str(data.get("question") or cluster[0]["eq"] or cluster[0]["q"]).split())
    solution = str(data.get("solution") or cluster[-1]["reply"]).strip()
    topic = str(data.get("topic") or "").strip().lower()
    part = stored["id"][0] if stored else next((t["part"] for t in topics if t["name"].lower() == topic), "")
    return {"action": "replace" if stored else "add", "chunk_id": f"{feature['id']}:{stored['id']}" if stored else "", "part": part,
            "title": stored["title"] if stored else question,
            "audience": stored["audience"] if stored else "Everyone",
            "body": solution,
            "keywords": [],  # search words are typed by a person, never suggested by the AI
            "feedback_ids": [e["id"] for e in cluster],
            "extra": {"feature_id": feature["id"] if feature else "", "also_features": also, "llm_question": question,
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


def _words(text):
    return [w.strip() for w in (text or "").split(",") if w.strip()]


def list_points(feature_id):
    """Every Q&A point of one feature (for 'replace an existing point'): id, title, audience, body, topic letter."""
    feature = features.get(feature_id)
    if not feature:
        raise ValueError("Feature not found")
    lines = features.kb_path(feature).read_text(encoding="utf-8").splitlines()
    out = []
    for n, line in enumerate(lines):
        m = HEAD.match(line)
        if m:
            info = chunk_info(lines, m.group(1))
            out.append({"id": info["id"], "title": info["title"], "audience": info["audience"],
                        "body": info["body"], "part": info["id"][0]})
    return out


def stage_card(pid, mode, feature_id, part, title, audience, body, keywords_text, chunk=""):
    """Reviewer pressed Save card. Checks everything the KB will need, writes NOTHING to the KB."""
    p = store.get_proposal(pid)
    if not p or p["status"] not in ("pending", "staged"):
        raise ValueError("This card was not found or is already handled")
    feature = features.get(feature_id)
    if not feature:
        raise ValueError("Pick a feature")
    title = " ".join((title or "").split())
    body = (body or "").strip()
    if mode not in ("replace", "add"):
        raise ValueError("Unknown action")
    if not title:
        raise ValueError("The question is empty")
    if not body:
        raise ValueError("The answer is empty")
    extra = p.get("extra") or {}
    local = ""
    if mode == "replace":
        m = KB_ID.match((chunk or p.get("chunk_id") or "").strip())
        local = m.group(2) if m and m.group(1) in (None, feature_id) else ""
        if not local:
            raise ValueError("Pick the point to replace")
        if not extra.get("manual") and feature_id != extra.get("feature_id"):
            raise ValueError("Nothing is stored for this one. Use Add new")
        lines = features.kb_path(feature).read_text(encoding="utf-8").splitlines()
        chunk_info(lines, local)  # raises if the point does not exist in this feature's file
        for o in store.list_proposals("staged"):
            st = (o.get("extra") or {}).get("stage") or {}
            if o["id"] != pid and st.get("mode") == "replace" and st.get("feature_id") == feature_id and st.get("chunk") == local:
                raise ValueError(f"Another saved card already replaces {feature_id}:{local}. Edit that one instead")
    elif not (part or "").strip():
        raise ValueError("Pick a topic")
    store.stage_proposal(pid, {"feature_id": feature_id, "mode": mode, "part": (part or "").strip(), "chunk": local},
                         title, _audience_text(audience), body, _words(keywords_text))


def unstage_card(pid):
    try:
        store.unstage_proposal(pid)
    except KeyError:
        raise ValueError("This card is not saved")


def add_manual_card():
    return store.add_manual_proposal()


def update_kb():
    """Writes ALL saved (staged) cards into the KB with ONE re-index. All or nothing."""
    staged = store.list_proposals("staged")
    if not staged:
        raise ValueError("No saved cards. Press Save card on a card first")
    # 1. translate everything BEFORE the KB is touched: a failed translation changes nothing
    prepared = []
    for p in staged:
        st = (p.get("extra") or {}).get("stage") or {}
        if not st:
            raise ValueError(f"Card {p['id']} has no saved action. Save it again")
        prepared.append((p, st, _clean_body(_question_to_english(p["title"])), _clean_body(_to_english(p["body"]))))
    with _lock:
        by_file, done = {}, []
        # 2. edit the files in memory, in card order
        for p, st, title_en, body_en in prepared:
            f = features.get(st["feature_id"])
            if not f:
                raise ValueError(f"Card {p['id']}: feature not found")
            kb = features.kb_path(f)
            lines = by_file.get(kb) or kb.read_text(encoding="utf-8").splitlines()
            if st["mode"] == "replace":
                lines, local = edit_chunk(lines, st["chunk"], title_en, p["audience"], body_en), st["chunk"]
            else:
                lines, local = add_chunk(lines, st["part"], title_en, p["audience"], body_en)
            by_file[kb] = lines
            done.append((p, title_en, body_en, f"{f['id']}:{local}", st))
        for kb, lines in list(by_file.items()):  # feature-level search words (typed by a person), same file
            words = [w for p, _, _, _, st in done if features.get(st["feature_id"]) and features.kb_path(features.get(st["feature_id"])) == kb for w in p["keywords"]]
            by_file[kb] = features.add_words_to_lines(lines, words)
        # 3. write and re-index once. The old text is kept in MEMORY only (no backup files on disk) so a failure
        #    can put every file back
        originals = {kb: kb.read_text(encoding="utf-8") for kb in by_file}
        try:
            for kb, lines in by_file.items():
                kb.write_text("\n".join(lines) + "\n", encoding="utf-8")
            _reingest()
        except Exception:
            for kb, old in originals.items():
                kb.write_text(old, encoding="utf-8")  # put every old file back
            try:
                _reingest()
            except Exception:
                pass
            raise
        import app.pipeline as pipeline_mod
        pipeline_mod._retriever = None  # next search opens the re-indexed KB
        get_retriever()
        for p, title_en, body_en, target, st in done:
            store.set_feedback_status(p["feedback_ids"], "merged")
            store.set_proposal_status(p["id"], "approved", title_en, p["audience"], body_en, st["mode"], target)
        log_info("review", f"Update KB: {len(done)} card(s), files {[k.name for k in by_file]}")
        return {"updated": len(done), "targets": [d[3] for d in done]}


def reject_proposal(pid):
    p = store.get_proposal(pid)
    if not p or p["status"] not in ("pending", "staged"):
        raise ValueError("This card was not found or is already handled")
    store.set_feedback_status(p["feedback_ids"], "rejected")
    store.set_proposal_status(pid, "rejected")
    log_info("review", f"Proposal {pid} rejected, {len(p['feedback_ids'])} repl(ies) marked rejected")
