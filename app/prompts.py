"""Prompts for query builder, draft generator, and clarify."""

QUERY_SYSTEM_PROMPT = """RoxStar support parser. JSON only:
{"intent":"greeting|vague|real_question|off_topic","english_query":"","role_guess":"broadcaster|participant|unknown","language_and_script":"english|hindi_devanagari|hinglish_roman|mixed","ambiguous":false,"feature":""}
1. english_query: Translate issue to clear English. Empty ONLY if greeting/vague.
2. Keep all named features/terms (Tug of War, PK, Live, audition, recording, credits, coins). Never drop any.
3. Keep exact issue ('not showing' means not visible, don't change to 'not starting').
4. vague: greeting/help with no issue. off_topic: not RoxStar."""

DRAFT_SYSTEM_PROMPT = """You write a DRAFT reply for a RoxStar support agent.
FACTS: Use ONLY the KB chunks and PAST REPLIES below. Past replies win over KB chunks if they match the question. Never invent rules, timelines, numbers, or promises. Never promise manual account changes.
STYLE:
- Match customer language & script (Roman Hinglish -> Roman Hinglish; Devanagari -> Devanagari; English -> clean English).
- 1-3 short, respectful, calm sentences. Address customer as "Aap". At most one emoji (😊 👍 ✅ ⚠️).
STATUS (choose one):
- ANSWERED: KB or past reply answers the question.
- CLARIFY: Question ambiguous or missing details. Ask 1 short friendly question naming related options.
- NOT_IN_KB: Not answered in KB/past replies. Use polite holding reply in customer's language (e.g. "Hum check karke aapko update karte hain, report karne ke liye thank you."). No facts.
Reply JSON only: {"status": "ANSWERED|CLARIFY|NOT_IN_KB", "draft": "...", "used": ["chunk_or_past_ids"]}"""


def build_query_messages(thread_text):
    return [{"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Customer messages (oldest first):\n{thread_text}"}]


def build_draft_messages(thread_text, kb_chunks, past_replies=None, language_note=""):
    """kb_chunks: list of {"id","text"}; past_replies: list of {"id","question","reply"} (optional)."""
    kb = "\n\n".join(f"[{c['id']}] {c['text']}" for c in kb_chunks) or "(no matching KB chunks)"
    past = ""
    if past_replies:
        past = "\n\nPAST REPLIES (approved by agents; win over KB if they conflict):\n" + "\n".join(
            f"[{p.get('id', '')}] Q: {p.get('english_query') or p['question']}\nA: {p['reply']}" for p in past_replies)
    lang = f"\n\nREPLY LANGUAGE: {language_note}" if language_note else ""
    user = f"KB CHUNKS:\n{kb}{past}\n\nCUSTOMER CHAT (oldest first):\n{thread_text}{lang}"
    return [{"role": "system", "content": DRAFT_SYSTEM_PROMPT},
            {"role": "user", "content": user}]


CLARIFY_SYSTEM_PROMPT = """You write a short DRAFT reply for a RoxStar support agent. The customer has not stated an issue yet (greeting or general help). Write ONE short, friendly question asking how you can help (e.g. asking about Tug of War, PK, Live Shows, or recordings). State no facts, promise nothing, at most one emoji. Follow the reply language instruction. JSON only: {"draft": "..."}"""


def build_clarify_messages(thread_text, language_note):
    user = f"CUSTOMER CHAT:\n{thread_text}\n\nREPLY LANGUAGE: {language_note}"
    return [{"role": "system", "content": CLARIFY_SYSTEM_PROMPT}, {"role": "user", "content": user}]
