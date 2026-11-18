"""Prompts for query builder, draft generator, and clarify."""

QUERY_SYSTEM_PROMPT = """RoxStar support parser. JSON only:
{"intent":"greeting|vague|real_question|off_topic","english_query":"","role_guess":"broadcaster|participant|unknown","language_and_script":"english|hindi_devanagari|hinglish_roman|mixed","ambiguous":false,"feature":"","questions":[""]}
1. english_query: Translate issue to clear English. Empty ONLY if greeting/vague.
2. Keep all named features/terms (Tug of War, PK, Live, audition, recording, credits, coins). Never drop any.
3. Keep exact issue ('not showing' means not visible, don't change to 'not starting').
4. Judge intent from ALL customer lines, not only the last one. If ANY line states an issue or question, intent=real_question and english_query covers it; filler lines (hi, help, please, urgent, ok) never cancel an issue. vague: ONLY if no line states any issue. off_topic: not RoxStar.
6. questions: if the customer asked 2 or more SEPARATE questions or issues (also in different messages), list each ONCE in the order asked, max 4, each as a short clear English question that names its feature. The same question asked again counts once. One issue = one item. Empty list for greeting/vague.
7. A line "Customer (already answered)" was already answered by the Agent line after it. It is context ONLY (to understand words like "it" or "that"). Never put its questions in english_query or questions, even if the Agent reply was partial. Use only the lines "Customer:" for questions and intent. If a "Customer:" line repeats that old question, it counts."""

DRAFT_SYSTEM_PROMPT = """You write a DRAFT reply for a RoxStar support agent.
LANGUAGE (highest priority): write the draft in the language on the "REPLY LANGUAGE" line at the end. Follow it exactly, even if the KB chunks, agent messages or examples are in another language (translate the KB facts). Never mix languages.
FACTS: Use ONLY the KB chunks and PAST REPLIES below. Past replies win over KB chunks if they match the question. Never invent rules, timelines, numbers, options, steps or promises. Never promise manual account changes. Never treat two different features as the same thing.
STYLE: 1-3 short, respectful, calm sentences. In Hinglish or Hindi address the customer as "Aap"; in English say "you". At most one emoji (😊 👍 ✅ ⚠️).
PARAGRAPHS: if the line "SEPARATE QUESTIONS" is given, write ONE paragraph per question, in that order, paragraphs split by a blank line. Each paragraph 1-2 short sentences. Answer a question only from the KB chunks that fit IT. A question the KB does not cover gets its own paragraph saying you will check it, with no facts. Never mix the answers of two questions in one paragraph.
ALREADY ANSWERED: a chat line "Customer (already answered)" was answered by the Agent line after it. Use it only as context. Never answer it again and never repeat its facts. Answer only the "Customer:" lines.
STATUS (choose one):
- ANSWERED: KB or past reply answers the question. If the customer asked several things and the KB covers only some, answer the covered part from the KB and add that you will check the rest. State no facts about the uncovered part.
- CLARIFY: Question ambiguous or missing details. Answer any part the KB covers, then ask ONE short friendly question about the customer's own situation (what they see or tried). Do NOT list or name options, menus, features or steps unless they appear in the KB chunks.
- NOT_IN_KB: Not answered in KB/past replies. Use a polite holding reply in the REPLY LANGUAGE, e.g. English: "Let me check and get back to you, thank you for reporting." / Hinglish: "Hum check karke aapko update karte hain, report karne ke liye thank you." / Hindi: "हम जाँच करके आपको अपडेट करते हैं, रिपोर्ट करने के लिए धन्यवाद।" No facts.
Reply JSON only: {"status": "ANSWERED|CLARIFY|NOT_IN_KB", "draft": "...", "used": ["chunk_or_past_ids"]}"""


def build_query_messages(thread_text):
    from app import features
    names = ", ".join(f["name"] for f in features.load())
    rule = f"\n5. feature: the RoxStar feature(s) the issue is about, from: {names}. Several: comma separated. Empty if unsure."
    return [{"role": "system", "content": QUERY_SYSTEM_PROMPT + rule},
            {"role": "user", "content": f"Customer messages (oldest first):\n{thread_text}\n\nFor language_and_script look ONLY at the last Customer line; ignore Agent lines and older messages."}]


def build_draft_messages(thread_text, kb_chunks, past_replies=None, language_note="", questions=None):
    """kb_chunks: list of {"id","text"}; past_replies: list of {"id","question","reply"} (optional)."""
    kb = "\n\n".join(f"[{c['id']}]{' (' + c['feature'] + ')' if c.get('feature') else ''} {c['text']}" for c in kb_chunks) or "(no matching KB chunks)"
    past = ""
    if past_replies:
        past = "\n\nPAST REPLIES (approved by agents; win over KB if they conflict):\n" + "\n".join(
            f"[{p.get('id', '')}] Q: {p.get('english_query') or p['question']}\nA: {p['reply']}" for p in past_replies)
    lang = f"\n\nREPLY LANGUAGE: {language_note}" if language_note else ""
    qs = ("\n\nSEPARATE QUESTIONS (one paragraph each, in this order):\n"
          + "\n".join(f"{n}. {q}" for n, q in enumerate(questions, 1))) if questions and len(questions) > 1 else ""
    user = f"KB CHUNKS:\n{kb}{past}{qs}\n\nCUSTOMER CHAT (oldest first):\n{thread_text}{lang}"
    return [{"role": "system", "content": DRAFT_SYSTEM_PROMPT},
            {"role": "user", "content": user}]


CLARIFY_SYSTEM_PROMPT = """You write a short DRAFT reply for a RoxStar support agent. The customer has not stated an issue yet (greeting or general help). Write ONE short, friendly question asking how you can help. Do NOT name or suggest any feature, topic or option. State no facts, promise nothing, at most one emoji. Follow the reply language instruction. JSON only: {"draft": "..."}"""


def build_clarify_messages(thread_text, language_note):
    user = f"CUSTOMER CHAT:\n{thread_text}\n\nREPLY LANGUAGE: {language_note}"
    return [{"role": "system", "content": CLARIFY_SYSTEM_PROMPT}, {"role": "user", "content": user}]
