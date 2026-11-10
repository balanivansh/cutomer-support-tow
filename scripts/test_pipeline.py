"""Try the draft pipeline from the terminal.
  python scripts/test_pipeline.py                       -> runs the built-in test threads
  python scripts/test_pipeline.py "goodmorning" "tug credit nahi hua" "please help"
                                                         -> one thread made of the messages you type
Free Groq tier allows about 8,000 tokens a minute, so the built-in run pauses between threads."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import generate_draft  # noqa: E402

THREADS = [
    (["goodmorning"], "CLARIFY (greeting only)"),
    (["goodmorning", "tug credit is not being updated", "pleaseee help me"], "CLARIFY or ANSWERED (credit vs coins)"),
    (["tug credit nahi mil raha please help"], "CLARIFY (credit or Gold Coins?)"),
    (["tug start ni ho rha"], "ANSWERED about B3/B4/B6"),
    (["mera tug credit 1 ho gaya hai 3 kardo jaldi"], "ANSWERED about Health Heart (B5/B13/C10)"),
    (["match ke baad coins nahi aaye, mai winner tha"], "ANSWERED about C8 (participant)"),
    (["मेरा TUG क्रेडिट कम हो गया है, कृपया जाँच करके बताइए"], "reply in Devanagari"),
    (["kitne tug khel sakte hain 100 coin se"], "ANSWERED: 5 matches of 4 or 3 of 6"),
    (["mera account ban kyu hua"], "NOT_IN_KB"),
    (["How do I reset my router?"], "NOT_IN_KB"),
]


def show(msgs, note, n):
    r = generate_draft(msgs, conv_id=f"t{n}")
    print(f"\n=== Thread {n}: {msgs}\n    expect: {note}")
    if r["status"] == "ERROR":
        print(f"    ERROR: {r['error']}  (see the [API_ERROR]/[LLM_ERROR]/[RETRIEVAL_ERROR] log line above)")
        return 0
    q = r.get("query") or {}
    print(f"    status : {r['status']}   used: {r['used']}   role: {q.get('role_guess')}   "
          f"lang: {q.get('language_and_script')}   ambiguous: {q.get('ambiguous')}")
    print(f"    english query: {q.get('english_query')!r}")
    print(f"    sources: {[(s['id'], round(s['dense_sim'], 2) if s['dense_sim'] is not None else None) for s in r['sources']]}")
    print(f"    DRAFT  : {r['draft']}")
    t = r.get("tokens") or {"in": 0, "out": 0}
    print(f"    tokens : in={t['in']} out={t['out']}   time: {r['ms']} ms")
    return t["in"] + t["out"]


def main():
    if len(sys.argv) > 1:
        show(sys.argv[1:], "your own thread", 1)
        return
    total = 0
    for n, (msgs, note) in enumerate(THREADS, 1):
        used = show(msgs, note, n)
        total += used
        if used and n < len(THREADS):
            time.sleep(25)  # stay under the free-tier tokens-per-minute limit
    print(f"\nTotal tokens used in this run: {total} (free daily limit is 200,000)")


if __name__ == "__main__":
    main()
