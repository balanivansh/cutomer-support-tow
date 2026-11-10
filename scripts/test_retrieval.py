"""Retrieval check. Each question is tested two ways:
  raw = the customer's text as typed (worst case, no role filter)
  eng = the clean English query + role that the query-builder step will produce in the real pipeline
Run from the project folder:  python scripts/test_retrieval.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.kb_common import get_logger  # noqa: E402
from app.retriever import Retriever  # noqa: E402

get_logger().setLevel("WARNING")  # keep the test output short

# (raw text, English query, role, acceptable chunk ids)
CASES = [
    ("How many free credits do I get?", "How many free credits do I get?", "broadcaster", {"B5"}),
    ("mujhe kitne free credit milte hain", "How many free credits does a broadcaster get?", "broadcaster", {"B5"}),
    ("मुझे कितने फ्री क्रेडिट मिलते हैं", "How many free credits do I get?", "broadcaster", {"B5"}),
    ("tug credit nahi mil raha", "Broadcaster's free Tug of War credit was not added OR Gold Coins reward not received", "broadcaster", {"B5", "B13", "B3", "B12"}),
    ("का TUG कम हो गया है, कृपया जाँच करें", "Broadcaster's Tug of War credit has decreased, please check", "broadcaster", {"B5", "B13", "C10", "B14"}),
    ("match ke baad coins nahi aaye", "Participant did not receive coins after the match", "participant", {"C8", "C6"}),
    ("mai winner tha mujhe kitne coins milenge", "How many coins does the winner get?", "participant", {"C6"}),
    ("broadcaster ko kitne coins milte hain", "How many coins does the broadcaster earn from a match?", "broadcaster", {"B12"}),
    ("match cancel kar sakte hain kya", "Can the broadcaster cancel a match and get coins back?", "broadcaster", {"B8"}),
    ("kya audience coins add kar sakti hai", "Can the audience add coins to fund a match?", "broadcaster", {"B4"}),
    ("leaderboard mein mera naam nahi aa raha", "Why am I not on today's leaderboard?", None, {"D1"}),
    ("my recording got rejected, why", "Why was my recording rejected?", "broadcaster", {"B14"}),
    ("stream band ho gayi aur match ruk gaya", "The live stream ended and the match stopped. Can it be resumed?", "broadcaster", {"B7"}),
    ("daily 4 match ka target kya hai", "What is the daily 4-match goal?", "broadcaster", {"B9"}),
    ("tug start nahi ho raha", "Tug of War match is not starting, participants cannot join", "broadcaster", {"B3", "B4", "B6"}),
    ("kitne tug khel sakte hain 100 coin se", "How many matches can be started with 100 coins?", "broadcaster", {"B3", "B2"}),
]
NOT_IN_KB = [
    ("mera account ban kyu hua", "Why was my account banned?"),
    ("How do I reset my router?", "How do I reset my router?"),
]


def fmt(res):
    return ", ".join(f"{x['id']} ({x['dense_sim']:.2f})" if x["dense_sim"] is not None else f"{x['id']} (kw only)"
                     for x in res)


def top_dense(res):
    return max((x["dense_sim"] for x in res if x["dense_sim"] is not None), default=0.0)


def grade(res, expected):
    ids = [x["id"] for x in res]
    if ids and ids[0] in expected:
        return "PASS", 1, 1
    if any(i in expected for i in ids):
        return "top3", 0, 1
    return "FAIL", 0, 0


def main():
    r = Retriever()
    t = {"raw": [0, 0], "eng": [0, 0]}
    print()
    for raw, eng, role, expected in CASES:
        rr = r.search(raw, top_k=3)
        er = r.search(eng, role=role, top_k=3)
        gr, a1, a3 = grade(rr, expected)
        ge, b1, b3 = grade(er, expected)
        t["raw"][0] += a1; t["raw"][1] += a3
        t["eng"][0] += b1; t["eng"][1] += b3
        print(f"[raw:{gr} eng:{ge}] {raw}   (expect {sorted(expected)})")
        print(f"    raw -> {fmt(rr)}")
        print(f"    eng -> {fmt(er)}   role={role}")
    n = len(CASES)
    print(f"\nraw text : top-1 {t['raw'][0]}/{n}, in top-3 {t['raw'][1]}/{n}")
    print(f"english  : top-1 {t['eng'][0]}/{n}, in top-3 {t['eng'][1]}/{n}   <- this is what the real pipeline sees")
    print("\nShould NOT match (compare top dense similarity with the PASS lines above):")
    for raw, eng in NOT_IN_KB:
        rr = r.search(raw, top_k=3)
        er = r.search(eng, top_k=3)
        print(f"- {raw}\n    raw -> top dense {top_dense(rr):.3f} {[x['id'] for x in rr]}"
              f"\n    eng -> top dense {top_dense(er):.3f} {[x['id'] for x in er]}")


if __name__ == "__main__":
    main()
