"""Retrieval-only eval (no LLM calls): does the top result come from the expected feature?
Run: .venv/bin/python3 -m scripts.eval_retrieval   (from tow-support-assistant)
hint = what the query builder would guess for `feature`; '' = no guess."""
from app import features
from app.retriever import Retriever, match_features

# (english query, expected feature id, hint)
CASES = [
    ("How do I invite another broadcaster to a PK battle?", "pk", "PK Battle"),
    ("PK invitation option is not visible", "pk", "PK Battle"),
    ("Free tickets are not working in PK", "pk", "PK Battle"),
    ("Why is my voice echoing during PK?", "pk", ""),
    ("PK option is disabled on my live", "pk", "PK Battle"),
    ("How many credits do I get in Tug of War?", "tow", "Tug of War"),
    ("How much do I win in Tug of War?", "tow", "Tug of War"),
    ("I am only receiving 2 credits instead of 3", "tow", ""),
    ("What is my Health Heart colour and weekly credits?", "tow", ""),
    ("How do I certify my friendship relationship?", "friendship", "Friendship Feature"),
    ("How do I earn points in the chat task?", "friendship", "Friendship Feature"),
    ("What is a customized partner frame?", "friendship", ""),
    ("When does voting close in the popular singer contest?", "popular_singer", "Popular Singer Contest"),
    ("How do I join the singing queue?", "popular_singer", ""),
    ("Who wins the singer contest?", "popular_singer", "Popular Singer Contest"),
    ("What is the difference between PK and Tug of War?", "pk", "Tug of War, PK Battle"),
    ("Both PK and Tug of War are disabled on my live", "pk", "Tug of War, PK Battle"),
    ("How do I recharge coins?", None, ""),
    ("How do I do a referral?", None, ""),
]

r = Retriever()
F = features.load()
ok = total = 0
for q, exp, hint in CASES:
    ids = match_features(hint, F)
    res = r.search(q, top_k=2, feature_ids=ids)
    top = res[0]
    feats = [c["feature_id"] for c in res]
    best = max((c["dense_sim"] or 0 for c in res), default=0)
    if exp is None:
        good = best < 0.50
        note = f"best dense {best:.2f} (want < 0.50 -> holding reply)"
    else:
        good = top["feature_id"] == exp
        note = f"top={top['id']} all={[c['id'] for c in res]} dense={best:.2f}"
    ok += good
    total += 1
    print(("OK  " if good else "FAIL"), q[:55].ljust(55), note)
print(f"\n{ok}/{total} as expected")
