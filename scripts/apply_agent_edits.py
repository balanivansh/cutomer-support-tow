"""Simulate agent edits for Q10 and Q19, verify feedback storage,
confirm live retrieval safety (no auto-learning), and test weekly review proposals.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import store, review
from app.pipeline import generate_draft

def main():
    store.init_db()

    # --- 1. Setup Q19 ---
    cid_19 = store.create_conversation()
    q19_cust = "meri tug or pk show nehi kor raha"
    store.add_message(cid_19, "customer", q19_cust)
    ctx_19 = store.generation_context(cid_19)

    draft_19_res = {
        "status": "NOT_IN_KB",
        "draft": "Aapka tug ya pk show start nahi ho raha hai, iske liye thoda aur detail de sakte hain? Jaise aapne kaunse tarike se start kiya (credit ya coins) aur koi error message aaya ho toh batayein. 😊",
        "used": [],
        "sources": [],
        "query": {"intent": "real_question", "role_guess": "broadcaster", "language_and_script": "hinglish_roman", "english_query": "The Tug of War match is not starting for the broadcaster."}
    }
    draft_id_19 = store.save_draft(cid_19, draft_19_res, ctx_19)

    edited_19 = (
        "aapka live audition period hai isliye\n"
        "apko live final approve krane ke liye, live se ek video recording post kriye\n"
        "24-48 ghante me aapka live final approve hoga aapki video recording ke basis par"
    )
    store.add_message(cid_19, "agent", edited_19)
    outcome_19 = store.save_feedback(cid_19, draft_id_19, q19_cust, edited_19)
    store.mark_answered(cid_19, draft_id_19)

    print(f"Q19 recorded:")
    print(f"  Conversation ID: {cid_19}")
    print(f"  Draft ID: {draft_id_19}")
    print(f"  Outcome: {outcome_19}")

    # --- 2. Setup Q10 ---
    cid_10 = store.create_conversation()
    q10_cust = "main 4 song pehle tug recording karke recommend apply tha phir v red hai abhi sir"
    store.add_message(cid_10, "customer", q10_cust)
    ctx_10 = store.generation_context(cid_10)

    draft_10_res = {
        "status": "ANSWERED",
        "draft": "Aapke recordings weekend review mein reject hue honge ya quality low thi, isliye Health Heart red ho gaya aur free tug credits kam ho gaye. Clear, high-quality singing recordings post kijiye aur Monday tak active rahiye taaki credits wapas aayen. 😊",
        "used": ["B14"],
        "sources": [{"id": "B14", "dense_sim": 0.72}],
        "query": {"intent": "real_question", "role_guess": "broadcaster", "language_and_script": "hinglish_roman", "english_query": "The recording of my Tug of War match from four songs ago is still showing as red and not approved"}
    }
    draft_id_10 = store.save_draft(cid_10, draft_10_res, ctx_10)

    edited_10 = (
        "har tug of war record karna hota hai, har ek hafte 21 Recordings minimum\n"
        "har tug of War ki video banaiye iss week next week ho jayega agar video sahi hai toh"
    )
    store.add_message(cid_10, "agent", edited_10)
    outcome_10 = store.save_feedback(cid_10, draft_id_10, q10_cust, edited_10)
    store.mark_answered(cid_10, draft_id_10)

    print(f"\nQ10 recorded:")
    print(f"  Conversation ID: {cid_10}")
    print(f"  Draft ID: {draft_id_10}")
    print(f"  Outcome: {outcome_10}")

    # --- 3. Safety Check: Verify Live Pipeline DOES NOT auto-retrieve these new replies ---
    print("\n--- Safety Verification: Testing Live Draft Pipeline ---")
    test_q = generate_draft(["meri tug or pk show nahi ho raha hai please help"], conv_id="safety_check")
    print(f"Live Query Status: {test_q.get('status')}")
    print(f"Live Query Draft : {test_q.get('draft')}")
    print(f"Sources Used     : {test_q.get('used')}")
    assert "audition period" not in test_q.get("draft", ""), "FAILED: Unreviewed reply leaked into live retrieval!"
    print("PASS: Unreviewed agent replies are strictly excluded from live customer drafting!")

    # --- 4. Weekly Review: Generate Proposals from Feedback ---
    print("\n--- Weekly Review: Building Proposals from Pending Agent Edits ---")
    proposed_count = review.build_report()
    print(f"Proposals generated: {proposed_count}")

    proposals = store.pending_proposals()
    print(f"\nTotal Pending Proposals in Admin Review: {len(proposals)}")
    for p in proposals:
        print("\n" + "="*60)
        print(f"PROPOSAL ID: {p['id']} | ACTION: {p['action']} | TOPIC: {p['part']}")
        print(f"Title (Question): {p['title']}")
        print(f"Audience        : {p['audience']}")
        print(f"Body (Solution) : {p['body']}")
        print(f"Keywords        : {p.get('keywords')}")
        print(f"Old Text        : {p.get('old_text') or '(None - New Addition)'}")
        print("="*60)

if __name__ == "__main__":
    main()
