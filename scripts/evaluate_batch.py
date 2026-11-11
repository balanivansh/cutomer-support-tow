"""Test runner for evaluating representative support questions:
- Multi-message cases
- Follow-up cases
- Single diverse queries
Pauses briefly between calls to respect Groq free-tier rate limits.
"""
import sys
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline import generate_draft

CASES = [
    # --- MULTI-MESSAGE CASES ---
    {
        "id": "Q1 (Multi-message)",
        "type": "multi_message",
        "messages": [
            "Customer: issues ye ki hme sirf ek hi tug of war ka chance diye he",
            "Customer: to fir hme baki ke 3 tug war kese complete kre ??",
            "Customer: chance hi nhi honga 1 tug ke bad 2 tug start krne ka to Mera task kese complete kr paunga me tug war ???? 1/ 1 chance complete he to baki ke 3 tug out of 4 se kese complete kre??"
        ]
    },
    {
        "id": "Q8 (Multi-message)",
        "type": "multi_message",
        "messages": [
            "Customer: hello mam...hamara credit etne din se ...2 hi aaya hai tug ka 3 month se live kr rahe hai ..na song recorment ho rahe hai hamare na tug credit wala 3 aaya hai...baki new id jo aa rahi hai..live unko 3 credit aaye hai fir batao...",
            "Customer: kya kare ham log...etne din se live kr rahe hai to...hamare pass to coine bhi nahi hote hai...to 4 tug kaise kare...kuch help kr dijiye taki live pr jyada dhan de ske or bhi accha karne ki koshish karenge.."
        ]
    },
    {
        "id": "Q17 (Multi-message)",
        "type": "multi_message",
        "messages": [
            "Customer: thank you 😊 meri tug ki recording recommendations pending me hai please check 🤩",
            "Customer: mere ko ek hi tug credit hai kab tak aayega 2 credit recording tug post aur recommendation bhi kar raha hu par mere ko 1 hi free credit hai 2 aur free credit kar dijiye"
        ]
    },

    # --- FOLLOW-UP CASES ---
    {
        "id": "Q3 (Follow-up)",
        "type": "follow_up",
        "messages": [
            "Customer: Hello Sir ji Mai kuch dino se health issues se guzar rhi thi, specially vocal problem. Aaj meri 3 free tug se 2 free tug ho gya.",
            "Agent: Aapke weekly recordings review ke hisaab se health heart yellow ho gaya hai.",
            "Customer: Kaise fir mujhe 3 free tug milega bataiye please."
        ]
    },
    {
        "id": "Q4 & Q5 (Follow-up)",
        "type": "follow_up",
        "messages": [
            "Customer: aaj tug war ka coin nhi aaya q aaj coin to aayenge na tug ke",
            "Agent: Winning gold coins match end hone ke thodi der baad credit ho jaate hain.",
            "Customer: hi tug me 63 hi mila coin aaj 81 milta hena"
        ]
    },
    {
        "id": "Q7 (Follow-up)",
        "type": "follow_up",
        "messages": [
            "Customer: hallo facing issues whenever going live its showing the tug of war is already on pls help",
            "Agent: Ek waqt par ek hi live stream ya match active ho sakta hai, please app restart karke check karein.",
            "Customer: restart kiya fir bhi already on show ho raha hai"
        ]
    },

    # --- SINGLE QUERIES (DIVERSE TOPICS) ---
    {
        "id": "Q2",
        "type": "single",
        "messages": [
            "Customer: Hello Team, I am doing Live on a daily basis and doing Tug of War but I have only 1 free credit on my live. please increase my Tug Credit 🙏 mere live pe 1 Tug ke kaaran koi rukta nahi hai, 3 Tug honge toh log rukenge bhi or enjoy bhi karenge 🙆♂️ please please please 🙏 mera Tug Credit 3 kar dijiye 🥺"
        ]
    },
    {
        "id": "Q6",
        "type": "single",
        "messages": [
            "Customer: aise tug me heart color green se yellow ho ja raha h sabhi ka"
        ]
    },
    {
        "id": "Q9",
        "type": "single",
        "messages": [
            "Customer: Mera tug ka recording nehi ho raha"
        ]
    },
    {
        "id": "Q10",
        "type": "single",
        "messages": [
            "Customer: main 4 song pehle tug recording karke recommend apply tha phir v red hai abhi sir"
        ]
    },
    {
        "id": "Q11",
        "type": "single",
        "messages": [
            "Customer: Good Morning Support Team Maine kal dekha tha ke bahut se logo ke tug me credits badhaye gaye hain even kuch log sirf 2-3 baar live aaye hain unko bhi 3 credit de diye gaye hain...aur me hamesha fair tug karate aaya hu aur meri health bhi green hai...Chaaro tug karata hu aur mujhe abhi bhi credit nahi diya gaya"
        ]
    },
    {
        "id": "Q12",
        "type": "single",
        "messages": [
            "Customer: mera tug last weekend par badhane k liye kaha tha aapne 🥺😭 abhi tak mera ek hi tug hai, kyu nahi badhaya ja raha 🙆♂️🥺 new broadcasters ka tug badh raha hai but mera nahi ☹️"
        ]
    },
    {
        "id": "Q13",
        "type": "single",
        "messages": [
            "Customer: yes boliye meri family ki ki ek id nmbr 29893 ka bahut mahine se ek hi tug hai daily tug karte hai wo. par unka tug aaj bhi 1 hi hai. kindly isko dhyan de. log unke tug pe nhi jaate kyonki 1 hi tug hai"
        ]
    },
    {
        "id": "Q14",
        "type": "single",
        "messages": [
            "Customer: broad caster ko live ke liye tow me gana diya he to participate ko bhi kuch do gane ke liye vese hi win hote he sir ji fac se jite he or real gane wale harte he"
        ]
    },
    {
        "id": "Q15",
        "type": "single",
        "messages": [
            "Customer: 4 tug karne ka bad bhi expi nahi aaya please check 🙏"
        ]
    },
    {
        "id": "Q16",
        "type": "single",
        "messages": [
            "Customer: as of now I use to get 20 tikit n 81 coin from tug n in the diamond event gifting"
        ]
    },
    {
        "id": "Q18",
        "type": "single",
        "messages": [
            "Customer: Dear Team, please mera 3 free tug karen, 3 paid tug karanq padta hai daily which is very difficult"
        ]
    },
    {
        "id": "Q19",
        "type": "single",
        "messages": [
            "Customer: meri tug or pk show nehi kor raha"
        ]
    },
    {
        "id": "Q20",
        "type": "single",
        "messages": [
            "Customer: hello respected sir , maam m tug daily karti hu or recording bhi karti hu meri adhi recording cover m chali jaati h pls aapne mera jo ek tug kam kiya meko de dijiye pls 🙂↕️ recording roz karti hu fir bhi pls"
        ]
    }
]

def run_evaluation():
    results = []
    print(f"Starting evaluation of {len(CASES)} cases...\n")
    for idx, c in enumerate(CASES, 1):
        print(f"[{idx}/{len(CASES)}] Testing {c['id']} ({c['type']})...")
        try:
            r = generate_draft(c["messages"], conv_id=f"eval_{idx}")
            q = r.get("query") or {}
            out = {
                "id": c["id"],
                "type": c["type"],
                "input_messages": c["messages"],
                "status": r.get("status"),
                "draft": r.get("draft"),
                "used_sources": r.get("used", []),
                "all_retrieved": [s["id"] for s in r.get("sources", [])],
                "english_query": q.get("english_query"),
                "intent": q.get("intent"),
                "role_guess": q.get("role_guess"),
                "language": q.get("language_and_script"),
            }
            results.append(out)
            print(f"    Status: {out['status']}")
            print(f"    Draft : {out['draft'][:100]}...")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "id": c["id"],
                "type": c["type"],
                "status": "ERROR",
                "error": str(e)
            })
        time.sleep(3)  # stay safely under Groq rate limits

    out_file = ROOT / "data" / "eval_results.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCompleted! Saved results to {out_file}")

if __name__ == "__main__":
    run_evaluation()
