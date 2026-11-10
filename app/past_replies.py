"""Past approved replies (stage 4b). Agent replies saved in the SQLite `feedback` table are embedded
(by the clean English query, so they match the pipeline's search query) into the Chroma collection
`past_replies`. Past replies are optional: any failure here returns [] and drafting continues.
Rebuild everything from SQLite:  python -m app.past_replies"""
import os

import chromadb

from app import store
from app.kb_common import CHROMA_PATH, embed, log_error, log_info

PAST_COLLECTION = "past_replies"
PAST_SIM_THRESHOLD = float(os.getenv("PAST_SIM_THRESHOLD") or 0.75)  # min similarity of English queries
PAST_TOP_K = int(os.getenv("PAST_TOP_K") or 2)

_col = None


def _collection():
    global _col
    if _col is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _col = client.get_or_create_collection(PAST_COLLECTION, metadata={"hnsw:space": "cosine"})
    return _col


def _should_index(row):
    """Save only what an agent changed or wrote. An unchanged (accepted) reply adds nothing new:
    it is either a KB answer, a holding reply, or a copy of an earlier past reply."""
    if not (row["final_text"] or "").strip():
        return False
    if row["draft_status"] == "CLARIFY":
        return False  # answers to vague messages teach nothing
    return row["outcome"] != "accepted"


def index_pending():
    """Index feedback rows with indexed=0. Marks them 1 (indexed) or 2 (skipped). Returns how many were indexed."""
    rows = store.pending_feedback()
    if not rows:
        return 0
    keep, skip = [], []
    for r in rows:
        key = (r["draft_query"].get("english_query") or r["customer_text"] or "").strip()
        (keep if key and _should_index(r) else skip).append((r, key))
    indexed = 0
    if keep:
        try:
            vecs = embed([k for _, k in keep])
            _collection().upsert(
                ids=[f"f{r['id']}" for r, _ in keep],
                embeddings=vecs,
                documents=[r["final_text"] for r, _ in keep],
                metadatas=[{"question": r["customer_text"] or "", "english_query": k, "outcome": r["outcome"],
                            "role": r["draft_query"].get("role_guess") or "unknown",
                            "created_at": r["created_at"], "conv_id": r["conv_id"]} for r, k in keep])
        except Exception as e:
            log_error("RETRIEVAL_ERROR", "past_index", "Could not index past replies; will retry next time", e)
            return 0
        store.mark_feedback_indexed([r["id"] for r, _ in keep], 1)
        indexed = len(keep)
    store.mark_feedback_indexed([r["id"] for r, _ in skip], 2)
    log_info("past_index", f"Indexed {indexed} past repl{'y' if indexed == 1 else 'ies'}, skipped {len(skip)}")
    return indexed


def search(english_query, conv_id=None):
    """Returns up to PAST_TOP_K [{id, question, reply, sim, outcome, created_at}], best first."""
    tag = f"conv={conv_id} " if conv_id else ""
    try:
        col = _collection()
        n = col.count()
        if n == 0:
            return []
        res = col.query(query_embeddings=[embed([english_query])[0]], n_results=min(PAST_TOP_K * 4, n),
                        include=["documents", "metadatas", "distances"])
    except Exception as e:
        log_error("RETRIEVAL_ERROR", "past_search", f"{tag}Past-reply search failed; continuing without it", e)
        return []
    out = []
    best = 0.0
    active = store.active_feedback_ids(res["ids"][0])  # merged / rejected replies are ignored
    for i, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if i not in active:
            continue
        sim = 1.0 - dist
        best = max(best, sim)
        if sim >= PAST_SIM_THRESHOLD:
            out.append({"id": i, "question": meta.get("question", ""), "english_query": meta.get("english_query", ""),
                        "reply": doc, "sim": sim,
                        "outcome": meta.get("outcome", ""), "created_at": meta.get("created_at", "")})
    out = out[:PAST_TOP_K]
    log_info("past_search", f"{tag}query='{english_query[:60]}' best_sim={best:.2f} (needs {PAST_SIM_THRESHOLD}) "
             f"-> {[(o['id'], round(o['sim'], 2)) for o in out]}")
    return out


def rebuild():
    """Drop the collection and re-index every feedback row from SQLite."""
    global _col
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(PAST_COLLECTION)
    except Exception:
        pass
    _col = None
    store.reset_feedback_index()
    return index_pending()


if __name__ == "__main__":
    store.init_db()
    print(f"Rebuilt past_replies index: {rebuild()} repl(ies) indexed")
