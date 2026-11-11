"""Hybrid retriever: Chroma dense search + BM25 keyword search, merged with reciprocal rank fusion.
Use:  r = Retriever();  r.search("English query", role="broadcaster", top_k=3)"""
import re

import chromadb
from rank_bm25 import BM25Okapi

from app.kb_common import CHROMA_PATH, KB_COLLECTION, embed, log_error, log_info

_SPLIT = re.compile(r"[\s\.,;:!?()\[\]{}\"'`*|#/\\<>=+~^%&@$\-–—।]+")


def tokenize(text):
    return [w for w in _SPLIT.split(text.lower()) if w]


class Retriever:
    def __init__(self):
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            self.col = client.get_collection(KB_COLLECTION)
            data = self.col.get(include=["documents", "metadatas"])
        except Exception as e:
            log_error("RETRIEVAL_ERROR", "open_collection",
                      "KB collection not found or unreadable. Run: python -m app.kb_ingest", e)
            raise
        self.ids = data["ids"]
        self.docs = data["documents"]
        self.metas = data["metadatas"]
        self.pos = {i: n for n, i in enumerate(self.ids)}
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs])
        log_info("open_collection", f"Retriever ready with {len(self.ids)} chunks")

    @staticmethod
    def _allowed(meta, role):
        if role == "broadcaster":
            return meta.get("for_broadcaster", True)
        if role == "participant":
            return meta.get("for_participant", True)
        return True

    def search(self, query, role=None, top_k=3, n_candidates=10, conv_id=None, query_vector=None):
        """role: 'broadcaster', 'participant' or None (no filter). Returns a list of dicts.
        Optional query_vector avoids duplicate embedding calculations when already computed."""
        tag = f"conv={conv_id} " if conv_id else ""
        try:
            qv = query_vector if query_vector is not None else embed([query])[0]
            where = None
            if role == "broadcaster":
                where = {"for_broadcaster": True}
            elif role == "participant":
                where = {"for_participant": True}
            res = self.col.query(query_embeddings=[qv],
                                 n_results=min(n_candidates, len(self.ids)),
                                 where=where, include=["distances"])
        except Exception as e:
            log_error("RETRIEVAL_ERROR", "dense_search", f"{tag}Dense search failed", e)
            raise
        dense_ids = res["ids"][0]
        dense_sim = {i: 1.0 - d for i, d in zip(dense_ids, res["distances"][0])}

        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(((s, n) for n, s in enumerate(scores)
                         if s > 0 and self._allowed(self.metas[n], role)), reverse=True)[:n_candidates]
        bm_ids = [self.ids[n] for _, n in ranked]
        bm_score = {self.ids[n]: float(s) for s, n in ranked}

        rrf = {}
        for rank, i in enumerate(dense_ids):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank + 1)
        for rank, i in enumerate(bm_ids):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank + 1)
        order = sorted(rrf, key=rrf.get, reverse=True)[:top_k]

        out = []
        for i in order:
            m = self.metas[self.pos[i]]
            out.append({"id": i, "title": m["title"], "audience": m.get("audience", "Everyone"),
                        "text": self.docs[self.pos[i]], "dense_sim": dense_sim.get(i),
                        "bm25": bm_score.get(i), "rrf": rrf[i]})
        log_info("search", f"{tag}query='{query[:60]}' role={role} -> {[r['id'] for r in out]}")
        return out
