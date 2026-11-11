"""Parse the support KB into chunks (one '###' heading = one chunk) and store them in Chroma.
Run:  python -m app.kb_ingest      (safe to re-run: it rebuilds the collection)"""
import re
import sys

from app.kb_common import (CHROMA_PATH, EMBEDDING_MODEL, KB_COLLECTION, KB_PATH,
                           embed, log_error, log_info)

PART_RE = re.compile(r"^##\s+(PART\s+[A-Z]):\s*(.*)$")
CHUNK_RE = re.compile(r"^###\s+([A-Z]\d+)\.\s*(.*)$")
AUD_RE = re.compile(r"^\*\*Audience:\*\*\s*(.+)$")


def role_flags(audience):
    a = audience.lower()
    everyone = "everyone" in a
    return ("broadcaster" in a) or everyone, ("participant" in a) or everyone


def parse_kb(text):
    chunks = []
    cur = None
    part_id, part_title = "", ""
    part_chunk_count = 0
    prelude = []

    def flush_chunk():
        nonlocal cur, part_chunk_count
        if cur is None:
            return
        body = "\n".join(cur["lines"]).strip()
        audience = cur["audience"] or "Everyone"
        fb, fp = role_flags(audience)
        chunks.append({
            "id": cur["id"],
            "text": f"{cur['title']}\n{body}".strip(),
            "meta": {"chunk_id": cur["id"], "title": cur["title"], "part": part_title,
                     "audience": audience, "for_broadcaster": fb, "for_participant": fp},
        })
        part_chunk_count += 1
        cur = None

    def flush_prelude():
        # A part with no '###' chunks (for example the time/reset table) becomes one chunk.
        body = "\n".join(prelude).strip()
        if part_id and part_chunk_count == 0 and body:
            pid = part_id.replace("PART ", "")
            chunks.append({
                "id": pid,
                "text": f"{part_title}\n{body}",
                "meta": {"chunk_id": pid, "title": part_title, "part": part_title,
                         "audience": "Everyone", "for_broadcaster": True, "for_participant": True},
            })

    for line in text.splitlines():
        s = line.rstrip()
        m = PART_RE.match(s)
        if m:
            flush_chunk()
            flush_prelude()
            part_id, part_title = m.group(1), m.group(2).strip()
            part_chunk_count = 0
            prelude = []
            continue
        m = CHUNK_RE.match(s)
        if m:
            flush_chunk()
            cur = {"id": m.group(1), "title": m.group(2).strip(), "audience": None, "lines": []}
            continue
        if s.strip() == "---":
            continue
        if cur is not None:
            a = AUD_RE.match(s.strip())
            if a and cur["audience"] is None:
                cur["audience"] = a.group(1).strip()
            else:
                cur["lines"].append(s)
        else:
            prelude.append(s)
    flush_chunk()
    flush_prelude()
    return chunks


def main():
    if not KB_PATH.exists():
        log_error("CONFIG_ERROR", "read_kb",
                  f"KB file not found at {KB_PATH}. Copy cust_supp_kb.md into data/ or fix KB_PATH in .env.")
        sys.exit(1)
    chunks = parse_kb(KB_PATH.read_text(encoding="utf-8"))
    ids = [c["id"] for c in chunks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        log_error("CONFIG_ERROR", "parse_kb", f"Duplicate chunk ids in KB: {sorted(dupes)}. Fix the KB headings.")
        sys.exit(1)
    if not chunks:
        log_error("CONFIG_ERROR", "parse_kb", "No chunks found. Check the KB uses '### B5. Title' style headings.")
        sys.exit(1)
    log_info("parse_kb", f"Parsed {len(chunks)} chunks from {KB_PATH.name}")

    try:
        vectors = embed([c["text"] for c in chunks])
    except Exception:
        sys.exit(1)

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            client.delete_collection(KB_COLLECTION)
        except Exception:
            pass
        col = client.create_collection(KB_COLLECTION, metadata={"hnsw:space": "cosine"})
        col.add(ids=ids, documents=[c["text"] for c in chunks],
                embeddings=vectors, metadatas=[c["meta"] for c in chunks])
        stored = col.count()
    except Exception as e:
        log_error("RETRIEVAL_ERROR", "store_chunks", f"Could not write to Chroma at {CHROMA_PATH}", e)
        sys.exit(1)

    log_info("store_chunks", f"Stored {stored} chunks in Chroma ({EMBEDDING_MODEL}, dim {len(vectors[0])})")
    for c in chunks:
        m = c["meta"]
        print(f"  {c['id']:<4} {m['audience']:<24} {len(c['text']):>5} chars  {m['title'][:60]}")
    if stored != len(chunks):
        log_error("RETRIEVAL_ERROR", "store_chunks", f"Expected {len(chunks)} chunks but Chroma holds {stored}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
